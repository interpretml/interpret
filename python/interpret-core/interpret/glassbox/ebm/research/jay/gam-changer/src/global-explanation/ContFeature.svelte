<script>
  import * as d3 from 'd3';
  import { onMount, onDestroy } from 'svelte';

  import { initIsotonicRegression } from '../isotonic-regression';

  import { round } from '../utils/utils';
  import { config } from '../config';
  import { drawLineLegend } from './draw';

  import { SelectedInfo } from './continuous/cont-class';
  import { createConfidenceData, createAdditiveData, createPointData, linkPointToAdditive } from './continuous/cont-data';
  import { brushDuring, brushEndSelect, selectAllBins, quitSelection } from './continuous/cont-brush';
  import { zoomStart, zoomEnd, zoomed, zoomScaleExtent, rExtent } from './continuous/cont-zoom';
  import { dragged, redrawOriginal, redrawMonotone, inplaceInterpolate,
    stepInterpolate, merge, drawLastEdit, regressionInterpolate } from './continuous/cont-edit';
  import { moveMenubar } from './continuous/cont-bbox';
  import { undoHandler, redoHandler, tryRestoreLastEdit,
    pushCurStateToHistoryStack, checkoutCommitHead } from './continuous/cont-history';

  import ContextMenu from '../components/ContextMenu.svelte';

  export let featureData = null;
  export let scoreRange = null;
  export let svgHeight = 400;
  export let ebm = null;
  export let sidebarStore = null;
  export let footerStore = null;
  export let footerActionStore = null;
  export let historyStore = null;

  let svg = null;
  let component = null;
  let multiMenu = null;
  let myContextMenu = null;
  let redoStack = [];

  let mounted = false;
  let initialized = false;

  // Visualization constants
  const svgPadding = config.svgPadding;
  const densityHeight = 90;

  // Viewbox width and height
  const width = 600;
  const height = 400;

  // Real SVG width
  let svgWidth = svgHeight * (width / height);

  // Some constant lengths of different elements
  let yAxisWidth;
  let lineChartWidth;
  const lineChartHeight = height - svgPadding.top - svgPadding.bottom - densityHeight;

  // Some styles
  const colors = config.colors;
  const defaultFont = config.defaultFont;
  const linePathWidth = 2.5;
  const bboxStrokeWidth = 1;
  const nodeStrokeWidth = 1;

  // --- Interactions ---
  // Brush interactions
  let brush = null;
  let initXDomain = null;
  let initYDomain = null;

  // Panning and zooming
  let zoom = null;

  // Select mode
  let state = {
    curXScale: null,
    curYScale: null,
    curTransform: null,
    selectedInfo: null,
    lastSelectedInfo: null,
    pointData: null,
    additiveData: null,
    pointDataBuffer: null,
    additiveDataBuffer: null,
    oriXScale: null,
    oriYScale: null,
    bboxPadding: 1,
  };
  let selectMode = false;
  state.selectedInfo = new SelectedInfo();

  // Editing mode

  // Context menu info
  let multiMenuControlInfo = {
    moveMode: false,
    toSwitchMoveMode: false,
    subItemMode: null,
    setValue: null,
    step: 3,
    interpolationMode: 'inplace',
  };

  // Isotonic regression
  let increasingISO = null;
  let decreasingISO = null;

  // Subscribe the history store
  let historyList = null;
  let historyStoreUnsubscribe = historyStore.subscribe(value => {historyList = value;});

  // Communicate with the sidebar
  let sidebarInfo = {};
  let sidebarStoreUnsubscribe = sidebarStore.subscribe(async value => {
    sidebarInfo = value;

    // Listen to events ['globalClicked', 'selectedClicked', 'sliceClicked']
    // from the sidebar
    switch(value.curGroup) {
    case 'globalClicked':
      console.log('globalClicked');

      footerStore.update(value => {
        if (value.sample.includes(',')) {
          value.sample = value.sample.slice(0, -1);
        }
        value.slice = '';
        return value;
      });

      // We keep track of the global metrics in history in any current scope
      // To restore the global tab, we just need to query the history stack
      sidebarInfo.curGroup = 'overwrite';
      sidebarStore.set(sidebarInfo);
      break;

    case 'selectedClicked':
      console.log('selectedClicked');

      footerStore.update(value => {
        if (value.sample.includes(',')) {
          value.sample = value.sample.slice(0, -1);
        }
        value.slice = '';
        return value;
      });

      // Step 1: If there is no selected nodes, then the metrics are all NAs
      if (!state.selectedInfo.hasSelected) {
        sidebarInfo.curGroup = 'nullify';
        sidebarStore.set(sidebarInfo);
      } else {
        // Step 2: Reset/Update EBM 3 times and compute three metrics on the selected nodes

        // Step 2.1: Original
        // Here we reset the EBM model completely, because
        // the intermediate historical events might update() different portions
        // of the EBM
        // Be careful! The first commit might be on a different feature!
        // It is way too complicated to load the initial edit then come back (need to revert
        // every edit on every feature!)
        // Here we just use ignore it [better than confusing the users with some other "original"]
        // await setEBM('original-only', historyList[0].state.pointData);

        // Nullify the original
        sidebarInfo.curGroup = 'nullify';
        sidebarStore.set(sidebarInfo);
        while (sidebarInfo.curGroup !== 'nullifyCompleted') {
          await new Promise(r => setTimeout(r, 300));
        }

        // Step 2.2: Last edit
        if (sidebarInfo.historyHead - 1 >= 0 &&
          historyList[sidebarInfo.historyHead - 1].type !== 'original' &&
          historyList[sidebarInfo.historyHead - 1].featureName === state.featureName) {
          await setEBM('last-only', historyList[sidebarInfo.historyHead - 1].state.pointData);
        }

        // Step 2.3: Current edit
        let curPointData = state.pointDataBuffer === null ?
          historyList[sidebarInfo.historyHead].state.pointData :
          state.pointDataBuffer;

        await setEBM('current-only', curPointData);
      }

      break;

    case 'sliceClicked': {
      console.log('sliceClicked');

      // Step 1: set the slice feature ID and level ID to EBM
      let sliceSize = ebm.setSliceData(sidebarInfo.sliceInfo.featureID, sidebarInfo.sliceInfo.level);

      footerStore.update(value => {
        if (!value.sample.includes(',')) value.sample += ',';
        value.slice = `<b>${sliceSize}</b> sliced`;
        return value;
      });

      // Step 2: Reset/Update EBM 3 times and compute three metrics on the selected nodes

      // Step 2.1: Original
      // Here we reset the EBM model completely, because
      // the intermediate historical events might update() different portions
      // of the EBM
      // await setEBM('original-only', historyList[0].state.pointData);

      // Nullify the original
      sidebarInfo.curGroup = 'nullify';
      sidebarStore.set(sidebarInfo);
      while (sidebarInfo.curGroup !== 'nullifyCompleted') {
        await new Promise(r => setTimeout(r, 300));
      }

      // Step 2.2: Last edit
      if (sidebarInfo.historyHead - 1 >= 0 &&
        historyList[sidebarInfo.historyHead - 1].type !== 'original' &&
        historyList[sidebarInfo.historyHead - 1].featureName === state.featureName) { 
        await setEBM('last-only', historyList[sidebarInfo.historyHead - 1].state.pointData);
      }

      // Step 2.3: Current edit
      let curPointData = state.pointDataBuffer === null ?
        historyList[sidebarInfo.historyHead].state.pointData :
        state.pointDataBuffer;

      await setEBM('current-only', curPointData);

      break;
    }

    // User clicks to preview a previous edit
    case 'headChanged': {
      const headFeatureName = historyList[value.historyHead].featureName;
      // Only checkout the commit if it is still on the same feature
      // Otherwise, this component should wait for its parent to kill it
      if (headFeatureName === state.featureName) {
        checkoutCommitHead(state, svg, multiMenu, resetContextMenu, resetFeatureSidebar,
          historyStore, setEBM, setEBMEditingFeature, sidebarStore);
      }
      break;
    }

    default:
      break;
    }
  });

  let footerValue = null;
  footerStore.subscribe(value => {
    footerValue = value;
  });

  // Listen to footer buttons
  let footerActionUnsubscribe = footerActionStore.subscribe(message => {
    switch(message){
    case 'undo': {
      console.log('undo clicked');

      if (historyList.length > 1) {
        undoHandler(state, svg, multiMenu, resetContextMenu, resetFeatureSidebar,
          historyStore, redoStack, setEBM, sidebarStore);
      }
      break;
    }

    case 'redo': {
      console.log('redo clicked');

      if (redoStack.length > 0) {
        redoHandler(state, svg, multiMenu, resetContextMenu, resetFeatureSidebar,
          historyStore, redoStack, setEBM, sidebarStore);
      }
      break;
    }
    
    case 'save':
      console.log('save clicked');
      break;

    case 'selectAll':
      console.log('selectAll clicked');

      // Select all bins if in select mode and nothing has been selected yet
      if (selectMode) {
        // Discard any existing selection
        quitSelection(svg, state, multiMenu, resetContextMenu, resetFeatureSidebar);

        // Cheeky way to select all nodes by fake a brush event
        selectAllBins(svg, state, bboxStrokeWidth, multiMenu, component,
          updateFeatureSidebar, nullifyMetrics, computeSelectedEffects, brush);
      }
      break;
    
    default:
      break;
    }

    footerActionStore.set('');
  });

  onMount(() => {
    mounted = true;
  });

  // Need to unsubscribe stores when the component is destroyed
  onDestroy(() => {
    sidebarStoreUnsubscribe();
    footerActionUnsubscribe();
    historyStoreUnsubscribe();
  });

  /**
   * Draw the plot in the SVG component
   * @param featureData
   */
  const drawFeature = async (featureData) => {
    initialized = true;
    console.log(featureData);

    // Track the feature name
    state.featureName = featureData.name;

    // Approximate the longest width of score (y-axis)
    yAxisWidth = 5 * d3.max(scoreRange.map(d => String(round(d, 1)).length));
    lineChartWidth = width - svgPadding.left - svgPadding.right - yAxisWidth;

    let svgSelect = d3.select(svg);

    // Initialize the isotonic regression model
    initIsoModel(featureData);

    // Set svg viewBox (3:2 WH ratio)
    svgSelect.attr('viewBox', '0 0 600 400')
      .attr('preserveAspectRatio', 'xMinYMin meet')
      .attr('width', svgWidth)
      .attr('height', svgHeight)
      // WebKit bug workaround (see https://bugs.webkit.org/show_bug.cgi?id=226683)
      .on('wheel', () => {});
    
    // Disable the default context menu when right click
    svgSelect.on('contextmenu', (event) => {
      event.preventDefault();
    });

    // Draw a legend for the line color
    drawLineLegend(svgSelect, width, svgPadding);

    let content = svgSelect.append('g')
      .attr('class', 'content')
      .attr('transform', `translate(${svgPadding.left}, ${svgPadding.top})`);

    // The bins have unequal length, and they are inner edges
    // Here we use the min and max values from the training set as our left and
    // right bounds on the x-axis (left most and right most edges)
    let xMin = featureData.binEdge[0];
    let xMax = featureData.binEdge[featureData.binEdge.length - 1];

    // For the y scale, it seems InterpretML presets the center at 0 (offset
    // doesn't really matter in EBM because we can modify intercept)
    // TODO: Provide interaction for users to change the center point
    // let yExtent = d3.extent(featureData.additive);

    let xScale = d3.scaleLinear()
      .domain([xMin, xMax])
      .range([0, lineChartWidth]);

    // Normalize the Y axis by the global score range
    let yScale = d3.scaleLinear()
      .domain(scoreRange)
      .range([lineChartHeight, 0]);
    
    state.oriXScale = xScale;
    state.oriYScale = yScale;
    state.curXScale = xScale;
    state.curYScale = yScale;

    // Use the # of ticks and y score range to set the default change unit for
    // the up and down in the context menu bar
    multiMenuControlInfo.changeUnit = round((scoreRange[1] - scoreRange[0]) / yScale.ticks().length, 4);
    
    // Store the initial domain for zooming
    initXDomain = [xMin, xMax];
    initYDomain = scoreRange; 

    // Create a data array by combining the bin edge and additive terms
    state.additiveData = createAdditiveData(featureData);

    // Create the confidence interval region
    let confidenceData = createConfidenceData(featureData);

    // Create a data array to draw nodes
    state.pointData = createPointData(featureData);

    // Link the point data and additive data (only need to call this function
    // we we initialize them from the data, no need to call it when we add new
    // bins in run time)
    linkPointToAdditive(state.pointData, state.additiveData);

    // Create histogram chart group
    let histChart = content.append('g')
      .attr('class', 'hist-chart-group');
    
    // Draw the line chart
    let lineChart = content.append('g')
      .attr('class', 'line-chart-group');

    let axisGroup = lineChart.append('g')
      .attr('class', 'axis-group');

    // Add a clip path to bound the lines (for zooming)
    lineChart.append('clipPath')
      .attr('id', `${featureData.name.replace(/\s/g, '')}-line-chart-clip`)
      .append('rect')
      .attr('width', lineChartWidth)
      .attr('height', lineChartHeight - 1);

    // For the histogram clip-path, need to carefully play around with the
    // transformation, the path should be in a static group; the group having
    // clip-path attr should be static. Therefore we apply the transformation to
    // histChart's child later.
    histChart.append('clipPath')
      .attr('id', `${featureData.name.replace(/\s/g, '')}-hist-chart-clip`)
      .append('rect')
      .attr('x', yAxisWidth)
      .attr('y', lineChartHeight)
      .attr('width', lineChartWidth)
      .attr('height', densityHeight);

    histChart.attr('clip-path', `url(#${featureData.name.replace(/\s/g, '')}-hist-chart-clip)`);
    
    let lineChartContent = lineChart.append('g')
      .attr('class', 'line-chart-content-group')
      .attr('clip-path', `url(#${featureData.name.replace(/\s/g, '')}-line-chart-clip)`)
      .attr('transform', `translate(${yAxisWidth}, 0)`);

    lineChartContent.append('rect')
      .attr('width', lineChartWidth)
      .attr('height', lineChartHeight)
      .style('opacity', 0);

    // Create a group to draw grids
    let gridGroup = lineChartContent.append('g')
      .attr('class', 'line-chart-grid-group');

    let confidenceGroup = lineChartContent.append('g')
      .attr('class', 'line-chart-confidence-group');

    let lineGroup = lineChartContent.append('g')
      .attr('class', 'line-chart-line-group real')
      .style('stroke', colors.line)
      .style('stroke-width', linePathWidth)
      .style('fill', 'none');

    // We draw the shape function with many line segments (path)
    lineGroup.selectAll('path')
      .data(state.additiveData, d => `${d.id}-${d.pos}`)
      .join('path')
      .attr('class', 'additive-line-segment')
      .attr('id', d => d.id)
      .attr('d', d => {
        return `M ${xScale(d.x1)}, ${yScale(d.y1)} L ${xScale(d.x2)} ${yScale(d.y2)}`;
      });

    lineChartContent.append('g')
      .attr('class', 'line-chart-line-group last-edit')
      .style('stroke', 'hsl(35, 100%, 85%)')
      .style('stroke-width', linePathWidth)
      .style('fill', 'none')
      .lower();

    lineGroup.clone(true)
      .classed('real', false)
      .style('stroke', 'hsl(0, 0%, 82%)')
      .lower();

    confidenceGroup.lower();
    gridGroup.lower();
    
    // Draw nodes for editing
    let nodeGroup = lineChartContent.append('g')
      .attr('class', 'line-chart-node-group')
      .style('visibility', 'hidden');
    
    nodeGroup.selectAll('circle')
      .data(Object.values(state.pointData), d => d.id)
      .join('circle')
      .attr('class', 'node')
      .attr('id', d => `node-${d.id}`)
      .attr('cx', d => xScale(d.x))
      .attr('cy', d => yScale(d.y))
      .attr('r', rExtent[0])
      .style('stroke-width', nodeStrokeWidth);
    
    // Draw the underlying confidence interval
    confidenceGroup.selectAll('rect')
      .data(confidenceData)
      .join('rect')
      .attr('class', 'confidence-rect')
      .attr('x', d => xScale(d.x1))
      .attr('y', d => yScale(d.y1))
      .attr('width', d => xScale(d.x2) - xScale(d.x1))
      .attr('height', d => yScale(d.y2) - yScale(d.y1))
      .style('fill', colors.lineConfidence)
      .style('opacity', 0.13);

    // Draw the line chart X axis
    let xAxisGroup = axisGroup.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(${yAxisWidth}, ${lineChartHeight})`)
      .call(d3.axisBottom(xScale));
    
    xAxisGroup.attr('font-family', defaultFont);
    
    // Draw the line chart Y axis
    let yAxisGroup = axisGroup.append('g')
      .attr('class', 'y-axis')
      .attr('transform', `translate(${yAxisWidth}, 0)`);
    
    yAxisGroup.call(d3.axisLeft(yScale));
    yAxisGroup.attr('font-family', defaultFont);

    yAxisGroup.append('g')
      .attr('class', 'y-axis-text')
      .attr('transform', `translate(${-yAxisWidth - 15}, ${lineChartHeight / 2}) rotate(-90)`)
      .append('text')
      .text('Score')
      .style('fill', 'black');

    // Draw the histograms at the bottom
    let histData = [];
    
    // Transform the count to frequency (percentage)
    let histCountSum = d3.sum(featureData.histCount);
    let histFrequency = featureData.histCount.map(d => d / histCountSum);

    for (let i = 0; i < histFrequency.length; i++) {
      histData.push({
        x1: featureData.histEdge[i],
        x2: featureData.histEdge[i + 1],
        height: histFrequency[i]
      });
    }

    let histYScale = d3.scaleLinear()
      .domain(d3.extent(histFrequency))
      .range([0, densityHeight]);

    // Draw the density histogram 
    let histChartContent = histChart.append('g')
      .attr('class', 'hist-chart-content-group')
      .attr('transform', `translate(${yAxisWidth}, ${lineChartHeight})`);

    histChartContent.selectAll('rect')
      .data(histData)
      .join('rect')
      .attr('class', 'hist-rect')
      .attr('x', d => xScale(d.x1))
      .attr('y', 0)
      .attr('width', d => xScale(d.x2) - xScale(d.x1))
      .attr('height', d => histYScale(d.height))
      .style('fill', colors.hist);
    
    // Draw a Y axis for the histogram chart
    let yAxisHistGroup = lineChart.append('g')
      .attr('class', 'y-axis')
      .attr('transform', `translate(${yAxisWidth}, ${lineChartHeight})`);
    
    yAxisHistGroup.call(
      d3.axisLeft(histYScale)
        .ticks(2)
    );

    yAxisHistGroup.attr('font-family', defaultFont);

    // Change 0.0 to 0
    yAxisHistGroup.selectAll('text')
      .style('fill', colors.histAxis)
      .filter((d, i, g) => d3.select(g[i]).text() === '0.0')
      .text('0');

    yAxisHistGroup.selectAll('path,line')
      .style('stroke', colors.histAxis);

    yAxisHistGroup.append('g')
      .attr('transform', `translate(${-yAxisWidth - 15}, ${densityHeight / 2}) rotate(-90)`)
      .append('text')
      .attr('class', 'y-axis-text')
      .text('Density')
      .style('fill', colors.histAxis);

    // Add brush
    brush = d3.brush()
      .on('end', e => brushEndSelect(
        e, state, svg, multiMenu, myContextMenu, bboxStrokeWidth, brush, component,
        resetContextMenu, sidebarStore, setEBM, updateEBM, updateFeatureSidebar,
        resetFeatureSidebar, nullifyMetrics, computeSelectedEffects
      ))
      .on('start brush', e => brushDuring(e, state, svg, multiMenu, ebm, footerStore))
      .extent([[0, 0], [lineChartWidth, lineChartHeight]])
      .filter((e) => {
        if (selectMode) {
          return e.button === 0;
        } else {
          return e.button === 2;
        }
      });

    let brushGroup = lineChartContent.append('g')
      .attr('class', 'brush')
      .call(brush);
    
    // Change the style of the select box
    brushGroup.select('rect.overlay')
      .attr('cursor', null);

    // Add panning and zooming
    zoom = d3.zoom()
      .scaleExtent(zoomScaleExtent)
      .on('zoom', e => zoomed(e, state, xScale, yScale, svg, linePathWidth,
        nodeStrokeWidth, yAxisWidth, lineChartWidth, lineChartHeight,
        multiMenu, component))
      .on('start', () => zoomStart(state, multiMenu))
      .on('end', () => zoomEnd(state, multiMenu))
      .filter(e => {
        if (selectMode) {
          return (e.type === 'wheel' || e.button === 2);
        } else {
          return (e.button === 0 || e.type === 'wheel');
        }
      });

    lineChartContent.call(zoom)
      .call(zoom.transform, d3.zoomIdentity);

    lineChartContent.on('dblclick.zoom', null);
    
    // Listen to double click to reset zoom
    lineChartContent.on('dblclick', () => {
      lineChartContent.transition('reset')
        .duration(750)
        .ease(d3.easeCubicInOut)
        .call(zoom.transform, d3.zoomIdentity);
    });

    // Update the footer for more instruction
    footerStore.update(value => {
      value.help = '<b>Drag</b> to pan view, <b>Scroll</b> to zoom';
      return value;
    });

    // Try to restore the last edit if possible
    let hasBeenCreated = await tryRestoreLastEdit(state, svg, multiMenu, resetContextMenu,
      resetFeatureSidebar, historyStore, redoStack, setEBM, sidebarStore);

    if (!hasBeenCreated) {
      // Push the initial state into the history stack
      pushCurStateToHistoryStack(state, 'original', 'Original graph', historyStore, sidebarStore);
    } else {
      pushCurStateToHistoryStack(state, 'original',
        `Reloaded commit ${hasBeenCreated.substring(0, 7)}`,
        historyStore, sidebarStore);
    }

    sidebarInfo.historyHead = historyList.length - 1;
    sidebarInfo.previewHistory = false;
    sidebarStore.set(sidebarInfo);

    // Use animation as a signifier for zoom affordance
    setTimeout(() => {
      lineChartContent.transition()
        .duration(400)
        .call(zoom.scaleTo, 0.95);
    }, 400);
  };

  const getEBMMetrics = async (scope='global') => {
    // Depending on the selected scope, we have different modes of getMetrics()
    let metrics;

    switch(scope) {
    case 'global':
      metrics = ebm.getMetrics();
      break;
    case 'selected': {
      let selectedBinIndexes = state.selectedInfo.nodeData.map(d => d.ebmID);
      metrics = ebm.getMetricsOnSelectedBins(selectedBinIndexes);
      break;
    }
    case 'slice':
      metrics = ebm.getMetricsOnSelectedSlice();
      break;
    default:
      break;
    }
    return metrics;
  };

  /**
   * Pass the metrics info to sidebar handler (classification or egression metrics tab)
   * @param metrics Metrics info from the EBM
   * @param curGroup Name of the message
   */
  const transferMetricToSidebar = (metrics, curGroup) => {
    if (ebm.isClassification) {
      sidebarInfo.accuracy = metrics.accuracy;
      sidebarInfo.rocAuc = metrics.rocAuc;
      sidebarInfo.balancedAccuracy = metrics.balancedAccuracy;
      sidebarInfo.confusionMatrix = metrics.confusionMatrix;
    } else {
      sidebarInfo.rmse = metrics.rmse;
      sidebarInfo.mae = metrics.mae;
    }

    sidebarInfo.curGroup = curGroup;

    sidebarStore.set(sidebarInfo);
  };

  const updateEBM = async (curGroup, nodeData=undefined) => {
    let changedBinIndexes = [];
    let changedScores = [];

    if (nodeData === undefined) {
      nodeData = state.selectedInfo.nodeData;
    }

    nodeData.forEach(d => {
      changedBinIndexes.push(d.ebmID);
      changedScores.push(d.y);
    });

    await ebm.updateModel(changedBinIndexes, changedScores);

    /**
     * Depending on the current scope, we have different metrics updating methods
     */
    switch(sidebarInfo.effectScope) {
    case 'global': {
      let metrics = await getEBMMetrics('global');

      // Pass the metrics to sidebar
      transferMetricToSidebar(metrics, curGroup);
      break;
    }
    case 'selected': {
      let metrics = await getEBMMetrics('selected');

      // Pass the metrics to sidebar
      transferMetricToSidebar(metrics, curGroup);
      break;
    }
    case 'slice': {
      let metrics = await getEBMMetrics('slice');

      // Pass the metrics to sidebar
      transferMetricToSidebar(metrics, curGroup);
      break;
    }
    case 'recoverEBM': {
      // No need to transfer the new metrics
      break;
    }
    default:
      break;
    }
  };

  /**
   * Overwrite the edge definition in the EBM WASM model.
   * @param {string} curGroup Message to the metrics sidebar
   * @param {object} curNodeData Node data in `state`
   * @param {featureName} featureName The name of feature to be edited
   * @param {bool} transfer If the new metrics need to be transferred to the sidebar
   */
  const setEBM = async (curGroup, curNodeData, featureName=undefined, transfer=true) => {

    // Update the complete bin edge definition in the EBM model
    let newBinEdges = [];
    let newScores = [];

    // The left point will always have index 0
    let curPoint = curNodeData[0];
    let curEBMID = 0;

    while (curPoint.rightPointID !== null) {
      // Collect x and y
      newBinEdges.push(curPoint.x);
      newScores.push(curPoint.y);

      // Update the new ID so we can map them to bin indexes later (needed for
      // selection to check sample number)
      curPoint.ebmID = curEBMID;
      curEBMID++;

      curPoint = curNodeData[curPoint.rightPointID];
    }

    // Add the right node
    newBinEdges.push(curPoint.x);
    newScores.push(curPoint.y);
    curPoint.ebmID = curEBMID;

    await ebm.setModel(newBinEdges, newScores, featureName);

    if (transfer) {
      switch(sidebarInfo.effectScope) {
      case 'global': {
        let metrics = await getEBMMetrics('global');
        transferMetricToSidebar(metrics, curGroup);
        break;
      }
      case 'selected': {
        let metrics = await getEBMMetrics('selected');
        transferMetricToSidebar(metrics, curGroup);
        break;
      }
      case 'slice': {
        let metrics = await getEBMMetrics('slice');
        transferMetricToSidebar(metrics, curGroup);
        break;
      }
      default:
        break;
      }
    }
  };

  /**
   * Change the currently editing feature in ebm wasm
  */
  const setEBMEditingFeature = (featureName) => {
    ebm.setEditingFeature(featureName);
  };

  /**
   * Set all metrics to null if there is no selection and the scope is 'selected'.
  */
  const nullifyMetrics = () => {
    if (!state.selectedInfo.hasSelected && sidebarInfo.effectScope === 'selected') {
      sidebarInfo.curGroup = 'nullify';
      sidebarStore.set(sidebarInfo);
    }
  };

  const computeSelectedEffects = async () => {
    if (sidebarInfo.effectScope === 'selected' && state.selectedInfo.hasSelected) {
      // Step 1: compute the original metrics
      // Be careful! The first commit might be on a different feature!
      // It is way too complicated to load the initial edit then come back (need to revert
      // every edit on every feature!)
      // Here we just use ignore it [better than confusing the users with some other "original"]
      // if (historyList[0].featureName !== state.featureName) {
      //   ebm.setEditingFeature(historyList[0].featureName);
      // }
      // await setEBM('original-only', historyList[0].state.pointData);
      // ebm.setEditingFeature(state.featureName);

      // Nullify the original
      sidebarInfo.curGroup = 'nullify';
      sidebarStore.set(sidebarInfo);
      while (sidebarInfo.curGroup !== 'nullifyCompleted') {
        await new Promise(r => setTimeout(r, 300));
      }

      // Step 2: Last edit
      if (sidebarInfo.historyHead - 1 >= 0 &&
        historyList[sidebarInfo.historyHead - 1].type !== 'original' &&
        historyList[sidebarInfo.historyHead - 1].featureName === state.featureName) {
        await setEBM('last-only', historyList[sidebarInfo.historyHead - 1].state.pointData);
      }

      // Step 3: Current edit
      await setEBM('current-only', historyList[sidebarInfo.historyHead].state.pointData);
    }
  };

  /**
   * Count the feature distribution for the selected test samples
   * @param {[number]} binIndexes Selected bin indexes
   */
  const updateFeatureSidebar = async (binIndexes) => {
    if (ebm.isDummy !== undefined) return;

    // Get the selected counts
    let selectedHistCounts = ebm.getSelectedSampleDist(binIndexes);

    // Update the counts in the store
    for (let i = 0; i < sidebarInfo.featurePlotData.cont.length; i++) {
      let curID = sidebarInfo.featurePlotData.cont[i].id;
      sidebarInfo.featurePlotData.cont[i].histSelectedCount = selectedHistCounts[curID];
    }

    for (let i = 0; i < sidebarInfo.featurePlotData.cat.length; i++) {
      let curID = sidebarInfo.featurePlotData.cat[i].id;
      sidebarInfo.featurePlotData.cat[i].histSelectedCount = selectedHistCounts[curID];
    }

    sidebarInfo.curGroup = 'updateFeature';
    sidebarStore.set(sidebarInfo);
  };

  /**
   * Reset the feature count of selected samples to 0
   */
  const resetFeatureSidebar = async () => {
    if (ebm.isDummy !== undefined) return;

    for (let i = 0; i < sidebarInfo.featurePlotData.cont.length; i++) {
      sidebarInfo.featurePlotData.cont[i].histSelectedCount = new Array(
        sidebarInfo.featurePlotData.cont[i].histSelectedCount.length).fill(0);
    }

    for (let i = 0; i < sidebarInfo.featurePlotData.cat.length; i++) {
      sidebarInfo.featurePlotData.cat[i].histSelectedCount = new Array(
        sidebarInfo.featurePlotData.cat[i].histSelectedCount.length).fill(0);
    }

    sidebarInfo.curGroup = 'updateFeature';
    sidebarStore.set(sidebarInfo);
  };

  // ---- Interaction Functions ----

  /**
   * Quit the sub-menu mode (move, sub-item in the context menu) when user clicks
   * the empty space during editing
   * This function is implemented as a callback for brushSelected() because it
   * needs access to variable `multiMenuControlInfo`
   */
  const resetContextMenu = () => {
    let moveMode = multiMenuControlInfo.moveMode;
    let subItemMode = multiMenuControlInfo.subItemMode;

    if (multiMenuControlInfo.moveMode) {
      multiMenuControlInfo.moveMode = false;
      multiMenuControlInfo.toSwitchMoveMode = true;

      // DO not update the data
      state.pointDataBuffer = null;
      state.additiveDataBuffer = null;
    }

    // End sub-menu mode
    if (multiMenuControlInfo.subItemMode !== null) {
      // Hide the confirmation panel
      myContextMenu.hideConfirmation(multiMenuControlInfo.subItemMode);
      multiMenuControlInfo.subItemMode = null;

      // Discard changes
      state.pointDataBuffer = null;
      state.additiveDataBuffer = null;
    }

    multiMenuControlInfo.setValue = null;

    // Update the footer message
    footerStore.update(value => {
      // Reset the baseline
      value.baseline = 0;
      value.baselineInit = false;
      value.state = '';
      value.help = '<b>Drag</b> to marquee select, <b>Scroll</b> to zoom';
      return value;
    });

    return {moveMode: moveMode, subItemMode: subItemMode};
  };

  /**
   * Event handler for the select button in the header
   */
  export const selectModeSwitched = () => {
    selectMode = !selectMode;

    let lineChartContent = d3.select(svg)
      .select('g.line-chart-content-group')
      .classed('select-mode', selectMode);
    
    lineChartContent.select('g.brush rect.overlay')
      .attr('cursor', null);

    // Update the footer message
    if (selectMode) {
      footerStore.update(value => {
        value.help = '<b>Drag</b> to marquee select, <b>Scroll</b> to zoom';
        return value;
      });
    } else {
      footerStore.update(value => {
        value.help = '<b>Drag</b> to pan view, <b>Scroll</b> to zoom';
        return value;
      });
    }
  };

  const initIsoModel = async () => {
    increasingISO = await initIsotonicRegression(true);
    decreasingISO = await initIsotonicRegression(false);
  };

  const multiMenuMoveClicked = async () => {
    // Enter the move mode

    // If users have done some other edits without committing, discard the changes
    multiMenuSubItemCancelClicked(null, true);

    // Step 1. create data clone buffers for user to change
    // We only do this when buffer has not been created --- it is possible that
    // user switch to move from other editing mode
    if (state.pointDataBuffer === null) {
      state.pointDataBuffer = JSON.parse(JSON.stringify(state.pointData));
      state.additiveDataBuffer = JSON.parse(JSON.stringify(state.additiveData));
    }

    let bboxGroup = d3.select(svg)
      .select('g.line-chart-content-group g.select-bbox-group')
      .style('cursor', 'row-resize')
      .call(d3.drag()
        .on('start', () => {
          footerStore.update(value => {
            if (!value.baselineInit) {
              value.baseline = 0;
              value.baselineInit = true;
            }
            return value;
          });
        })
        .on('drag', (e) => dragged(e, state, svg, sidebarInfo.totalSampleNum > 2000,
          footerStore, updateEBM))
      );
    
    bboxGroup.select('rect.original-bbox')
      .classed('animated', true);
    
    // Show the last edit
    if (state.additiveDataLastEdit !== undefined) {
      drawLastEdit(state, svg);
    }

    // Copy current metrics as last metrics
    sidebarInfo.curGroup = 'last';
    sidebarStore.set(sidebarInfo);

    // Update the footer message
    footerStore.update(value => {
      value.help = '<b>Drag</b> the <b>selected region</b> to change score';
      return value;
    });
  };

  /**
   * Call this handler when users click the check button
   */
  const multiMenuMoveCheckClicked = async () => {
    // Check if user is in previous commit
    if (sidebarInfo.previewHistory) {
      let proceed = confirm('Current graph is not on the latest edit, committing' +
        ' this edit would overwrite all later edits on this feature. Is it OK?'
      );
      if (!proceed) {
        multiMenuMoveCancelClicked();
        return;
      }
    }

    // Save the changes
    state.pointData = JSON.parse(JSON.stringify(state.pointDataBuffer));
    state.additiveData = JSON.parse(JSON.stringify(state.additiveDataBuffer));

    // Remove the drag
    let bboxGroup = d3.select(svg)
      .select('g.line-chart-content-group g.select-bbox-group')
      .style('cursor', null)
      .on('.drag', null);
    
    // stop the animation
    bboxGroup.select('rect.original-bbox')
      .classed('animated', false);

    // Move the menu bar
    d3.select(multiMenu)
      .call(moveMenubar, svg, component);

    // Save this change to lastEdit, update lastEdit graph
    if (state.additiveDataLastEdit !== undefined) {
      state.additiveDataLastLastEdit = JSON.parse(JSON.stringify(state.additiveDataLastEdit));
    }
    state.additiveDataLastEdit = JSON.parse(JSON.stringify(state.additiveData));

    // Update metrics
    sidebarInfo.curGroup = 'commit';
    sidebarStore.set(sidebarInfo);

    // Query the global metrics and save it in the history if the scope is not in global
    if (sidebarInfo.effectScope !== 'global') {
      let metrics = await getEBMMetrics('global');
      transferMetricToSidebar(metrics, 'commit-not-global');
    }

    // Wait until the the effect sidebar is updated
    if (ebm.isDummy === undefined) {
      while (sidebarInfo.curGroup !== 'commitCompleted') {
        await new Promise(r => setTimeout(r, 500));
      }
    }

    // Update the footer message
    let curEditBaseline = 0;
    footerStore.update(value => {
      // Reset the baseline
      curEditBaseline = value.baseline;
      value.baseline = 0;
      value.baselineInit = false;
      value.type = '';
      value.state = '';
      value.help = '<b>Drag</b> to marquee select, <b>Scroll</b> to zoom';
      return value;
    });

    // Save into the history
    // Generate the description message
    const binNum = state.selectedInfo.nodeData.length;
    const binLeft = state.selectedInfo.nodeData[0];
    const binRight = state.pointData[state.pointData[state.selectedInfo.nodeData[binNum - 1].id].rightPointID];
    const binRange = binRight === undefined ? `${round(binLeft.x, 2)} <= x` : `${round(binLeft.x, 2)} <= x < ${round(binRight.x, 2)}`;
    const message = `${curEditBaseline >= 0 ? 'Increased' : 'Decreased'} scores of ${binNum} ` +
      `bins (${binRange}) by ${round(Math.abs(curEditBaseline), 2)}.`;

    pushCurStateToHistoryStack(state, 'move', message, historyStore, sidebarStore);

    // Any new commit purges the redo stack
    redoStack = [];
  };

  /**
   * Call this handler when users click the cancel button
   */
  const multiMenuMoveCancelClicked = () => {
    // Discard the changes
    state.pointDataBuffer = null;
    state.additiveDataBuffer = null;

    // Recover the original graph
    redrawOriginal(state, svg, true, () => {
      // Move the menu bar after animation
      d3.select(multiMenu)
        .call(moveMenubar, svg, component);

      // Recover the EBM
      updateEBM('recoverEBM');
    });

    // Remove the drag
    let bboxGroup = d3.select(svg)
      .select('g.line-chart-content-group g.select-bbox-group')
      .style('cursor', null)
      .on('.drag', null);
    
    // stop the animation
    bboxGroup.select('rect.original-bbox')
      .classed('animated', false);
    
    // Redraw the last edit if possible
    if (state.additiveDataLastLastEdit !== undefined){
      state.additiveDataLastEdit = JSON.parse(JSON.stringify(state.additiveDataLastLastEdit));
      drawLastEdit(state, svg);
      // Prepare for next redrawing after recovering the last last edit graph
      state.additiveDataLastEdit = JSON.parse(JSON.stringify(state.additiveData));
    }

    // Update the metrics
    sidebarInfo.curGroup = 'recover';
    sidebarStore.set(sidebarInfo);

    // Update the footer message
    footerStore.update(value => {
      // Reset the baseline
      value.baseline = 0;
      value.baselineInit = false;
      value.state = '';
      value.type = '';
      value.help = '<b>Drag</b> to marquee select, <b>Scroll</b> to zoom';
      return value;
    });
  };

  /**
   * Event handler when user clicks the increasing button
  */
  const multiMenuIncreasingClicked = async () => {
    console.log('increasing clicked');

    // Animate the bbox
    d3.select(svg)
      .select('g.line-chart-content-group g.select-bbox-group')
      .select('rect.original-bbox')
      .classed('animated', true);

    // Check if the selected nodes are in a continuous range

    // Fit an isotonic regression model
    let xs = [];
    let ys = [];
    let ws = [];

    state.selectedInfo.nodeData.forEach((d) => {
      xs.push(state.pointData[d.id].x);
      ys.push(state.pointData[d.id].y);
      ws.push(state.pointData[d.id].count);
    });

    // WASM only uses 1-3ms for the whole graph!
    increasingISO.reset();
    increasingISO.fit(xs, ys, ws);
    let isoYs = increasingISO.predict(xs);

    state.pointDataBuffer = JSON.parse(JSON.stringify(state.pointData));
    state.additiveDataBuffer = JSON.parse(JSON.stringify(state.additiveData));

    // Update the last edit graph
    drawLastEdit(state, svg);

    redrawMonotone(state, svg, isoYs, () => {
      updateEBM('current');
    });
    myContextMenu.showConfirmation('increasing', 600);

    // Update EBM
    sidebarInfo.curGroup = 'last';
    sidebarStore.set(sidebarInfo);

    // Update the footer message
    footerStore.update(value => {
      value.state = `Made ${xs.length} bins <b>monotonically increasing</b>`;
      value.type = 'increasing';
      return value;
    });
  };
  
  /**
   * Event handler when user clicks the decreasing button
   */
  const multiMenuDecreasingClicked = () => {
    console.log('decreasing clicked');

    // Animate the bbox
    d3.select(svg)
      .select('g.line-chart-content-group g.select-bbox-group')
      .select('rect.original-bbox')
      .classed('animated', true);

    // Fit an isotonic regression model
    let xs = [];
    let ys = [];
    let ws = [];

    state.selectedInfo.nodeData.forEach((d) => {
      xs.push(state.pointData[d.id].x);
      ys.push(state.pointData[d.id].y);
      ws.push(state.pointData[d.id].count);
    });

    // WASM only uses 1-3ms for the whole graph!
    decreasingISO.reset();
    decreasingISO.fit(xs, ys, ws);
    let isoYs = decreasingISO.predict(xs);

    state.pointDataBuffer = JSON.parse(JSON.stringify(state.pointData));
    state.additiveDataBuffer = JSON.parse(JSON.stringify(state.additiveData));

    // Update the last edit graph
    drawLastEdit(state, svg);

    redrawMonotone(state, svg, isoYs, () => {updateEBM('current');});
    myContextMenu.showConfirmation('decreasing', 600);

    // Update EBM
    sidebarInfo.curGroup = 'last';
    sidebarStore.set(sidebarInfo);
    
    // Update the footer message
    footerStore.update(value => {
      value.state = `Made ${xs.length} bins <b>monotonically decreasing</b>`;
      value.type = 'decreasing';
      return value;
    });
  };

  /**
   * Event handler when user clicks the interpolation button
   */
  const multiMenuInterpolationClicked = () => {
    console.log('interpolation clicked');

    // Animate the bbox
    d3.select(svg)
      .select('g.line-chart-content-group g.select-bbox-group')
      .select('rect.original-bbox')
      .classed('animated', true);

    state.pointDataBuffer = JSON.parse(JSON.stringify(state.pointData));
    state.additiveDataBuffer = JSON.parse(JSON.stringify(state.additiveData));

    // Special for interpolation: we need to create a buffer for the selectedInfo
    // as well (selectedInfo.boundingBox would not change, no need to buffer it)
    state.selectedInfo.nodeDataBuffer = JSON.parse(JSON.stringify(state.selectedInfo.nodeData));

    // Update the last edit graph
    drawLastEdit(state, svg);

    // Set the EBM model (need to change bin definition)
    const callBack = () => {setEBM('current', state.pointDataBuffer);};

    // Determine whether to use in-place interpolation
    if (state.selectedInfo.nodeData.length == 1) {
      return;
    } else if (state.selectedInfo.nodeData.length == 2) {
      multiMenuControlInfo.interpolationMode = 'equal';
      stepInterpolate(state, svg, multiMenuControlInfo.step, callBack);
    } else {
      multiMenuControlInfo.interpolationMode = 'inplace';
      inplaceInterpolate(state, svg, callBack);
    }

    myContextMenu.showConfirmation('interpolation', 600);

    // Update EBM
    sidebarInfo.curGroup = 'last';
    sidebarStore.set(sidebarInfo);

    // Update the footer message
    footerStore.update(value => {
      value.interpolateStyle = 'Interpolated';
      value.state = `<b>Interpolated</b> ${state.selectedInfo.nodeData.length} bins <b>in place</b>`;
      value.interpolateEqual = 'in place';
      if (multiMenuControlInfo.interpolationMode === 'equal') {
        value.type = 'equal-interpolate';
      } else {
        value.type = 'inplace-interpolate';
      }
      return value;
    });
  };

  /**
   * Event handler when user clicks the control button in the interpolation sub-menu
  */
  const multiMenuInterpolateUpdated = async () => {
    console.log('interpolation updated');
    let beforeBinNum = 0;

    const callBack = () => {
      setEBM('current', state.pointDataBuffer);
    };

    if (multiMenuControlInfo.interpolationMode === 'inplace') {
      // Special case: we want to do inplace interpolation from the original data
      // Need to recover original data
      state.pointDataBuffer = JSON.parse(JSON.stringify(state.pointData));
      state.additiveDataBuffer = JSON.parse(JSON.stringify(state.additiveData));
      state.selectedInfo.nodeDataBuffer = JSON.parse(JSON.stringify(state.selectedInfo.nodeData));
      beforeBinNum = state.selectedInfo.nodeDataBuffer.length;
      inplaceInterpolate(state, svg, callBack);

      footerValue.interpolateStyle = 'Interpolated';
      footerValue.interpolateEqual = 'in place';

    } else if (multiMenuControlInfo.interpolationMode === 'equal') {
      // Here we don't reset the pointDataBuffer
      // If user clicks here direction => step interpolate between start & end
      // If user has clicked regression => regression with equal bins
      beforeBinNum = state.selectedInfo.nodeDataBuffer.length;
      stepInterpolate(state, svg, multiMenuControlInfo.step, callBack);

      footerValue.interpolateEqual = `with ${multiMenuControlInfo.step} equal-size bins`;

    } else if (multiMenuControlInfo.interpolationMode === 'regression') {
      // Need to recover original data
      state.pointDataBuffer = JSON.parse(JSON.stringify(state.pointData));
      state.additiveDataBuffer = JSON.parse(JSON.stringify(state.additiveData));
      state.selectedInfo.nodeDataBuffer = JSON.parse(JSON.stringify(state.selectedInfo.nodeData));
      beforeBinNum = state.selectedInfo.nodeDataBuffer.length;

      regressionInterpolate(state, svg, callBack);

      footerValue.interpolateStyle = 'Regression transformed';
      footerValue.interpolateEqual = 'in place';

    } else {
      console.error('Unknown regression mode ', multiMenuControlInfo.interpolationMode);
    }

    // Update the footer message
    footerValue.state = `<b>${footerValue.interpolateStyle}</b> ${beforeBinNum}
      bins <b>${footerValue.interpolateEqual}</b>`;

    if (footerValue.interpolateEqual !== 'in place') {
      if (footerValue.interpolateStyle === 'Interpolated') {
        footerValue.type = 'equal-interpolate';
      } else {
        footerValue.type = 'equal-regression';
      }
    } else {
      if (footerValue.interpolateStyle === 'Interpolated') {
        footerValue.type = 'inplace-interpolate';
      } else {
        footerValue.type = 'inplace-regression';
      }
    }

    footerStore.set(footerValue);
  };

  const multiMenuMergeClicked = () => {
    console.log('merge clicked');

    // Animate the bbox
    d3.select(svg)
      .select('g.line-chart-content-group g.select-bbox-group')
      .select('rect.original-bbox')
      .classed('animated', true);

    state.pointDataBuffer = JSON.parse(JSON.stringify(state.pointData));
    state.additiveDataBuffer = JSON.parse(JSON.stringify(state.additiveData));

    // Update EBM
    const callBack = () => {updateEBM('current');};

    // Update the last edit graph
    drawLastEdit(state, svg);

    merge(state, svg, 'left', callBack);

    myContextMenu.showConfirmation('merge', 600);

    sidebarInfo.curGroup = 'last';
    sidebarStore.set(sidebarInfo);

    // Update the footer message
    footerStore.update(value => {
      value.type = 'align';
      value.state = `Set scores of ${state.selectedInfo.nodeData.length} bins to
        <b>${round(state.selectedInfo.nodeData[0].y, 4)}</b>`;
      return value;
    });
  };

  const multiMenuInputChanged = () => {
    // Animate the bbox
    d3.select(svg)
      .select('g.line-chart-content-group g.select-bbox-group')
      .select('rect.original-bbox')
      .classed('animated', true);

    state.pointDataBuffer = JSON.parse(JSON.stringify(state.pointData));
    state.additiveDataBuffer = JSON.parse(JSON.stringify(state.additiveData));

    // Update EBM
    const callBack = () => {updateEBM('current');};
    merge(state, svg, multiMenuControlInfo.setValue, callBack);

    myContextMenu.showConfirmation('change', 600);

    sidebarInfo.curGroup = 'last';
    sidebarStore.set(sidebarInfo);

    const target = round(multiMenuControlInfo.setValue, 4);

    // Update the footer message
    footerStore.update(value => {
      value.type = 'align';
      value.state = `Set scores of ${state.selectedInfo.nodeData.length} bins to <b>${target}</b>`;
      return value;
    });
  };

  const multiMenuMergeUpdated = () => {
    state.pointDataBuffer = JSON.parse(JSON.stringify(state.pointData));
    state.additiveDataBuffer = JSON.parse(JSON.stringify(state.additiveData));

    // Update EBM
    const callBack = () => {updateEBM('current');};
    let target = merge(state, svg, multiMenuControlInfo.mergeMode, callBack);
    target = round(target, 4);

    sidebarInfo.curGroup = 'last';
    sidebarStore.set(sidebarInfo);

    // Update the footer message
    footerStore.update(value => {
      value.type = 'align';
      value.state = `Set scores of ${state.selectedInfo.nodeData.length} bins to <b>${target}</b>`;
      return value;
    });
  };

  const multiMenuDeleteClicked = () => {
    console.log('delete clicked');

    // Animate the bbox
    d3.select(svg)
      .select('g.line-chart-content-group g.select-bbox-group')
      .select('rect.original-bbox')
      .classed('animated', true);

    state.pointDataBuffer = JSON.parse(JSON.stringify(state.pointData));
    state.additiveDataBuffer = JSON.parse(JSON.stringify(state.additiveData));

    // Update the last edit graph
    drawLastEdit(state, svg);

    // Update EBM
    const callBack = () => {updateEBM('current');};

    merge(state, svg, 0, callBack);

    myContextMenu.showConfirmation('delete', 600);

    // Update EBM
    sidebarInfo.curGroup = 'last';
    sidebarStore.set(sidebarInfo);

    // Update the footer message
    footerStore.update(value => {
      value.type = 'delete';
      value.state = `Set scores of ${state.selectedInfo.nodeData.length} bins to <b>${0}</b>`;
      return value;
    });
  };

  /**
   * Event handler when user clicks the check icon in the sub-menu
   */
  const multiMenuSubItemCheckClicked = () => {

    if (multiMenuControlInfo.subItemMode === null) {
      console.error('No sub item is selected but check is clicked!');
    }

    const existingModes = new Set(['increasing', 'decreasing', 'interpolation', 'change', 'merge', 'delete']);
    if (!existingModes.has(multiMenuControlInfo.subItemMode)) {
      console.error(`Encountered unknown subItemMode: ${multiMenuControlInfo.subItemMode}`);
    }

    // Stop the bbox animation
    d3.select(svg)
      .select('g.line-chart-content-group g.select-bbox-group')
      .select('rect.original-bbox')
      .classed('animated', false);

    const binNum = state.selectedInfo.nodeData.length;

    // Save the changes
    state.pointData = JSON.parse(JSON.stringify(state.pointDataBuffer));
    state.additiveData = JSON.parse(JSON.stringify(state.additiveDataBuffer));

    // Update the last edit data to current data (redraw the graph only when user enters
    // editing mode next time)
    if (state.additiveDataLastEdit !== undefined) {
      state.additiveDataLastLastEdit = JSON.parse(JSON.stringify(state.additiveDataLastEdit));
    }
    state.additiveDataLastEdit = JSON.parse(JSON.stringify(state.additiveData));

    // Special [interpolation]: need to save the new selectedInfo as well
    if (multiMenuControlInfo.subItemMode === 'interpolation') {
      state.selectedInfo.nodeData = JSON.parse(JSON.stringify(state.selectedInfo.nodeDataBuffer));
      state.selectedInfo.nodeDataBuffer = null;
    }

    // Hide the confirmation panel
    myContextMenu.hideConfirmation(multiMenuControlInfo.subItemMode);

    // Move the menu bar
    d3.select(multiMenu)
      .call(moveMenubar, svg, component);
    
    // Exit the sub-item mode
    multiMenuControlInfo.subItemMode = null;
    multiMenuControlInfo.setValue = null;

    // Update metrics
    sidebarInfo.curGroup = 'commit';
    sidebarStore.set(sidebarInfo);

    // Update the footer message
    let editType = '';
    footerStore.update(value => {
      editType = value.type;
      // Reset the baseline
      value.baseline = 0;
      value.baselineInit = false;
      value.state = '';
      value.type = '';
      value.help = '<b>Drag</b> to marquee select, <b>Scroll</b> to zoom';
      return value;
    });

    // Push the commit to history

    // Get the info of edited bins
    const binLeft = state.selectedInfo.nodeData[0];
    const binRight = state.pointData[state.pointData[
      state.selectedInfo.nodeData[state.selectedInfo.nodeData.length - 1].id].rightPointID];
    const binRange = binRight === undefined ? `${round(binLeft.x, 2)} <= x` : `${round(binLeft.x, 2)} <= x < ${round(binRight.x, 2)}`;
    let description = '';

    switch(editType) {
    case 'increasing':
      description = `Made ${binNum} bins (${binRange}) monotonically increasing.`;
      break;
    case 'decreasing':
      description = `Made ${binNum} bins (${binRange}) monotonically decreasing.`;
      break;
    case 'inplace-interpolate':
      description = `Interpolated ${binNum} bins (${binRange}) in place.`;
      break;
    case 'inplace-regression':
      description = `Regression transformed ${binNum} bins (${binRange}) in place.`;
      break;
    case 'equal-interpolate':
      description = `Interpolated ${binNum} bins (${binRange}) into ${multiMenuControlInfo.step} equal-size bins.`;
      break;
    case 'equal-regression':
      description = `Regression transformed ${binNum} bins (${binRange}) into ${multiMenuControlInfo.step} equal-size bins.`;
      break;
    case 'align':
      description = `Set ${binNum} bins (${binRange}) to score ${round(state.selectedInfo.nodeData[0].y, 4)}.`;
      break;
    case 'delete':
      description = `Set ${binNum} bins (${binRange}) to score 0.`;
      break;
    default:
      break;
    }

    pushCurStateToHistoryStack(state, editType, description, historyStore, sidebarStore);

    // Any new commit purges the redo stack
    redoStack = [];
  };

  /**
   * Event handler when user clicks the cross icon in the sub-menu
   */
  const multiMenuSubItemCancelClicked = (e, cancelFromMove = false) => {
    console.log('sub item cancel clicked');
    if (!cancelFromMove && multiMenuControlInfo.subItemMode === null) {
      console.error('No sub item is selected but check is clicked!');
    }

    const existingModes = new Set(['increasing', 'decreasing', 'interpolation', 'change', 'merge', 'delete']);
    if (!cancelFromMove && !existingModes.has(multiMenuControlInfo.subItemMode)) {
      console.error(`Encountered unknown subItemMode: ${multiMenuControlInfo.subItemMode}`);
    }

    // Stop the bbox animation
    d3.select(svg)
      .select('g.line-chart-content-group g.select-bbox-group')
      .select('rect.original-bbox')
      .classed('animated', false);

    // Discard the change
    state.pointDataBuffer = null;
    state.additiveDataBuffer = null;

    // Recover the last edit graph
    if (state.additiveDataLastLastEdit !== undefined){
      state.additiveDataLastEdit = JSON.parse(JSON.stringify(state.additiveDataLastLastEdit));
      drawLastEdit(state, svg);
      // Prepare for next redrawing after recovering the last last edit graph
      state.additiveDataLastEdit = JSON.parse(JSON.stringify(state.additiveData));
    }

    // If the current edit is interpolation, we need to recover the bin definition
    // in the EBM model
    let callBack = () => {};

    if (!cancelFromMove) {
      if (multiMenuControlInfo.subItemMode === 'interpolation') {
        callBack = () => {setEBM('current', state.pointData);};
      } else {
        // For other types of update, we need to revoke the changes
        callBack = () => {updateEBM('recoverEBM');};
      }
    }

    // Update metrics
    if (!cancelFromMove) {
      sidebarInfo.curGroup = 'recover';
      sidebarStore.set(sidebarInfo);
    }

    // Recover the original graph
    redrawOriginal(state, svg, true, () => {
      // Move the menu bar after the animation
      d3.select(multiMenu)
        .call(moveMenubar, svg, component);

      // Update the EBM in "background"
      callBack();
    });

    // Hide the confirmation panel
    myContextMenu.hideConfirmation(multiMenuControlInfo.subItemMode);

    // Exit the sub-item mode
    multiMenuControlInfo.subItemMode = null;
    multiMenuControlInfo.setValue = null;

    // Update the footer message
    footerStore.update(value => {
      // Reset the baseline
      value.baseline = 0;
      value.baselineInit = false;
      value.state = '';
      value.type = '';
      value.help = '<b>Drag</b> to marquee select, <b>Scroll</b> to zoom';
      return value;
    });
  };

  $: featureData && ebm && mounted && !initialized && featureData.name === ebm.editingFeatureName && drawFeature(featureData);

</script>

<style type='text/scss'>
  @import '../define';
  @import './common.scss';

  :global(.explain-panel circle.node) {
    fill: $blue-icon;
    stroke: change-color($blue-icon, $lightness: 95%) ;
  }

  :global(.explain-panel circle.node.selected) {
    fill: $orange-400;
    stroke: white;
  }

  :global(.explain-panel .additive-line-segment) {
    stroke-linejoin: round;
    stroke-linecap: round;
  }

  :global(.explain-panel path.additive-line-segment.selected) {
    stroke: adjust-color($orange-400, $lightness: -8%);
  }

  :global(.explain-panel .line-chart-content-group) {
    cursor: grab;
  }

  :global(.explain-panel .line-chart-content-group:active) {
    cursor: grabbing;
  }

  :global(.explain-panel .line-chart-content-group.select-mode) {
    cursor: crosshair;
  }

  :global(.explain-panel .line-chart-content-group.select-mode:active) {
    cursor: crosshair;
  }

  :global(.explain-panel .svg-icon) {
    display: flex;
    justify-content: center;
    align-items: center;

    :global(svg) {
      width: 1.2em;
      height: 1.2em;
    }
  }

  :global(.explain-panel .select-bbox) {
    fill: none;
  }

  :global(.explain-panel .select-bbox-group) {
    pointer-events: all;
  }

  @keyframes dash {
      to {
      stroke-dashoffset: -1000;
    }
  }

  @-moz-keyframes dash {
      to {
      stroke-dashoffset: -1000;
    }
  }

  @-webkit-keyframes dash {
      to {
      stroke-dashoffset: -1000;
    }
  }

  :global(.explain-panel rect.original-bbox.animated) {
    -webkit-animation: dash 50s infinite linear;
    animation: dash 50s infinite linear;
  }

  .context-menu-container {
    pointer-events: fill;

    &.hidden {
      pointer-events: none;
      cursor: none;
    }
  }

</style>

<div class='explain-panel' bind:this={component}>
    <a id="download-anchor" style="display:none"> </a>

    <div class='context-menu-container hidden' bind:this={multiMenu}>
      <ContextMenu 
        bind:controlInfo={multiMenuControlInfo}
        bind:this={myContextMenu} 
        on:inputChanged={multiMenuInputChanged}
        on:moveButtonClicked={multiMenuMoveClicked}
        on:increasingClicked={multiMenuIncreasingClicked}
        on:decreasingClicked={multiMenuDecreasingClicked}
        on:interpolationClicked={multiMenuInterpolationClicked}
        on:mergeClicked={multiMenuMergeClicked}
        on:mergeUpdated={multiMenuMergeUpdated}
        on:deleteClicked={multiMenuDeleteClicked}
        on:moveCheckClicked={multiMenuMoveCheckClicked}
        on:moveCancelClicked={multiMenuMoveCancelClicked}
        on:subItemCheckClicked={multiMenuSubItemCheckClicked}
        on:subItemCancelClicked={multiMenuSubItemCancelClicked}
        on:interpolateUpdated={multiMenuInterpolateUpdated}
      /> 
    </div>

  <div class='svg-container'>
    <svg class='svg-explainer' width={svgWidth} height={svgHeight} bind:this={svg}></svg>
  </div>
  
</div>