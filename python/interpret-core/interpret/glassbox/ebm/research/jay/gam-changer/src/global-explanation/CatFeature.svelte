<script>
  import * as d3 from 'd3';
  import { onMount } from 'svelte';
  import { round } from '../utils/utils';
  import { config } from '../config';

  import { drawBarLegend } from './draw';
  import { SelectedInfo } from './categorical/cat-class';
  import { zoomStart, zoomEnd, zoomed, zoomScaleExtent, rExtent } from './categorical/cat-zoom';
  import { brushDuring, brushEndSelect } from './categorical/cat-brush';
  import { moveMenubar } from './continuous/cont-bbox';
  import { drawLastEdit, drawBufferGraph, grayOutConfidenceLine, redrawOriginal } from './categorical/cat-edit';

  import ContextMenu from '../components/ContextMenu.svelte';

  export let featureData = null;
  export let labelEncoder = null;
  export let scoreRange = null;
  export let svgHeight = 400;

  let svg = null;
  let component = null;
  let brush = null;
  let multiMenu = null;
  let myContextMenu = null;

  let mounted = false;
  let initialized = false;

  // Visualization constants
  const svgPadding = config.svgPadding;
  const densityHeight = 90;

  // Viewbox width and height
  let width = 600;
  const height = 400;

  // Context menu info
  let multiMenuControlInfo = {
    moveMode: false,
    toSwitchMoveMode: false,
    subItemMode: null,
    setValue: null
  };

  // Real width (depends on the svgHeight prop)
  let svgWidth = svgHeight * (width / height);

  // Show some hidden elements for development
  const showRuler = false;

  // Some styles
  const colors = config.colors;
  const defaultFont = config.defaultFont;
  let barWidth = null;

  // Select mode
  let state = {
    curXScale: null,
    curYScale: null,
    curTransform: null,
    selectedInfo: null,
    pointData: null,
    pointDataBuffer: null,
    oriXScale: null,
    oriYScale: null,
    bboxPadding: 5,
  };
  let selectMode = false;
  state.selectedInfo = new SelectedInfo();

  /**
   * Create a path to indicate the confidence interval for the additive score of
   * categorical variables.
   * @param d
   * @param xScale
   * @param yScale
   */
  const createDotConfidencePath = (d, width, xScale, yScale) => {

    let topMid = {
      x: xScale(d.x),
      y: yScale(d.y + d.error)
    };

    let btmMid = {
      x: xScale(d.x),
      y: yScale(d.y - d.error)
    };
    
    // Draw the top line
    let pathStr = `M ${topMid.x - width}, ${topMid.y} L ${topMid.x + width}, ${topMid.y} `;

    // Draw the vertical line
    pathStr = pathStr.concat(`M ${topMid.x}, ${topMid.y} L ${btmMid.x}, ${btmMid.y} `);

    // Draw the bottom line
    pathStr = pathStr.concat(`M ${btmMid.x - width}, ${btmMid.y} L ${btmMid.x + width}, ${btmMid.y} `);

    return pathStr;
  };

  onMount(() => {mounted = true;});

  /**
   * Draw the plot in the SVG component
   * @param featureData
   */
  const drawFeature = (featureData) => {
    console.log(featureData);
    let svgSelect = d3.select(svg);

    // For categorical variables, the width depends on the number of levels
    // Level # <= 4 => 300, level # <= 10 => 450, others => 600
    // let levelNum = featureData.binLabel.length;
    // if (levelNum <= 10) width = 450;
    // if (levelNum <= 4) width = 300;

    // Make the svg keep the viewbox 3:2 ratio
    svgWidth = svgHeight * (width / height);

    // Set svg viewBox (3:2 WH ratio)
    svgSelect.attr('viewBox', `0 0 ${width} ${height}`)
      .attr('preserveAspectRatio', 'xMinYMin meet')
      .attr('width', svgWidth)
      .attr('height', svgHeight);

    // Draw a border for the svg
    svgSelect.append('rect')
      .attr('class', 'border')
      .classed('hidden', !showRuler)
      .attr('width', 600)
      .attr('height', 400)
      .style('fill', 'none')
      .style('stroke', 'pink');

    // Some constant lengths of different elements
    // Approximate the longest width of score (y-axis)
    const yAxisWidth = 5 * d3.max(scoreRange.map(d => String(round(d, 1)).length));
    const chartWidth = width - svgPadding.left - svgPadding.right - yAxisWidth;
    const chartHeight = height - svgPadding.top - svgPadding.bottom - densityHeight;

    // Draw the bar legend
    drawBarLegend(svgSelect, width, svgPadding);

    let content = svgSelect.append('g')
      .attr('class', 'content')
      .attr('transform', `translate(${svgPadding.left}, ${svgPadding.top})`);
    
    let binValues = featureData.binLabel.map(d => labelEncoder[d]);

    let xScale = d3.scalePoint()
      .domain(binValues)
      .padding(0.7)
      .range([0, chartWidth]);

    // For the y scale, it seems InterpretML presets the center at 0 (offset
    // doesn't really matter in EBM because we can modify intercept)
    // TODO: Provide interaction for users to change the center point
    // Normalize the Y axis by the global score range
    let yScale = d3.scaleLinear()
      .domain(scoreRange)
      .range([chartHeight, 0]);

    state.oriXScale = xScale;
    state.oriYScale = yScale;
    state.curXScale = xScale;
    state.curYScale = yScale;

    // Create a data array by combining the bin labels, additive terms, and errors
    let pointData = {};

    for (let i = 0; i < featureData.binLabel.length; i++) {
      pointData[featureData.binLabel[i]] = {
        x: labelEncoder[featureData.binLabel[i]],
        y: featureData.additive[i],
        id: featureData.binLabel[i],
        error: featureData.error[i]
      };
    }

    state.pointData = pointData;

    // Create histogram chart group
    let histChart = content.append('g')
      .attr('class', 'hist-chart-group');

    // For the histogram clippath, need to carefully play around with the
    // transformation, the path should be in a static group; the group having
    // clip-path attr should be static. Therefore we apply the transformation to
    // histChart's child later.
    histChart.append('clipPath')
      .attr('id', `${featureData.name.replace(/\s/g, '')}-hist-chart-clip`)
      .append('rect')
      .attr('x', yAxisWidth)
      .attr('y', chartHeight)
      .attr('width', chartWidth)
      .attr('height', densityHeight);
    
    histChart.attr('clip-path', `url(#${featureData.name.replace(/\s/g, '')}-hist-chart-clip)`);
    
    // Draw the dot plot
    let scatterPlot = content.append('g')
      .attr('class', 'scatter-plot-group');

    scatterPlot.append('clipPath')
      .attr('id', `${featureData.name.replace(/\s/g, '')}-chart-clip`)
      .append('rect')
      .attr('width', chartWidth)
      .attr('height', chartHeight - 1);

    scatterPlot.append('clipPath')
      .attr('id', `${featureData.name.replace(/\s/g, '')}-x-axis-clip`)
      .append('rect')
      .attr('x', yAxisWidth)
      .attr('y', chartHeight)
      .attr('width', chartWidth)
      .attr('height', densityHeight);

    // Create axis group early so it shows up at the bottom
    let axisGroup = scatterPlot.append('g')
      .attr('class', 'axis-group');
    
    let scatterPlotContent = scatterPlot.append('g')
      .attr('class', 'scatter-plot-content-group')
      .attr('clip-path', `url(#${featureData.name.replace(/\s/g, '')}-chart-clip)`)
      .attr('transform', `translate(${yAxisWidth}, 0)`);

    // Append a rect so we can listen to events
    scatterPlotContent.append('rect')
      .attr('width', chartWidth)
      .attr('height', chartHeight)
      .style('opacity', 0);

    // Create a group to draw grids
    let gridGroup = scatterPlotContent.append('g')
      .attr('class', 'scatter-plot-grid-group');

    let confidenceGroup = scatterPlotContent.append('g')
      .attr('class', 'scatter-plot-confidence-group');

    let barGroup = scatterPlotContent.append('g')
      .attr('class', 'scatter-plot-bar-group real');

    let scatterGroup = scatterPlotContent.append('g')
      .attr('class', 'scatter-plot-dot-group');

    barWidth = Math.min(30, xScale(pointData[2].x) - xScale(pointData[1].x));

    // We draw bars from the 0 baseline to the dot position
    barGroup.style('fill', colors.bar)
      .selectAll('rect')
      .data(Object.values(pointData), d => d.id)
      .join('rect')
      .attr('class', 'additive-bar')
      .attr('x', d => xScale(d.x) - barWidth / 2)
      .attr('y', d => d.y > 0 ? yScale(d.y) : yScale(0))
      .attr('width', barWidth)
      .attr('height', d => Math.abs(yScale(d.y) - yScale(0)));

    // We draw the shape function with many line segments (path)
    scatterGroup.selectAll('circle')
      .data(Object.values(pointData), d => d.id)
      .join('circle')
      .attr('class', 'additive-dot')
      .attr('cx', d => xScale(d.x))
      .attr('cy', d => yScale(d.y))
      .attr('r', rExtent[0])
      .style('stroke-width', 1);

    // Draw the underlying confidence interval
    confidenceGroup.style('stroke', colors.dotConfidence)
      .style('stroke-width', 2)
      .selectAll('path')
      .data(Object.values(pointData))
      .join('path')
      .attr('class', 'dot-confidence')
      .attr('d', d => createDotConfidencePath(d, 5, xScale, yScale));

    // Clone the rects and dots for original and last edit
    confidenceGroup.lower();

    barGroup.clone(true)
      .classed('last-edit-back', true)
      .classed('real', false)
      .lower()
      .selectAll('rect')
      .remove();

    barGroup.clone(true)
      .classed('last-edit-front', true)
      .classed('real', false)
      .raise()
      .selectAll('rect')
      .remove();

    barGroup.clone(true)
      .classed('original', true)
      .classed('real', false)
      .style('fill', 'hsl(0, 0%, 85%)')
      .style('opacity', 1)
      .lower();
    
    // Add level lines to the original bar group
    let originalFront = barGroup.clone(true)
      .classed('original-front', true)
      .classed('real', false)
      .raise();

    originalFront.selectAll('rect')
      .remove();

    originalFront.selectAll('path.original-line')
      .data(Object.values(pointData), d => d.id)
      .join('path')
      .attr('class', 'original-line')
      .style('stroke', 'hsl(0, 0%, 75%)')
      .attr('d', d => `M ${state.oriXScale(d.x) - barWidth / 2}, ${state.oriYScale(d.y)} l ${barWidth}, 0`);

    // Make sure the dots are on top
    confidenceGroup.raise();
    scatterGroup.raise();
    gridGroup.lower();
    

    // Draw the chart X axis
    // Hack: create a wrapper so we can apply clip before transformation
    let xAxisGroup = axisGroup.append('g')
      .attr('class', 'x-axis-wrapper')
      .attr('clip-path', `url(#${featureData.name.replace(/\s/g, '')}-x-axis-clip)`)
      .append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(${yAxisWidth}, ${chartHeight})`)
      .call(d3.axisBottom(xScale));
    
    xAxisGroup.attr('font-family', defaultFont)
      .select('path')
      .style('stroke-width', 1.5);

    // Rotate the x axis label if there are too many values
    if (featureData.binLabel.length > 6) {
      xAxisGroup.selectAll('g.tick text')
        .attr('y', 0)
        .attr('x', 9)
        .attr('dy', '-0.6em')
        .attr('transform', 'rotate(90)')
        .style('text-anchor', 'start');
    }
    
    // Draw the chart Y axis
    let yAxisGroup = axisGroup.append('g')
      .attr('class', 'y-axis')
      .attr('transform', `translate(${yAxisWidth}, 0)`);
    
    yAxisGroup.call(d3.axisLeft(yScale));
    yAxisGroup.attr('font-family', defaultFont);

    yAxisGroup.append('g')
      .attr('class', 'y-axis-text')
      .attr('transform', `translate(${-yAxisWidth - 15}, ${chartHeight / 2}) rotate(-90)`)
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
        x1: labelEncoder[featureData.histEdge[i]],
        x2: labelEncoder[featureData.histEdge[i + 1]],
        height: histFrequency[i]
      });
    }

    let histYScale = d3.scaleLinear()
      .domain(d3.extent(histFrequency))
      .range([0, densityHeight]);

    let histWidth = Math.min(30, xScale(histData[0].x2) - xScale(histData[0].x1));

    // Draw the density histogram 
    let histChartContent = histChart.append('g')
      .attr('class', 'hist-chart-content-group')
      .attr('transform', `translate(${yAxisWidth}, ${chartHeight})`);

    histChartContent.selectAll('rect')
      .data(histData)
      .join('rect')
      .attr('class', 'hist-rect')
      .attr('x', d => xScale(d.x1) - histWidth / 2)
      .attr('y', 0)
      .attr('width', histWidth)
      .attr('height', d => histYScale(d.height))
      .style('fill', colors.hist);
    
    // Draw a Y axis for the histogram chart
    let yAxisHistGroup = scatterPlot.append('g')
      .attr('class', 'y-axis')
      .attr('transform', `translate(${yAxisWidth}, ${chartHeight})`);
    
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
        e, state, svg, multiMenu, brush, component, resetContextMenu, barWidth
      ))
      .on('start brush', e => brushDuring(e, state, svg, multiMenu))
      .extent([[0, 0], [chartWidth, chartHeight]])
      .filter((e) => {
        if (selectMode) {
          return e.button === 0;
        } else {
          return e.button === 2;
        }
      });

    let brushGroup = scatterPlotContent.append('g')
      .attr('class', 'brush')
      .call(brush);
    
    // Change the style of the select box
    brushGroup.select('rect.overlay')
      .attr('cursor', null);

    // Add panning and zooming
    let zoom = d3.zoom()
      .scaleExtent(zoomScaleExtent)
      .translateExtent([[0, -Infinity], [width, Infinity]])
      .on('zoom', e => zoomed(e, state, xScale, yScale, svg, 2,
        1, yAxisWidth, chartWidth, chartHeight, multiMenu, component))
      .on('start', () => zoomStart(state, multiMenu))
      .on('end', () => zoomEnd(state, multiMenu))
      .filter(e => {
        if (selectMode) {
          return (e.type === 'wheel' || e.button === 2);
        } else {
          return (e.button === 0 || e.type === 'wheel');
        }
      });

    scatterPlotContent.call(zoom)
      .call(zoom.transform, d3.zoomIdentity);

    scatterPlotContent.on('dblclick.zoom', null);
    
    // Listen to double click to reset zoom
    scatterPlotContent.on('dblclick', () => {
      scatterPlotContent.transition('reset')
        .duration(750)
        .ease(d3.easeCubicInOut)
        .call(zoom.transform, d3.zoomIdentity);
    });

    initialized = true;
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

    return {moveMode: moveMode, subItemMode: subItemMode};
  };

  const multiMenuInputChanged = () => {

  };

  const dragged = (e) => {
    
    const dataYChange = state.curYScale.invert(e.y) - state.curYScale.invert(e.y - e.dy);

    // Change the data based on the y-value changes, then redraw nodes (preferred method)
    state.selectedInfo.nodeData.forEach(d => {

      // Step 1.1: update point data
      state.pointDataBuffer[d.id].y += dataYChange;
    });

    // Step 1.2: update the bbox info
    state.selectedInfo.updateNodeData(state.pointDataBuffer);
    state.selectedInfo.computeBBox(state.pointDataBuffer);

    // Draw the new graph
    drawBufferGraph(state, svg, false, 400);

    // Update the sidebar info
    // if (dragTimeout !== null) {
    //   clearTimeout(dragTimeout);
    // }
    // dragTimeout = setTimeout(() => {
    //   updateEBM('current');
    // }, useTimeout ? 300 : 0);
  };

  const multiMenuMoveClicked = () => {

    // Step 1. create data clone buffers for user to change
    // We only do this when buffer has not been created --- it is possible that
    // user switch to move from other editing mode
    if (state.pointDataBuffer === null) {
      state.pointDataBuffer = JSON.parse(JSON.stringify(state.pointData));
    }

    let bboxGroup = d3.select(svg)
      .select('g.scatter-plot-content-group g.select-bbox-group')
      .style('cursor', 'row-resize')
      .call(d3.drag()
        .on('start', () => {
          grayOutConfidenceLine(state, svg);
          //TODO: update footer
        })
        .on('drag', (e) => dragged(e))
      );
    
    bboxGroup.select('rect.original-bbox')
      .classed('animated', true);
    
    // Show the last edit
    if (state.pointDataLastEdit !== undefined) {
      drawLastEdit(state, svg, barWidth);
    }

  };

  const multiMenuMergeClicked = () => {

  };

  const multiMenuDeleteClicked = () => {

  };

  const multiMenuMoveCheckClicked = () => {
    // Save the changes
    state.pointData = JSON.parse(JSON.stringify(state.pointDataBuffer));

    // Remove the drag
    let bboxGroup = d3.select(svg)
      .select('g.scatter-plot-content-group g.select-bbox-group')
      .style('cursor', null)
      .on('.drag', null);
    
    // stop the animation
    bboxGroup.select('rect.original-bbox')
      .classed('animated', false);

    // Move the menu bar
    d3.select(multiMenu)
      .call(moveMenubar, svg, component);

    // Save this change to lastEdit, update lastEdit graph
    if (state.pointDataLastEdit !== undefined) {
      state.pointDataLastLastEdit = JSON.parse(JSON.stringify(state.pointDataLastEdit));
    }
    state.pointDataLastEdit = JSON.parse(JSON.stringify(state.pointData));
  };

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
      // updateEBM('recoverEBM');
    });

    redrawOriginal(state, svg);

    // Remove the drag
    let bboxGroup = d3.select(svg)
      .select('g.scatter-plot-content-group g.select-bbox-group')
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
    
    // Update the footer message
  };

  const multiMenuSubItemCheckClicked = () => {

  };

  const multiMenuSubItemCancelClicked = () => {

  };

  /**
   * Event handler for the select button in the header
   */
  export const selectModeSwitched = () => {
    selectMode = !selectMode;

    let lineChartContent = d3.select(svg)
      .select('g.scatter-plot-content-group')
      .classed('select-mode', selectMode);
    
    lineChartContent.select('g.brush rect.overlay')
      .attr('cursor', null);
  };

  $: featureData && mounted && !initialized && drawFeature(featureData);

</script>

<style type='text/scss'>
  @import '../define';
  @import './common.scss';

  :global(.explain-panel path.dot-confidence.edited) {
    stroke: hsl(0, 0%, 75%);
  }

  :global(.explain-panel rect.additive-bar.selected) {
    fill: $orange-300;
    opacity: 0.9;
  }

  :global(.explain-panel .last-edit-back rect.additive-bar) {
    fill: hsl(35, 100%, 85%);
    opacity: 0.7;
  }

  :global(.explain-panel .last-edit-front path.additive-line) {
    stroke: hsl(35, 100%, 85%);
    opacity: 1;
  }

  :global(.explain-panel circle.additive-dot) {
    fill: $blue-icon;
    stroke: white;
  }

  :global(.explain-panel circle.additive-dot.selected) {
    fill: $orange-400;
    stroke: white;
  }

  :global(.explain-panel .scatter-plot-content-group) {
    cursor: grab;
  }

  :global(.explain-panel .scatter-plot-content-group:active) {
    cursor: grabbing;
  }

  :global(.explain-panel .scatter-plot-content-group.select-mode) {
    cursor: crosshair;
  }

  :global(.explain-panel .scatter-plot-content-group.select-mode:active) {
    cursor: crosshair;
  }

</style>

<div class='explain-panel' bind:this={component}>

  <div class='context-menu-container hidden' bind:this={multiMenu}>
    <ContextMenu 
      bind:controlInfo={multiMenuControlInfo}
      bind:this={myContextMenu}
      type='cat'
      on:inputChanged={multiMenuInputChanged}
      on:moveButtonClicked={multiMenuMoveClicked}
      on:mergeClicked={multiMenuMergeClicked}
      on:deleteClicked={multiMenuDeleteClicked}
      on:moveCheckClicked={multiMenuMoveCheckClicked}
      on:moveCancelClicked={multiMenuMoveCancelClicked}
      on:subItemCheckClicked={multiMenuSubItemCheckClicked}
      on:subItemCancelClicked={multiMenuSubItemCancelClicked}
    /> 
  </div>

  <div class='svg-container'>
    <svg class='svg-explainer' bind:this={svg}></svg>
  </div>
  
</div>