<script>
  import * as d3 from 'd3';
  import { onMount } from 'svelte';
  import { round } from '../utils/utils';
  import { config } from '../config';
  import { drawHorizontalColorLegend } from './draw';

  import { state } from './inter-cat-cat/cat-cat-state';
  import { SelectedInfo } from './inter-cat-cat/cat-cat-class';
  import { zoomStart, zoomEnd, zoomed, zoomScaleExtent, rExtent } from './inter-cat-cat/cat-cat-zoom';
  import { brushDuring, brushEndSelect } from './inter-cat-cat/cat-cat-brush';

  import ToggleSwitch from '../components/ToggleSwitch.svelte';
  import ContextMenu from '../components/ContextMenu.svelte';

  export let featureData = null;
  export let labelEncoder = null;
  export let scoreRange = null;
  export let svgHeight = 400;

  let svg = null;
  let component = null;
  let multiMenu = null;

  let mounted = false;
  let initialized = false;

  // Visualization constants
  const svgPadding = config.svgPadding;
  const densityHeight = 90;

  // Viewbox width and height
  let width = 600;
  const height = 400;

  // Real width (depends on the svgHeight prop)
  let svgWidth = svgHeight * (width / height);

  // Show some hidden elements for development
  const showRuler = false;

  // Colors
  const colors = config.colors;

  // Select mode
  let selectMode = false;
  state.selectedInfo = new SelectedInfo();

  /**
   * Create additiveData which is used to draw dots on the plot.
   * @param {[object]} featureData Original feature data passed from the parent component
   * @param {[object]} data Processed feature data (separated by long/short categorical variables)
   */
  const createAdditiveData = (featureData, data) => {
    let additiveData = [];

    for (let i = 0; i < featureData.additive.length; i++) {
      for (let j = 0; j < featureData.additive[i].length; j++) {
        additiveData.push({
          longLabel: data.longDim === 0 ? data.longBinLabel[i] : data.longBinLabel[j],
          longLabelName: data.longDim === 0 ? data.longBinName[i] : data.longBinName[j],
          shortLabel: data.longDim === 0 ? data.shortBinLabel[j] : data.shortBinLabel[i],
          shortLabelName: data.longDim === 0 ? data.shortBinName[j] : data.shortBinName[i],
          additive: featureData.additive[i][j],
          error: featureData.error[i][j]
        });
      }
    }

    return additiveData;
  };

  /**
   * Separate feature data into categorical variable with more & fewer levels
   * @param {object} featureData
   */
  const preProcessData = (featureData) => {
    let data = {};
    let len1 = featureData.binLabel1.length;
    let len2 = featureData.binLabel2.length;

    if (len1 >= len2) {
      data.longBinLabel = featureData.binLabel1;
      data.shortBinLabel = featureData.binLabel2;
      data.longName = featureData.name1;
      data.shortName = featureData.name2;
      data.longHistEdge = featureData.histEdge1;
      data.shortHistEdge = featureData.histEdge2;
      data.longHistCount = featureData.histCount1;
      data.shortHistCount = featureData.histCount2;
      data.longDim = 0;
      data.shortDim = 1;
    } else {
      data.longBinLabel = featureData.binLabel2;
      data.shortBinLabel = featureData.binLabel1;
      data.longName = featureData.name2;
      data.shortName = featureData.name1;
      data.longHistEdge = featureData.histEdge2;
      data.shortHistEdge = featureData.histEdge1;
      data.longHistCount = featureData.histCount2;
      data.shortHistCount = featureData.histCount1;
      data.longDim = 1;
      data.shortDim = 0;
    }

    // Encode the categorical level names
    data.longBinName = data.longBinLabel.map(d => labelEncoder[data.longName][d]);
    data.shortBinName = data.shortBinLabel.map(d => labelEncoder[data.shortName][d]);

    data.longHistEdgeName = data.longHistEdge.map(d => labelEncoder[data.longName][d]);
    data.shortHistEdgeName = data.shortHistEdge.map(d => labelEncoder[data.shortName][d]);

    return data;
  };

  onMount(() => {mounted = true;});

  const drawFeature = (featureData) => {
    console.log(featureData);
    initialized = true;

    let svgSelect = d3.select(svg);

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

    let content = svgSelect.append('g')
      .attr('class', 'content')
      .attr('transform', `translate(${svgPadding.left}, ${svgPadding.top})`);

    // We want to draw the categorical variable with more levels on the x-axis,
    // and the other one on the y-axis
    let data = preProcessData(featureData);

    // Some constant lengths of different elements
    // Approximate the longest width of score (y-axis)
    const yAxisWidth = 6 * d3.max(data.shortBinName.map(d => String(d).length));

    const legendConfig = {
      startColor: '#2166ac',
      endColor: '#b2182b',
      width: 180,
      height: 6
    };
    const legendHeight = legendConfig.height;
    
    const chartWidth = width - svgPadding.left - svgPadding.right - yAxisWidth;
    const chartHeight = height - svgPadding.top - svgPadding.bottom - densityHeight - legendHeight;

    // We put longer categorical variable on the x-axis
    let xScale = d3.scalePoint()
      .domain(data.longBinName)
      .padding(config.scalePointPadding)
      .range([0, chartWidth]);

    // Shorter categorical variable on the y-axis
    let yScale = d3.scalePoint()
      .domain(data.shortBinName)
      .padding(config.scalePointPadding)
      .range([chartHeight, 0]);

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

    let additiveData = createAdditiveData(featureData, data);

    // Create color scale for the bar chart
    let maxAbsScore = 0;
    featureData.additive.forEach(curArray => {
      curArray.forEach(d => {
        if (Math.abs(d) > maxAbsScore) maxAbsScore = Math.abs(d);
      });
    });

    // One can consider to use the color scale to encode the global range
    // let maxAbsScore = Math.max(Math.abs(scoreRange[0]), Math.abs(scoreRange[1]));
    let colorScale = d3.scaleLinear()
      .domain([-maxAbsScore, 0, maxAbsScore])
      .range([legendConfig.startColor, 'white', legendConfig.endColor]);

    // Draw the scatter chart
    let scatterPlot = content.append('g')
      .attr('transform', `translate(${0}, ${legendHeight})`)
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

    scatterPlot.append('clipPath')
      .attr('id', `${featureData.name.replace(/\s/g, '')}-y-axis-clip`)
      .append('rect')
      .attr('x', -svgPadding.left)
      .attr('y', 0)
      .attr('width', svgPadding.left + yAxisWidth)
      .attr('height', chartHeight);
    
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
    scatterPlotContent.append('g')
      .attr('class', 'scatter-plot-grid-group');

    let scatterGroup = scatterPlotContent.append('g')
      .attr('class', 'scatter-plot-scatter-group');

    let axisGroup = scatterPlot.append('g')
      .attr('class', 'axis-group');

    // Draw the scatter plot
    scatterGroup.style('stroke', 'hsla(0, 0%, 0%, 0.5)')
      .style('stroke-width', 1)
      .selectAll('circle.dot')
      .data(additiveData)
      .join('circle')
      .attr('class', 'dot')
      .attr('cx', d => xScale(d.longLabelName))
      .attr('cy', d => yScale(d.shortLabelName))
      .attr('r', config.catDotRadius)
      .style('fill', d => colorScale(d.additive));


    // Draw the line chart X axis
    let xAxisGroup = axisGroup.append('g')
      .attr('class', 'x-axis-wrapper')
      .attr('clip-path', `url(#${featureData.name.replace(/\s/g, '')}-x-axis-clip)`)
      .append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(${yAxisWidth}, ${chartHeight})`)
      .call(d3.axisBottom(xScale));
    
    xAxisGroup.attr('font-family', config.defaultFont);

    xAxisGroup.append('g')
      .attr('transform', `translate(${chartWidth / 2}, ${data.longBinName.length > 6 ? 65 : 25})`)
      .append('text')
      .attr('class', 'x-axis-text')
      .text(data.longName)
      .style('fill', 'black')
      .clone(true)
      .style('stroke', 'white')
      .style('stroke-width', 3)
      .lower();

    // Rotate the x axis label if there are too many values
    if (data.longBinName.length > 6) {
      xAxisGroup.selectAll('g.tick text')
        .attr('y', 0)
        .attr('x', 9)
        .attr('dy', '-0.6em')
        .attr('transform', 'rotate(90)')
        .style('text-anchor', 'start');
    }
    
    // Draw the line chart Y axis
    let yAxisGroup = axisGroup.append('g')
      .attr('class', 'y-axis-wrapper')
      .attr('clip-path', `url(#${featureData.name.replace(/\s/g, '')}-y-axis-clip)`)
      .append('g')
      .attr('class', 'y-axis')
      .attr('transform', `translate(${yAxisWidth}, 0)`);
    
    yAxisGroup.call(d3.axisLeft(yScale));
    yAxisGroup.attr('font-family', config.defaultFont);

    yAxisGroup.append('g')
      .attr('transform', `translate(${-yAxisWidth - 15}, ${chartHeight / 2}) rotate(-90)`)
      .append('text')
      .attr('class', 'y-axis-text')
      .text(data.shortName)
      .style('fill', 'black');

    // Draw a color legend
    let legendGroup = content.append('g')
      .attr('class', 'legend-group')
      .attr('transform', `translate(${width - legendConfig.width - svgPadding.right -
        svgPadding.left}, ${-20})`);
    
    drawHorizontalColorLegend(legendGroup, legendConfig, maxAbsScore);

    // Draw the cont histograms at the bottom
    let histData = [];
    
    // Transform the count to frequency (percentage)
    let histCountSum = d3.sum(data.longHistCount);
    let histFrequency = data.longHistCount.map(d => d / histCountSum);

    for (let i = 0; i < histFrequency.length; i++) {
      histData.push({
        x1: data.longHistEdgeName[i],
        x2: data.longHistEdgeName[i + 1],
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
      .attr('transform', `translate(${yAxisWidth}, ${chartHeight + legendHeight})`);

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

    yAxisHistGroup.attr('font-family', config.defaultFont);

    // Change 0.0 to 0
    yAxisHistGroup.selectAll('text')
      .style('fill', colors.histAxis)
      .filter((d, i, g) => d3.select(g[i]).text() === '0.0')
      .text('0');

    yAxisHistGroup.selectAll('path,line')
      .style('stroke', colors.histAxis);

    yAxisHistGroup.append('g')
      .attr('class', 'y-axis-text')
      .attr('transform', `translate(${-yAxisWidth - 15}, ${densityHeight / 2}) rotate(-90)`)
      .append('text')
      .text('density')
      .style('fill', colors.histAxis);


    // Add panning and zooming
    let zoom = d3.zoom()
      .scaleExtent(zoomScaleExtent)
      .translateExtent([[0, 0], [width, height]])
      .on('zoom', e => zoomed(e, xScale, yScale, svg, 2,
        1, yAxisWidth, chartWidth, chartHeight, legendHeight, multiMenu, component))
      .on('start', () => zoomStart(multiMenu))
      .on('end', () => zoomEnd(multiMenu))
      .filter(e => {
        if (selectMode) {
          return (e.type === 'wheel' || e.button === 2);
        } else {
          return (e.button === 0 || e.type === 'wheel');
        }
      });

    scatterPlotContent.call(zoom)
      .call(zoom.transform, d3.zoomIdentity);

    // Use animation as a signifier for zoom affordance
    // setTimeout(() => {
    //   scatterPlotContent.transition()
    //     .duration(400)
    //     .call(zoom.scaleTo, 0.95);
    // }, 400);

    scatterPlotContent.on('dblclick.zoom', null);
    
    // Listen to double click to reset zoom
    scatterPlotContent.on('dblclick', () => {
      scatterPlotContent.transition('reset')
        .duration(750)
        .ease(d3.easeCubicInOut)
        .call(zoom.transform, d3.zoomIdentity);
    });

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

  <!-- <div class='context-menu-container hidden' bind:this={multiMenu}>
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
  </div> -->

  <div class='svg-container'>
    <svg class='svg-explainer' bind:this={svg}></svg>
  </div>
  
</div>