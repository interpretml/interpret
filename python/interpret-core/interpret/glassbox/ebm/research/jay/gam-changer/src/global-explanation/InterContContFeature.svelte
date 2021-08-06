<script>
  import * as d3 from 'd3';
  import { onMount } from 'svelte';
  import { round, transpose2dArray } from '../utils/utils';
  import { config } from '../config';
  import { drawHorizontalColorLegend } from './draw';

  import { state } from './inter-cont-cont/cont-cont-state';
  import { SelectedInfo } from './inter-cont-cont/cont-cont-class';
  import { zoomStart, zoomEnd, zoomed, zoomScaleExtent, rExtent } from './inter-cont-cont/cont-cont-zoom';
  import { brushDuring, brushEndSelect } from './inter-cont-cont/cont-cont-brush';

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

  // Select mode
  let selectMode = false;
  state.selectedInfo = new SelectedInfo();

  // Show some hidden elements for development
  const showRuler = false;

  // Colors
  const colors = config.colors;

  const drawFeatureBar = (featureData) => {
    initialized = true;
    console.log(featureData);

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

    // Some constant lengths of different elements
    // Approximate the longest width of score (y-axis)
    const yAxisWidth = 5 * d3.max(featureData.binLabel2.map(d => String(round(d, 2)).length + 1));
    const legendConfig = {
      startColor: '#2166ac',
      endColor: '#b2182b',
      width: 180,
      height: 6
    };
    const legendHeight = legendConfig.height;
    
    const chartWidth = width - svgPadding.left - svgPadding.right - yAxisWidth;
    const chartHeight = height - svgPadding.top - svgPadding.bottom - densityHeight - legendHeight;

    // We put continuous 1 on the x-axis
    let xMin = featureData.binLabel1[0];
    let xMax = featureData.binLabel1[featureData.binLabel1.length - 1];

    let xScale = d3.scaleLinear()
      .domain([xMin, xMax])
      .range([0, chartWidth]);

    // Continuous 2 on the y-axis
    let yMin = featureData.binLabel2[0];
    let yMax = featureData.binLabel2[featureData.binLabel2.length - 1];

    let yScale = d3.scaleLinear()
      .domain([yMin, yMax])
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

    let additiveData = transpose2dArray(featureData.additive);

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

    // Draw the bar chart
    let barChart = content.append('g')
      .attr('transform', `translate(${0}, ${legendHeight})`)
      .attr('class', 'bar-chart-group');

    barChart.append('clipPath')
      .attr('id', `${featureData.name.replace(/\s/g, '')}-chart-clip`)
      .append('rect')
      .attr('width', chartWidth)
      .attr('height', chartHeight - 1);
    
    let barChartContent = barChart.append('g')
      .attr('class', 'bar-chart-content-group')
      .attr('clip-path', `url(#${featureData.name.replace(/\s/g, '')}-chart-clip)`)
      .attr('transform', `translate(${yAxisWidth}, 0)`);

    // Append a rect so we can listen to events
    barChartContent.append('rect')
      .attr('width', chartWidth)
      .attr('height', chartHeight)
      .style('opacity', 0);

    // Create a group to draw grids
    barChartContent.append('g')
      .attr('class', 'bar-chart-grid-group');

    let barGroup = barChartContent.append('g')
      .attr('class', 'bar-chart-bar-group')
      .style('stroke', 'hsla(0, 0%, 0%, 0.05)')
      .style('stroke-width', 1);

    let axisGroup = barChart.append('g')
      .attr('class', 'axis-group');

    // Draw the bars one by one (iterate through continuous 2 at y-axis)
    for (let l = 0; l < featureData.additive[0].length; l++) {
      let curHeight = yScale(featureData.binLabel2[l]) - yScale(featureData.binLabel2[l + 1]);

      barGroup.append('g')
        .attr('class', `bar-group-${l}`)
        .attr('transform', `translate(${0}, ${yScale(featureData.binLabel2[l])})`)
        .selectAll('rect.bar')
        .data(additiveData[l])
        .join('rect')
        .attr('class', 'bar')
        .attr('x', (d, i) => xScale(featureData.binLabel1[i]))
        .attr('y', -curHeight)
        .attr('width', (d, i) => xScale(featureData.binLabel1[i + 1]) - xScale(featureData.binLabel1[i]))
        .attr('height', curHeight)
        .style('fill', d => colorScale(d));
    }

    // Draw the line chart X axis
    let xAxisGroup = axisGroup.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(${yAxisWidth}, ${chartHeight})`)
      .call(d3.axisBottom(xScale));
    
    xAxisGroup.attr('font-family', config.defaultFont);

    xAxisGroup.append('g')
      .attr('transform', `translate(${chartWidth / 2}, ${25})`)
      .append('text')
      .attr('class', 'x-axis-text')
      .text(featureData.name1)
      .style('fill', 'black');
    
    // Draw the line chart Y axis
    let yAxisGroup = axisGroup.append('g')
      .attr('class', 'y-axis')
      .attr('transform', `translate(${yAxisWidth}, 0)`);
    
    yAxisGroup.call(d3.axisLeft(yScale));
    yAxisGroup.attr('font-family', config.defaultFont);

    yAxisGroup.append('g')
      .attr('transform', `translate(${-yAxisWidth - 15}, ${chartHeight / 2}) rotate(-90)`)
      .append('text')
      .attr('class', 'y-axis-text')
      .text(featureData.name2)
      .style('fill', 'black');
    
    // Draw a color legend
    let legendGroup = content.append('g')
      .attr('class', 'legend-group')
      .attr('transform', `translate(${width - legendConfig.width -
        svgPadding.right - svgPadding.left}, ${-20})`);
    
    drawHorizontalColorLegend(legendGroup, legendConfig, maxAbsScore);

    // Draw the cont histograms at the bottom
    let histData = [];
    
    // Transform the count to frequency (percentage)
    let histCountSum = d3.sum(featureData.histCount1);
    let histFrequency = featureData.histCount1.map(d => d / histCountSum);

    for (let i = 0; i < histFrequency.length; i++) {
      histData.push({
        x1: featureData.histEdge1[i],
        x2: featureData.histEdge1[i + 1],
        height: histFrequency[i]
      });
    }

    let histYScale = d3.scaleLinear()
      .domain(d3.extent(histFrequency))
      .range([0, densityHeight]);

    // Draw the density histogram 
    let histChartContent = histChart.append('g')
      .attr('class', 'hist-chart-content-group')
      .attr('transform', `translate(${yAxisWidth}, ${chartHeight + legendHeight})`);

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
    let yAxisHistGroup = barChart.append('g')
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
      .text('Density')
      .style('fill', colors.histAxis);

    // Add panning and zooming
    let zoom = d3.zoom()
      .scaleExtent(zoomScaleExtent)
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

    barChartContent.call(zoom)
      .call(zoom.transform, d3.zoomIdentity);

    barChartContent.on('dblclick.zoom', null);
    
    // Listen to double click to reset zoom
    barChartContent.on('dblclick', () => {
      barChartContent.transition('reset')
        .duration(750)
        .ease(d3.easeCubicInOut)
        .call(zoom.transform, d3.zoomIdentity);
    });

    // Use animation as a signifier for zoom affordance
    setTimeout(() => {
      barChartContent.transition()
        .duration(400)
        .call(zoom.scaleTo, 0.95);
    }, 400);

  };

  onMount(() => {mounted = true;});

  const drawFeature = drawFeatureBar;

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

  :global(.explain-panel .legend-title) {
    font-size: 0.9em;
    dominant-baseline: hanging;
  }

  :global(.explain-panel .legend-value) {
    font-size: 13px;
    dominant-baseline: middle;
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