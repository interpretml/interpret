<script>
  import * as d3 from 'd3';
  import { onMount } from 'svelte';
  import { round } from '../utils/utils';
  import { config } from '../config';
  import { drawHorizontalColorLegend } from './draw';

  import { state } from './inter-cont-cat/cont-cat-state';
  import { SelectedInfo } from './inter-cont-cat/cont-cat-class';
  import { drawFeatureLine } from './inter-cont-cat/cont-cat-draw';
  import { zoomStart, zoomEnd, zoomedLine, zoomedBar, zoomScaleExtent, rExtent } from './inter-cont-cat/cont-cat-zoom';
  import { brushDuring, brushEndSelect } from './inter-cont-cat/cont-cat-brush';

  import ToggleSwitch from '../components/ToggleSwitch.svelte';
  // import ContextMenu from '../components/ContextMenu.svelte';

  export let featureData = null;
  export let labelEncoder = null;
  export let scoreRange = null;
  export let svgHeight = 400;
  export let chartType = 'bar';

  let svg = null;
  let component = null;
  let multiMenu = null;
  let myContextMenu = null;
  let multiMenuControlInfo = null;

  let mounted = false;
  let initialized = false;

  // Interactions
  let selectMode = false;
  state.selectedInfo = new SelectedInfo();

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
  const defaultFont = config.defaultFont;

  /**
   * Create additiveData which is used to draw line segments for each categorical
   * level.
   * @param {[object]} featureData Original feature data passed from the parent component
   * @param {[object]} data Processed feature data (separated by cont/cat variables)
   */
  const createAdditiveData = (featureData, data) => {
    let additiveData = [];

    if (data.catDim === 0) {
      for (let c = 0; c < featureData.additive.length; c++) {
        let curValues = [];

        for (let i = 0; i < featureData.additive[0].length - 1; i++) {
          curValues.push({
            sx: data.contBinLabel[i],
            sAdditive: featureData.additive[c][i],
            sError: featureData.error[c][i],
            tx: data.contBinLabel[i + 1],
            tAdditive: featureData.additive[c][i + 1],
            tError: featureData.error[c][i + 1]
          });
        }

        // Finally, add the ending point (xMax without additive value)
        // We would use the second last point's additive value and error value
        let endI = featureData.additive[0].length - 1;
        curValues.push({
          sx: data.contBinLabel[endI],
          sAdditive: featureData.additive[c][endI],
          sError: featureData.error[c][endI],
          tx: data.contBinLabel[endI + 1],
          tAdditive: featureData.additive[c][endI],
          tError: featureData.error[c][endI]
        });

        additiveData.push(curValues);
      }
    } else {
      for (let c = 0; c < featureData.additive[0].length; c++) {
        let curValues = [];

        for (let i = 0; i < featureData.additive.length - 1; i++) {
          curValues.push({
            sx: data.contBinLabel[i],
            sAdditive: featureData.additive[i][c],
            sError: featureData.error[i][c],
            tx: data.contBinLabel[i + 1],
            tAdditive: featureData.additive[i + 1][c],
            tError: featureData.error[i + 1][c]
          });
        }

        // Finally, add the ending point (xMax without additive value)
        // We would use the second last point's additive value and error value
        let endI = featureData.additive.length - 1;
        curValues.push({
          sx: data.contBinLabel[endI],
          sAdditive: featureData.additive[endI][c],
          sError: featureData.error[endI][c],
          tx: data.contBinLabel[endI + 1],
          tAdditive: featureData.additive[endI][c],
          tError: featureData.error[endI][c]
        });

        additiveData.push(curValues);
      }
    }

    return additiveData;
  };

  /**
   * Separate feature data into categorical and continuous variables.
   * @param {object} featureData
   */
  const preProcessData = (featureData) => {
    let data = {};

    if (featureData.type1 === 'continuous' && featureData.type2 === 'categorical') {
      data.contBinLabel = featureData.binLabel1;
      data.catBinLabel = featureData.binLabel2;
      data.contName = featureData.name1;
      data.catName = featureData.name2;
      data.contHistEdge = featureData.histEdge1;
      data.catHistEdge = featureData.histEdge2;
      data.contHistCount = featureData.histCount1;
      data.catHistCount = featureData.histCount2;
      data.contDim = 0;
      data.catDim = 1;
    } else if (featureData.type2 === 'continuous' && featureData.type1 === 'categorical'){
      data.contBinLabel = featureData.binLabel2;
      data.catBinLabel = featureData.binLabel1;
      data.contName = featureData.name2;
      data.catName = featureData.name1;
      data.contHistEdge = featureData.histEdge2;
      data.catHistEdge = featureData.histEdge1;
      data.contHistCount = featureData.histCount2;
      data.catHistCount = featureData.histCount1;
      data.contDim = 1;
      data.catDim = 0;
    } else {
      console.error('The interaction is not continuous x categorical.');
    }

    // Encode the cat var level
    data.catBinName = data.catBinLabel.map(d => labelEncoder[data.catName][d]);
    data.catHistEdgeName = data.catHistEdge.map(d => labelEncoder[data.catName][d]);

    return data;
  };

  onMount(() => {mounted = true;});

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

    // We want to draw continuous variable on the x-axis, and categorical variable
    // as lines. Need to figure out individual variable type.
    let data = preProcessData(featureData);

    // Some constant lengths of different elements
    // Approximate the longest width of score (y-axis)
    const yAxisWidth = 6 * d3.max(data.catHistEdgeName.map(d => String(d).length));

    const legendConfig = {
      startColor: '#2166ac',
      endColor: '#b2182b',
      width: 180,
      height: 6
    };
    const legendHeight = legendConfig.height;
    
    const chartWidth = width - svgPadding.left - svgPadding.right - yAxisWidth;
    const chartHeight = height - svgPadding.top - svgPadding.bottom - densityHeight - legendHeight;

    // We put continuous variable on the x-axis
    let xMin = data.contBinLabel[0];
    let xMax = data.contBinLabel[data.contBinLabel.length - 1];

    let xScale = d3.scaleLinear()
      .domain([xMin, xMax])
      .range([0, chartWidth]);

    // Categorical variable on the y-axis
    let yScale = d3.scaleBand()
      .domain(data.catHistEdgeName)
      .paddingInner(0.4)
      .paddingOuter(0.3)
      .range([chartHeight, 0])
      .round(true);

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
    additiveData.forEach(curArray => {
      curArray.forEach(d => {
        if (Math.abs(d.sAdditive) > maxAbsScore) maxAbsScore = Math.abs(d.sAdditive);
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

    barChart.append('clipPath')
      .attr('id', `${featureData.name.replace(/\s/g, '')}-y-axis-clip`)
      .append('rect')
      .attr('x', -svgPadding.left)
      .attr('y', 0)
      .attr('width', svgPadding.left + yAxisWidth)
      .attr('height', chartHeight);
    
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
      .attr('class', 'bar-plot-grid-group');

    let barGroup = barChartContent.append('g')
      .attr('class', 'bar-chart-bar-group')
      .style('stroke', 'hsla(0, 0%, 0%, 0.06)');

    let axisGroup = barChart.append('g')
      .attr('class', 'axis-group');

    // Draw the bars one by one (iterate through the categorical levels)
    for (let l = 0; l < additiveData.length; l++) {
      barGroup.append('g')
        .attr('class', `bar-group-${l}`)
        .attr('transform', `translate(${0}, ${yScale(data.catBinName[l])})`)
        .selectAll('rect.bar')
        .data(additiveData[l])
        .join('rect')
        .attr('class', 'bar')
        .attr('x', d => xScale(d.sx))
        .attr('y', 0)
        .attr('width', d => xScale(d.tx) - xScale(d.sx))
        .attr('height', yScale.bandwidth())
        .style('fill', d => colorScale(d.sAdditive));
    }

    // Draw the line chart X axis
    let xAxisGroup = axisGroup.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(${yAxisWidth}, ${chartHeight})`)
      .call(d3.axisBottom(xScale));
    
    xAxisGroup.attr('font-family', defaultFont);

    xAxisGroup.append('g')
      .attr('transform', `translate(${chartWidth / 2}, ${25})`)
      .append('text')
      .attr('class', 'x-axis-text')
      .text(data.contName)
      .style('fill', 'black');
    
    // Draw the line chart Y axis
    let yAxisGroup = axisGroup.append('g')
      .attr('class', 'y-axis-wrapper')
      .attr('clip-path', `url(#${featureData.name.replace(/\s/g, '')}-y-axis-clip)`)
      .append('g')
      .attr('class', 'y-axis')
      .attr('transform', `translate(${yAxisWidth}, 0)`);
    
    yAxisGroup.call(d3.axisLeft(yScale));
    yAxisGroup.attr('font-family', defaultFont);

    yAxisGroup.append('g')
      .attr('class', 'y-axis-text')
      .attr('transform', `translate(${-yAxisWidth - 15}, ${chartHeight / 2}) rotate(-90)`)
      .append('text')
      .text(data.catName)
      .style('fill', 'black');

    // Add panning and zooming
    let zoom = d3.zoom()
      .translateExtent([[-Infinity, 0], [Infinity, height]])
      .scaleExtent(zoomScaleExtent)
      .on('zoom', e => zoomedBar(e, xScale, yScale, svg, 2,
        1, yAxisWidth, chartWidth, chartHeight, legendHeight, null, component))
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

    // Use animation as a signifier for zoom affordance
    // setTimeout(() => {
    //   barChartContent.transition()
    //     .duration(400)
    //     .call(zoom.scaleTo, 0.95);
    // }, 400);
    
    // Listen to double click to reset zoom
    barChartContent.on('dblclick', () => {
      barChartContent.transition('reset')
        .duration(750)
        .ease(d3.easeCubicInOut)
        .call(zoom.transform, d3.zoomIdentity);
    });
    
    // Draw a color legend
    let legendGroup = content.append('g')
      .attr('class', 'legend-group')
      .attr('transform', `translate(${width - legendConfig.width - svgPadding.right - svgPadding.left}, ${-20})`);
    
    drawHorizontalColorLegend(legendGroup, legendConfig, maxAbsScore);

    // Draw the cont histograms at the bottom
    let histData = [];
    
    // Transform the count to frequency (percentage)
    let histCountSum = d3.sum(data.contHistCount);
    let histFrequency = data.contHistCount.map(d => d / histCountSum);

    for (let i = 0; i < histFrequency.length; i++) {
      histData.push({
        x1: data.contHistEdge[i],
        x2: data.contHistEdge[i + 1],
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
      .attr('class', 'y-axis-hist')
      .attr('transform', `translate(${yAxisWidth}, ${chartHeight})`)
      .lower();
    
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
  };

  let drawFeature = null;

  if (chartType === 'bar') {
    drawFeature = drawFeatureBar;
  } else if (chartType === 'line') {
    drawFeature = (featureData) => {
      drawFeatureLine(featureData, svg, width,
        height, svgWidth, svgHeight, svgPadding, preProcessData, scoreRange,
        round, densityHeight, createAdditiveData, defaultFont, colors,
        zoomScaleExtent, zoomedLine,  zoomStart, zoomEnd, component, multiMenu,
        selectMode);
      initialized = true;
    };
  } else {
    console.error('The provided chart type is not supported.');
  }

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

  $: featureData && mounted && !initialized && drawFeatureBar(featureData);

</script>

<style type='text/scss'>
  @import '../define';
  @import './common.scss';

  :global(.explain-panel .hidden) {
    visibility: hidden;
    pointer-events: none;
  }

  :global(.explain-panel .bar-chart-content-group) {
    cursor: grab;
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