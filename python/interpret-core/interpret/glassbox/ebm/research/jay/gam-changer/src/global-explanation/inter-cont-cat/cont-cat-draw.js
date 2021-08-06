import * as d3 from 'd3';

/**
 * Draw the plot in the SVG component
 * @param featureData
 */
export const drawFeatureLine = (featureData, svg, width, height, svgWidth,
  svgHeight, svgPadding, preProcessData, scoreRange, round, densityHeight,
  createAdditiveData, defaultFont, colors, zoomScaleExtent, zoomedLine,
  zoomStart, zoomEnd, component, multiMenu, selectMode) => {

  console.log(featureData);
  let svgSelect = d3.select(svg);

  // Set svg viewBox (3:2 WH ratio)
  svgSelect.attr('viewBox', `0 0 ${width} ${height}`)
    .attr('preserveAspectRatio', 'xMinYMin meet')
    .attr('width', svgWidth)
    .attr('height', svgHeight);

  let content = svgSelect.append('g')
    .attr('class', 'content')
    .attr('transform', `translate(${svgPadding.left}, ${svgPadding.top})`);

  // We want to draw continuous variable on the x-axis, and categorical variable
  // as lines. Need to figure out individual variable type.
  let data = preProcessData(featureData);

  console.log(data);

  // Some constant lengths of different elements
  // Approximate the longest width of score (y-axis)
  const yAxisWidth = 5 * d3.max(scoreRange.map(d => String(round(d, 1)).length));

  let legendConfig = {
    maxWidth: 100,
    leftMargin: 15,
    rightMargin: 8,
    lineHeight: 21,
    rectWidth: 25,
    rectHeight: 3,
    rectGap: 7,
    leftPadding: 5,
    topPadding: 8,
    btmPadding: 15
  };

  // Pre-populate the categorical variable legend to compute its width
  let hiddenLegendGroup = content.append('g')
    .style('visibility', 'hidden');

  hiddenLegendGroup.append('text')
    .attr('class', 'legend-title')
    .text(data.catName);

  let hiddenLegendContent = hiddenLegendGroup.append('g')
    .attr('transform', `translate(${0}, ${legendConfig.lineHeight})`);

  let hiddenLegendValues = hiddenLegendContent.selectAll('g.legend-value')
    .data(data.catHistEdge)
    .join('g')
    .attr('class', 'legend-value')
    .attr('transform', (d, i) => `translate(${0}, ${i * legendConfig.lineHeight})`);

  hiddenLegendValues.append('rect')
    .attr('y', -legendConfig.rectHeight / 2)
    .attr('width', legendConfig.rectWidth)
    .attr('height', legendConfig.rectHeight)
    .style('fill', 'navy');

  hiddenLegendValues.append('text')
    .attr('x', legendConfig.rectWidth + legendConfig.rectGap)
    .text(d => d);

  // Get the width and height of the legend box
  let bbox = hiddenLegendGroup.node().getBBox();

  // TODO: need to handle case where categorical labels are too long
  legendConfig.width = Math.min(round(bbox.width, 2), legendConfig.maxWidth);
  legendConfig.height = round(bbox.height, 2);

  // Compute the offset for the content box so we can center it
  let innerBbox = hiddenLegendContent.node().getBBox();
  legendConfig.centerOffset = (legendConfig.width - round(innerBbox.width, 2)) / 2;

  hiddenLegendGroup.remove();

  const chartWidth = width - svgPadding.left - svgPadding.right - yAxisWidth -
    legendConfig.width - legendConfig.rightMargin - legendConfig.leftMargin;
  const chartHeight = height - svgPadding.top - svgPadding.bottom - densityHeight;

  let xMin = data.contBinLabel[0];
  let xMax = data.contBinLabel[data.contBinLabel.length - 1];

  let xScale = d3.scaleLinear()
    .domain([xMin, xMax])
    .range([0, chartWidth]);

  // For the y scale, it seems InterpretML presets the center at 0 (offset
  // doesn't really matter in EBM because we can modify intercept)
  // TODO: Provide interaction for users to change the center point
  // Normalize the Y axis by the global score range
  let yScale = d3.scaleLinear()
    .domain(scoreRange)
    .range([chartHeight, 0]);

  // Create a data array by combining the bin labels, additive terms, and errors
  // Each line only counts additive term at one categorical level
  let additiveData = createAdditiveData(featureData, data);

  console.log(additiveData);

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

  // Draw the line chart
  let lineChart = content.append('g')
    .attr('class', 'line-chart-group');

  lineChart.append('clipPath')
    .attr('id', `${featureData.name.replace(/\s/g, '')}-chart-clip`)
    .append('rect')
    .attr('width', chartWidth)
    .attr('height', chartHeight - 1);

  lineChart.append('clipPath')
    .attr('id', `${featureData.name.replace(/\s/g, '')}-x-axis-clip`)
    .append('rect')
    .attr('x', yAxisWidth)
    .attr('y', chartHeight)
    .attr('width', chartWidth)
    .attr('height', densityHeight);

  let axisGroup = lineChart.append('g')
    .attr('class', 'axis-group');

  let lineChartContent = lineChart.append('g')
    .attr('class', 'line-chart-content-group')
    .attr('clip-path', `url(#${featureData.name.replace(/\s/g, '')}-chart-clip)`)
    .attr('transform', `translate(${yAxisWidth}, 0)`);

  // Append a rect so we can listen to events
  lineChartContent.append('rect')
    .attr('width', chartWidth)
    .attr('height', chartHeight)
    .style('opacity', 0);

  // Create a group to draw grids
  lineChartContent.append('g')
    .attr('class', 'line-chart-grid-group');

  let confidenceGroup = lineChartContent.append('g')
    .attr('class', 'line-chart-confidence-group');

  let lineGroup = lineChartContent.append('g')
    .attr('class', 'line-chart-line-group');

  // We draw the shape function with many line segments (path)
  // We draw it line by line
  let colorMap = new Map();

  for (let c = 0; c < additiveData.length; c++) {

    // Create line color
    let lineColor = d3.schemeTableau10[c];

    lineGroup.style('stroke-width', 2)
      .style('fill', 'none')
      .append('g')
      .style('stroke', lineColor)
      .attr('class', `line-group-${c}`)
      .selectAll('path')
      .data(additiveData[c])
      .join('path')
      .attr('class', 'additive-line-segment')
      .attr('d', d => {
        return `M ${xScale(d.sx)}, ${yScale(d.sAdditive)} L ${xScale(d.tx)}
            ${yScale(d.sAdditive)} L ${xScale(d.tx)}, ${yScale(d.tAdditive)}`;
      });

    // Draw the underlying confidence interval
    confidenceGroup.append('g')
      .attr('class', `confidence-group-${c}`)
      .selectAll('rect')
      .data(additiveData[c])
      .join('rect')
      .attr('class', 'line-confidence')
      .attr('x', d => xScale(d.sx))
      .attr('y', d => yScale(d.sAdditive + d.sError))
      .attr('width', d => xScale(d.tx) - xScale(d.sx))
      .attr('height', d => yScale(d.sAdditive - d.sError) - yScale(d.sAdditive + d.sError))
      .style('fill', lineColor)
      .style('opacity', 0.2);

    colorMap.set(data.catHistEdge[c], lineColor);
  }

  // Draw the chart X axis
  let xAxisGroup = axisGroup.append('g')
    .attr('class', 'x-axis')
    .attr('transform', `translate(${yAxisWidth}, ${chartHeight})`)
    .call(d3.axisBottom(xScale));

  xAxisGroup.attr('font-family', defaultFont);

  // Draw the chart Y axis
  let yAxisGroup = axisGroup.append('g')
    .attr('class', 'y-axis')
    .attr('transform', `translate(${yAxisWidth}, 0)`);

  yAxisGroup.call(d3.axisLeft(yScale));
  yAxisGroup.attr('font-family', defaultFont);

  yAxisGroup.append('g')
    .attr('transform', `translate(${-yAxisWidth - 15}, ${chartHeight / 2}) rotate(-90)`)
    .append('text')
    .attr('class', 'y-axis-text')
    .text('score')
    .style('fill', 'black');

  // Draw the histograms at the bottom
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

  let histWidth = Math.min(50, xScale(histData[0].x2) - xScale(histData[0].x1));

  // Draw the density histogram 
  let histChartContent = histChart.append('g')
    .attr('class', 'hist-chart-content-group')
    .attr('transform', `translate(${yAxisWidth}, ${chartHeight})`);

  histChartContent.selectAll('rect')
    .data(histData)
    .join('rect')
    .attr('class', 'hist-rect')
    .attr('x', d => xScale(d.x1))
    .attr('y', 0)
    .attr('width', histWidth)
    .attr('height', d => histYScale(d.height))
    .style('fill', colors.hist);

  // Draw a Y axis for the histogram chart
  let yAxisHistGroup = lineChart.append('g')
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
    .attr('class', 'y-axis-text')
    .attr('transform', `translate(${-yAxisWidth - 15}, ${densityHeight / 2}) rotate(-90)`)
    .append('text')
    .text('Density')
    .style('fill', colors.histAxis);

  // Add panning and zooming
  let zoom = d3.zoom()
    .scaleExtent(zoomScaleExtent)
    .on('zoom', e => zoomedLine(e, xScale, yScale, svg, 2,
      1, yAxisWidth, chartWidth, chartHeight, null, component))
    .on('start', () => zoomStart(multiMenu))
    .on('end', () => zoomEnd(multiMenu))
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

  // Draw a legend for the categorical data
  let legendGroup = content.append('g')
    .attr('class', 'legend-group')
    .attr('transform', `translate(${yAxisWidth + chartWidth + legendConfig.leftMargin}, ${0})`);

  legendGroup.append('rect')
    .attr('x', -legendConfig.leftPadding)
    .attr('y', -legendConfig.topPadding)
    .attr('width', legendConfig.width + legendConfig.leftPadding * 2)
    .attr('height', legendConfig.height + legendConfig.topPadding + legendConfig.btmPadding)
    .style('stroke', 'hsla(0, 0%, 0%, 0.1)')
    .style('fill', 'none');

  legendGroup.append('text')
    .attr('class', 'legend-title')
    .attr('x', legendConfig.width / 2)
    .style('text-anchor', 'middle')
    .text(data.catName);

  let legendContent = legendGroup.append('g')
    .attr('class', 'legend-content')
    .attr('transform', `translate(${legendConfig.centerOffset}, ${legendConfig.lineHeight + 10})`);

  let legendValues = legendContent.selectAll('g.legend-value')
    .data(data.catHistEdge)
    .join('g')
    .attr('class', 'legend-value')
    .attr('transform', (d, i) => `translate(${0}, ${i * legendConfig.lineHeight})`);

  legendValues.append('rect')
    .attr('y', -legendConfig.rectHeight / 2)
    .attr('width', legendConfig.rectWidth)
    .attr('height', legendConfig.rectHeight)
    .style('fill', d => colorMap.get(d));

  legendValues.append('text')
    .attr('x', legendConfig.rectWidth + legendConfig.rectGap)
    .text(d => d);

};