import * as d3 from 'd3';
import { config } from '../config';
import { round } from '../utils/utils';

const barHeight = 18;
const textHeight = 22;
const sectionGap = 28;

/**
* Function to draw a curve (PR curve or ROC curve).
*/
export const drawCurve = (curve, isPR, svg, lineGroup, groupColors) => {
  const svgPadding = { top: 10, right: 40, bottom: 40, left: 40 };
  const defaultFont = config.defaultFont;

  let axisColor = groupColors.axis;

  // Create data based on the curve type
  let xText = isPR ? 'Recall' : 'False Positive Rate';
  let yText = isPR ? 'Precision' : 'True Positive Rate';

  let chartWidth = 200 - svgPadding.left - svgPadding.right;
  let chartHeight = 170 - svgPadding.top - svgPadding.bottom;
  let yAxisWidth = 15;

  // Create x and y axis
  let xScale = d3.scaleLinear().domain([0, 1]).range([0, chartWidth]);
  let yScale = d3.scaleLinear().domain([0, 1]).range([chartHeight, 0]);

  let xAxis = d3.axisBottom(xScale).ticks(2);
  let yAxis = d3.axisLeft(yScale).ticks(2);

  // Draw axis if user does not specify the argument
  if (svg.select('.x-axis-group').size() === 0) {

    // Add border
    svg.append('rect')
      .attr('class', 'border-rect')
      .attr('x', svgPadding.left + yAxisWidth)
      .attr('y', svgPadding.top)
      .attr('width', chartWidth)
      .attr('height', chartHeight)
      .style('fill', 'none')
      .style('stroke', axisColor);

    // Add axis to the plot
    svg.append('g')
      .attr('class', 'x-axis-group')
      .attr('transform', `translate(${svgPadding.left + yAxisWidth}, ${chartHeight + svgPadding.top})`)
      .call(xAxis)
      .attr('font-family', defaultFont)
      .style('color', axisColor)
      .append('text')
      .attr('class', 'axis-label')
      .attr('x', chartWidth / 2)
      .attr('y', 30)
      .attr('fill', axisColor)
      .style('text-anchor', 'middle')
      .text(xText);

    svg.append('g')
      .attr('class', 'y-axis-group')
      .attr('transform', `translate(${svgPadding.left + yAxisWidth}, ${svgPadding.top})`)
      .call(yAxis)
      .attr('font-family', defaultFont)
      .style('color', axisColor)
      .append('text')
      .attr('transform', 'rotate(-90)')
      .attr('class', 'axis-label')
      .attr('x', -chartHeight / 2)
      .attr('y', -30)
      .attr('fill', axisColor)
      .style('text-anchor', 'middle')
      .text(yText);
  }

  // Generate line path
  let lineValue = d3.line()
    .curve(d3.curveStepAfter)
    .x(d => xScale(round(d[1], 4)))
    .y(d => yScale(round(d[0], 4)));

  // Add lines to the plot
  svg.selectAll(`g.${lineGroup}`)
    .data([curve])
    .join(
      enter => enter.append('g')
        .attr('class', `${lineGroup}`)
        .attr('transform', `translate(${svgPadding.left + yAxisWidth}, ${svgPadding.top})`)
        .append('path')
        .attr('d', d => lineValue(d))
        .style('stroke', groupColors[lineGroup])
        .style('fill', 'none'),
      update => update.select('path')
        .attr('d', d => lineValue(d)),
      exit => exit.remove()
    );

};

/**
 * Draw the bar charts in the classification metric tab. It also allocates space
 * for the confusion matrix at the bottom of the bar chart.
 * @param width Sidebar total width
 * @param svgPadding Object describing the paddings of the sidebar content
 * @param component Sidebar component
 * @param barData Data of metrics
 */
export const drawClassificationBarChart = (width, svgPadding, component, barData) => {

  const lineY = barHeight * 3 + textHeight + sectionGap / 2 + 2;
  const lineWidth = width - svgPadding.left - svgPadding.right;

  let svg = d3.select(component)
    .select('.bar-svg');

  const groupData = [
    { name: 'accuracy', text: 'Accuracy' },
    { name: 'balancedAccuracy', text: 'Balanced Accuracy' },
    { name: 'rocAuc', text: 'ROC AUC' },
    { name: 'confusionMatrix', text: 'Confusion Matrix' }
  ];

  let widthScale = d3.scaleLinear()
    .domain([0, 1])
    .range([0, width - svgPadding.left - svgPadding.right])
    .unknown(25);

  const rectOrder = ['original', 'last', 'current'];

  // Initialize the group structure if it is the first call
  if (svg.select('.bar-group').size() === 0) {

    let barGroup = svg.append('g')
      .attr('class', 'bar-group')
      .attr('transform', `translate(0, ${10})`);

    // Add three bar chart groups
    let bars = barGroup.selectAll('g.bar')
      .data(groupData)
      .join('g')
      .attr('class', d => `bar ${d.name}-group`)
      .attr('transform', (d, i) => `translate(${svgPadding.left}, ${i * (3 * barHeight + textHeight + sectionGap)})`);

    bars.append('text')
      .attr('class', 'metric-title')
      .text(d => d.text);

    bars.append('path')
      .attr('d', `M ${0}, ${lineY} L ${lineWidth}, ${lineY}`)
      .style('stroke', 'hsla(0, 0%, 0%, 0.2)')
      .style('visibility', d => d.name === 'confusionMatrix' ? 'hidden' : 'show');

    // Add color legend next to Accuracy
    let legendGroup = barGroup.select('g.accuracy-group');

    const legendData = [
      { name: 'origin', class: 'original', title: 'Metrics of the original graph', width: 42, x: 0 },
      { name: 'last', class: 'last', title: 'Metrics of the last edit', width: 28, x: 47 },
      { name: 'current', class: 'current', title: 'Metrics of the current graph', width: 50, x: 80 }
    ];

    let items = legendGroup.selectAll('g.legend-item')
      .data(legendData)
      .join('g')
      .style('cursor', 'default')
      .attr('transform', d => `translate(${80 + d.x}, 0)`);

    items.append('title')
      .text(d => d.title);

    items.append('rect')
      .attr('width', d => d.width)
      .attr('height', 16)
      .attr('rx', 3)
      .attr('class', d => d.class);

    items.append('text')
      .attr('class', 'legend-title')
      .attr('y', 2)
      .attr('x', d => d.width / 2)
      .text(d => d.name);
    
    // Create bars and texts
    Object.keys(barData).forEach(k => {

      barGroup.select(`.${k}-group`)
        .selectAll('rect.bar')
        .data(barData[k].slice(0, 3))
        .join('rect')
        .attr('class', (d, i) => `bar ${rectOrder[i]}`)
        .attr('y', (d, i) => (i) * (barHeight + 0) + textHeight)
        .attr('width', d => widthScale(d))
        .attr('height', barHeight);

      barGroup.select(`.${k}-group`)
        .selectAll('text.bar')
        .data(barData[k].slice(0, 3))
        .join('text')
        .attr('class', (d, i) => `bar-label ${rectOrder[i]}`)
        .attr('x', 3)
        .attr('y', (d, i) => (i) * (barHeight + 0) + barHeight / 2 + textHeight + 1)
        .text(d => round(d, 4));
    });
  }

  // Update the bars

  let barGroup = svg.select('.bar-group');

  Object.keys(barData).forEach(k => {
    for(let i = 0; i < 3; i++) {
      barGroup.select(`.${k}-group`)
        .select(`rect.bar.${rectOrder[i]}`)
        .transition('update-metric')
        .duration(200)
        .attr('width', widthScale(barData[k][i]));
      
      barGroup.select(`.${k}-group`)
        .select(`text.bar-label.${rectOrder[i]}`)
        .text(barData[k][i] === null || isNaN(barData[k][i]) ? 'NA' : round(barData[k][i], 4));
    }
  });

};

/**
 * Draw the confusion matrix
 * @param width Sidebar total width
 * @param svgPadding Object describing the paddings of the sidebar content
 * @param component Sidebar component
 */
export const drawConfusionMatrix = (width, svgPadding, component, confusionMatrixData) => {

  let barGroup = d3.select(component)
    .select('.bar-svg')
    .select('.bar-group');

  // Draw the confusion matrix if it has not been created yet
  if (barGroup.select('.confusion-matrix-content').empty()) {
    let matrixGroup = barGroup.select('.confusionMatrix-group')
      .append('g')
      .attr('class', 'confusion-matrix-content')
      .attr('transform', `translate(0, ${textHeight})`);

    // Compute the rectangle width
    const middleGap = 6;
    const rectHeight = 20;
    const explanationHeight = 14;
    const explanationWidth = 40;
    const sRectWidth = (width - svgPadding.left - svgPadding.right - middleGap - explanationWidth) / 4;
    const lRectWidth = 2 * sRectWidth;

    const drawRectGroup = (curGroup, data) => {
      // Draw the rectangles
      curGroup.selectAll('rect.matrix-element')
        .data(data)
        .join('rect')
        .attr('class', d => `matrix-element ${d.group}`)
        .attr('x', d => d.x)
        .attr('y', d => d.y)
        .attr('width', d => d.width)
        .attr('height', rectHeight);

      // Draw the text
      curGroup.selectAll('text.matrix-label')
        .data(data)
        .join('text')
        .attr('class', d => `matrix-label ${d.group}`)
        .attr('x', d => d.x + d.width / 2)
        .attr('y', d => d.y + rectHeight / 2 + 1)
        .text('32');
    };

    const topData = [
      { x: 0, y: 0, width: sRectWidth, group: 'original' },
      { x: sRectWidth, y: 0, width: sRectWidth, group: 'last' },
      { x: 0, y: rectHeight, width: lRectWidth, group: 'current' }
    ];

    const botData = [
      { x: 0, y: rectHeight, width: sRectWidth, group: 'original' },
      { x: sRectWidth, y: rectHeight, width: sRectWidth, group: 'last' },
      { x: 0, y: 0, width: lRectWidth, group: 'current' }
    ];

    // TP group
    let matSubGroup = matrixGroup.append('g')
      .attr('class', 'matrix-group-tp')
      .attr('transform', `translate(${0}, ${0})`);

    matSubGroup.append('text')
      .attr('class', 'matrix-explanation')
      .text('Predicted Yes');

    matSubGroup.append('g', 'matrix-element')
      .attr('transform', `translate(${0}, ${explanationHeight})`)
      .call(drawRectGroup, topData);

    // FP group
    matSubGroup = matrixGroup.append('g')
      .attr('class', 'matrix-group-fp')
      .attr('transform', `translate(${lRectWidth + middleGap}, ${0})`);

    matSubGroup.append('text')
      .attr('class', 'matrix-explanation')
      .text('Predicted No');

    let curText = matSubGroup.append('text')
      .attr('class', 'matrix-explanation dominant-middle')
      .attr('y', rectHeight + explanationHeight - explanationHeight / 2 + 2);

    curText.append('tspan')
      .attr('class', 'dominant-middle')
      .attr('x', lRectWidth + 3)
      .attr('dy', 0)
      .text('Actual');

    curText.append('tspan')
      .attr('class', 'dominant-middle')
      .attr('x', lRectWidth + 3)
      .attr('dy', '1em')
      .text('Yes');

    matSubGroup.append('g', 'matrix-element')
      .attr('transform', `translate(${0}, ${explanationHeight})`)
      .call(drawRectGroup, topData);

    // FN group
    matSubGroup = matrixGroup.append('g')
      .attr('class', 'matrix-group-fn')
      .attr('transform', `translate(${0}, ${2 * rectHeight + middleGap + explanationHeight})`);

    matSubGroup.append('g', 'matrix-element')
      .attr('transform', `translate(${0}, ${0})`)
      .call(drawRectGroup, botData);

    // TN group
    matSubGroup = matrixGroup.append('g')
      .attr('class', 'matrix-group-tn')
      .attr('transform', `translate(${lRectWidth + middleGap}, ${2 * rectHeight + middleGap + explanationHeight})`);

    curText = matSubGroup.append('text')
      .attr('class', 'matrix-explanation dominant-middle')
      .attr('y', rectHeight - explanationHeight / 2 + 2);

    curText.append('tspan')
      .attr('class', 'dominant-middle')
      .attr('x', lRectWidth + 3)
      .attr('dy', 0)
      .text('Actual');

    curText.append('tspan')
      .attr('class', 'dominant-middle')
      .attr('x', lRectWidth + 3)
      .attr('dy', '1em')
      .text('No');

    matSubGroup.append('g', 'matrix-element')
      .attr('transform', `translate(${0}, ${0})`)
      .call(drawRectGroup, botData);
  }

  // Update the text in the matrix with confusionMatrixData
  let contentGroup = barGroup.select('.confusion-matrix-content');

  Object.keys(confusionMatrixData).forEach(k => {
    let groups = ['original', 'last', 'current'];

    for (let i = 0; i < groups.length; i++) {
      let g = groups[i];
      let curText;

      if (confusionMatrixData[k][i] === null) {
        curText = 'NA';
      } else {
        curText = round(confusionMatrixData[k][i] * 100, 1);
      }

      if (g === 'current') {
        curText = `${curText}%`;
      }

      contentGroup.select(`.matrix-group-${k}`)
        .select(`.matrix-label.${g}`)
        .text(curText);
    }


  });



};
