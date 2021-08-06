import * as d3 from 'd3';
import { config } from '../config';

/**
 * Create a horizontal color legend.
 * @param legendGroup
 * @param legendConfig
 * @param largestAbs
 */
export const drawHorizontalColorLegend = (legendGroup, legendConfig, largestAbs) => {
  // Define the gradient
  let legendGradientDef = legendGroup.append('defs')
    .append('linearGradient')
    .attr('x1', 0)
    .attr('y1', 0)
    .attr('x2', 1)
    .attr('y2', 0)
    // TODO: use state to track the number of legend-gradient and make each of
    // them unique
    .attr('id', 'legend-gradient');

  legendGradientDef.append('stop')
    .attr('stop-color', legendConfig.startColor)
    .attr('offset', 0);

  legendGradientDef.append('stop')
    .attr('stop-color', '#ffffff')
    .attr('offset', 0.5);

  legendGradientDef.append('stop')
    .attr('stop-color', legendConfig.endColor)
    .attr('offset', 1);

  legendGroup.append('rect')
    .attr('x', 0)
    .attr('y', 0)
    .attr('width', legendConfig.width)
    .attr('height', legendConfig.height)
    .style('fill', 'url(#legend-gradient)')
    .style('stroke', 'hsla(0, 0%, 0%, 0.5)')
    .style('stroke-width', 0.3);

  // Draw the legend axis
  let legendScale = d3.scaleLinear()
    .domain([-largestAbs, largestAbs])
    .range([0, legendConfig.width]);

  let axisGroup = legendGroup.append('g')
    .attr('transform', `translate(${0}, ${legendConfig.height})`)
    .call(d3.axisBottom(legendScale).ticks(5));
  
  axisGroup.attr('font-family', config.defaultFont)
    .style('stroke-width', 0.5);

  legendGroup.append('text')
    .attr('class', 'legend-title')
    .attr('x', -10)
    .attr('y', 0)
    .style('dominant-baseline', 'hanging')
    .style('text-anchor', 'end')
    .style('font-weight', 300)
    .text('Score');
};

export const fadeRemove = (g, time = 500, ease = d3.easeCubicInOut) => {
  g.transition()
    .duration(time)
    .ease(ease)
    .style('opacity', 0)
    .on('end', (d, i, g) => {
      d3.select(g[i]).remove();
    });
};

export const startLoading = (loadingBar) => {
  d3.select(loadingBar).classed('animated', true);
};

export const endLoading = (loadingBar) => {
  d3.select(loadingBar).classed('animated', false);
};

/**
 * Draw a legend for the line colors
 * @param {object} svgSelect svgSelect for the whole SVG
 * @param {number} width SVG width (inside viewbox)
 * @param {object} svgPadding SVG paddings
 */
export const drawLineLegend = (svgSelect, width, svgPadding) => {
  let legendGroup = svgSelect.append('g')
    .attr('class', 'legend-group')
    .attr('transform', `translate(${width - svgPadding.right - 266}, 6)`);

  const legendRectWidth = 22;
  const legendRectHeight = 4;

  const legendData = [
    {
      name: 'original',
      class: 'original',
      title: 'Original bins',
      x: 0,
      rectColor: '#D1D1D1',
    },
    {
      name: 'last',
      class: 'last',
      title: 'Bins from last edit',
      x: 76,
      rectColor: '#FFDFB3',
    },
    {
      name: 'current',
      class: 'current',
      title: 'Current bins',
      x: 130,
      rectColor: '#263B73',
    },
    {
      name: 'editing',
      class: 'editing',
      title: 'Currently editing bins',
      x: 204,
      rectColor: '#D67D00',
    },
  ];

  let items = legendGroup.selectAll('g.legend-item')
    .data(legendData)
    .join('g')
    .style('cursor', 'default')
    .attr('transform', d => `translate(${d.x}, 0)`);

  items.append('title')
    .text(d => d.title);

  items.append('rect')
    .attr('y', 6)
    .attr('width', legendRectWidth)
    .attr('height', legendRectHeight)
    .attr('rx', 1)
    .attr('class', d => d.class)
    .style('fill', d => d.rectColor);

  items.append('text')
    .attr('class', 'line-legend-title')
    .attr('y', 2)
    .attr('x', legendRectWidth + 4)
    .text(d => d.name);
};

/**
 * Draw a legend for the bar colors
 * @param {object} svgSelect svgSelect for the whole SVG
 * @param {number} width SVG width (inside viewbox)
 * @param {object} svgPadding SVG paddings
 */
export const drawBarLegend = (svgSelect, width, svgPadding) => {
  let legendGroup = svgSelect.append('g')
    .attr('class', 'legend-group')
    .attr('transform', `translate(${width - svgPadding.right - 225}, 6)`);

  const legendRectWidth = 12;
  const legendRectHeight = 12;

  const legendData = [
    {
      name: 'original',
      class: 'original',
      title: 'Original bins',
      x: 0,
      rectColor: '#D1D1D1',
    },
    {
      name: 'last',
      class: 'last',
      title: 'Bins from last edit',
      x: 65,
      rectColor: '#FFDFB3',
    },
    {
      name: 'current',
      class: 'current',
      title: 'Current bins',
      x: 110,
      rectColor: '#5582EC',
    },
    {
      name: 'editing',
      class: 'editing',
      title: 'Currently editing bins',
      x: 175,
      rectColor: '#FFAA33',
    },
  ];

  let items = legendGroup.selectAll('g.legend-item')
    .data(legendData)
    .join('g')
    .style('cursor', 'default')
    .attr('transform', d => `translate(${d.x}, 0)`);

  items.append('title')
    .text(d => d.title);

  items.append('rect')
    .attr('y', 1)
    .attr('width', legendRectWidth)
    .attr('height', legendRectHeight)
    .attr('rx', 1)
    .attr('class', d => d.class)
    .style('fill', d => d.rectColor);

  items.append('text')
    .attr('class', 'line-legend-title')
    .attr('y', 2)
    .attr('x', legendRectWidth + 4)
    .text(d => d.name);
};