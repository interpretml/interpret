import * as d3 from 'd3';
import { l1Distance } from '../utils/utils';


/**
 * Draw the legend and title for the first time.
 * @param {object} component Sidebar component
 * @param {number} width Width of sidebar
 * @param {object} svgCatPadding Paddings for the current svg
 */
export const initLegend = (component, width, svgCatPadding) => {
  let legendSVG = d3.select(component)
    .select('svg#legend')
    .attr('width', width)
    .attr('height', 40);

  let leftGroup = legendSVG.append('g')
    .attr('class', 'left')
    .attr('transform', `translate(${svgCatPadding.left}, 10)`);

  let rightGroup = legendSVG.append('g')
    .attr('class', 'right')
    .attr('transform', `translate(${svgCatPadding.left + 170}, 6)`);


  leftGroup.append('text')
    .attr('x', 0)
    .attr('y', 0)
    .style('dominant-baseline', 'hanging')
    .style('font-size', '0.9em')
    .style('font-weight', 600)
    .text('Frequency Distributions')
    .append('tspan')
    .style('font-size', '0.9em')
    .style('font-weight', 400)
    .style('fill', 'gray')
    .style('dominant-baseline', 'hanging')
    .attr('x', 0)
    .attr('y', 0)
    .attr('dy', '1.1em')
    .text('Sorted by correlation ↓');

  let labelWidth = 48;
  rightGroup.append('rect')
    .attr('width', labelWidth)
    .attr('height', 16)
    .attr('rx', 3)
    .style('fill', '#B5CEE3');

  rightGroup.append('text')
    .attr('class', 'legend-title')
    .attr('y', 2)
    .attr('x', labelWidth / 2)
    .text('all');

  rightGroup.append('rect')
    .attr('y', 20)
    .attr('width', labelWidth)
    .attr('height', 16)
    .attr('rx', 3)
    .style('fill', '#FFD499');

  rightGroup.append('text')
    .attr('class', 'legend-title')
    .attr('y', 22)
    .attr('x', labelWidth / 2)
    .text('select');
};


/**
 * Draw the continuous feature plot for the first time.
 * @param {object} component Side bar component
 * @param {object} f Current feature
 * @param {object} svgContPadding Padding info
 * @param {number} totalSampleNum Number of selected test samples
 * @param {number} width Sidebar width
 * @param {number} svgHeight SVG Height
 * @param {number} titleHeight Feature name text height
 */
export const initContFeature = (component, f, svgContPadding, totalSampleNum,
  width, svgHeight, titleHeight) => {

  let svg = d3.select(component)
    .select(`.feature-${f.id}`)
    .select('svg');

  let lowContent = svg.append('g')
    .attr('class', 'low-content')
    .attr('transform', `translate(${svgContPadding.left}, ${svgContPadding.top})`);

  let midContent = svg.append('g')
    .attr('class', 'mid-content')
    .attr('transform', `translate(${svgContPadding.left}, ${svgContPadding.top})`);

  let topContent = svg.append('g')
    .attr('class', 'top-content')
    .attr('transform', `translate(${svgContPadding.left}, ${svgContPadding.top})`);

  // Add the feature title
  topContent.append('text')
    .attr('class', 'feature-title')
    .attr('x', 0)
    .text(f.name);

  // Compute the frequency of test samples
  let curDensity = f.histCount.map((d, i) => [f.histEdge[i], d / totalSampleNum]);

  // Create the axis scales
  let expectedBarWidth = (width - svgContPadding.left - svgContPadding.right) / f.histEdge.length;
  let xScale = d3.scaleLinear()
    .domain(d3.extent(f.histEdge))
    .range([0, width - svgContPadding.left - svgContPadding.right - expectedBarWidth]);

  let barWidth = xScale(f.histEdge[1]) - xScale(f.histEdge[0]);

  let yScale = d3.scaleLinear()
    .domain([0, d3.max(curDensity, d => d[1])])
    .range([svgHeight - svgContPadding.bottom, svgContPadding.top + titleHeight]);

  lowContent.selectAll('rect.global-bar')
    .data(curDensity)
    .join('rect')
    .attr('class', 'global-bar')
    .attr('x', d => xScale(d[0]))
    .attr('y', d => yScale(d[1]))
    .attr('width', barWidth)
    .attr('height', d => svgHeight - svgContPadding.bottom - yScale(d[1]));

  // Draw overlay layer
  let curDensitySelected = new Array(f.histCount.length).fill(0);

  const yMax = d3.max(curDensitySelected) === 0 ? 1 : d3.max(curDensitySelected);

  let yScaleBar = d3.scaleLinear()
    .domain([0, yMax])
    .range([svgHeight - svgContPadding.bottom, svgContPadding.top + titleHeight]);

  midContent.selectAll('rect.selected-bar')
    .data(curDensitySelected)
    .join('rect')
    .attr('class', 'selected-bar')
    .attr('x', (d, i) => xScale(f.histEdge[i]))
    .attr('y', d => yScaleBar(d))
    .attr('width', barWidth)
    .attr('height', d => svgHeight - svgContPadding.bottom - yScaleBar(d));
};


/**
 * Draw the categorical feature plot for the first time.
 * @param {object} component Side bar component
 * @param {object} f Current feature
 * @param {object} svgCatPadding Padding info
 * @param {number} catBarWidth Width of bars in the histogram plot
 * @param {number} totalSampleNum Number of selected test samples
 * @param {number} width Sidebar width
 * @param {number} svgHeight SVG Height
 * @param {number} titleHeight Feature name text height
 */
export const initCatFeature = (component, f, svgCatPadding, totalSampleNum,
  width, catBarWidth, svgHeight, titleHeight) => {
  let svg = d3.select(component)
    .select(`.feature-${f.id}`)
    .select('svg');

  let lowContent = svg.append('g')
    .attr('class', 'low-content')
    .attr('transform', `translate(${svgCatPadding.left}, ${svgCatPadding.top})`);

  let midContent = svg.append('g')
    .attr('class', 'mid-content')
    .attr('transform', `translate(${svgCatPadding.left}, ${svgCatPadding.top})`);

  let topContent = svg.append('g')
    .attr('class', 'top-content')
    .attr('transform', `translate(${svgCatPadding.left}, ${svgCatPadding.top})`);

  // Sort the bins from high count to low count, and save the sorting order
  // (needed to update selected bins)
  let curData = f.histEdge.map((d, i) => ({
    edge: f.histEdge[i],
    count: f.histCount[i],
    density: f.histCount[i] / totalSampleNum,
    // Initialize selected bars with 0 density
    selectedCount: 0,
    selectedDensity: 0
  }));

  curData.sort((a, b) => b.count - a.count);

  // Create the axis scales
  // histEdge, histCount, histDensity
  let xScale = d3.scaleBand()
    .domain(curData.map(d => d.edge))
    .paddingInner(0.1)
    .range([0, width - svgCatPadding.left - svgCatPadding.right]);

  let yScale = d3.scaleLinear()
    .domain([0, d3.max(curData, d => d.density)])
    .range([svgHeight - svgCatPadding.bottom, svgCatPadding.top + titleHeight]);

  // Add the feature title
  topContent.append('text')
    .attr('class', 'feature-title')
    .attr('x', xScale(curData[0].edge))
    .text(f.name);

  // Draw a short pink rectangle as baseline (to signal missing value / small value)
  lowContent.selectAll('rect.base-bar')
    .data(curData)
    .join('rect')
    .attr('class', 'base-bar')
    .attr('x', d => xScale(d.edge))
    .attr('y', svgHeight - svgCatPadding.bottom - 1)
    .attr('width', xScale.bandwidth())
    .attr('height', 1)
    .style('fill', 'hsl(0, 0%, 95%)');

  // Draw the global histogram
  lowContent.selectAll('rect.global-bar')
    .data(curData)
    .join('rect')
    .attr('class', 'global-bar')
    .attr('x', d => xScale(d.edge))
    .attr('y', d => yScale(d.density))
    .attr('width', xScale.bandwidth())
    .attr('height', d => svgHeight - svgCatPadding.bottom - yScale(d.density));

  // Draw overlay layer
  let yScaleSelected = d3.scaleLinear()
    .domain([0, 1])
    .range([svgHeight - svgCatPadding.bottom, svgCatPadding.top + titleHeight]);

  midContent.selectAll('rect.selected-bar')
    .data(curData)
    .join('rect')
    .attr('class', 'selected-bar')
    .attr('x', d => xScale(d.edge))
    .attr('y', d => yScaleSelected(d.selectedDensity))
    .attr('width', xScale.bandwidth())
    .attr('height', d => svgHeight - svgCatPadding.bottom - yScaleSelected(d.selectedDensity));
};


/**
 * Update the continuous feature plot with distributions of the selected samples.
 * @param {object} component Side bar component
 * @param {object} f Current feature
 * @param {number} selectedSampleCount Number of selected test samples
 * @param {number} totalSampleNum Number of total test samples
 * @param {number} svgHeight SVG Height
 * @param {object} svgContPadding Padding info
 * @param {number} titleHeight Feature name text height
 */
export const updateContFeature = (component, f, selectedSampleCount, totalSampleCount,
  svgHeight, svgContPadding, titleHeight) => {
  let svg = d3.select(component)
    .select(`svg#cont-feature-svg-${f.id}`);

  let curDensitySelected = f.histSelectedCount.map(c => selectedSampleCount === 0 ? 0 : c / selectedSampleCount);
  let globalDensity = f.histCount.map(c => c / totalSampleCount);

  // Compute teh distance between subset density vs. global density
  f.distanceScore = l1Distance(globalDensity, curDensitySelected);

  const yMax = d3.max(curDensitySelected) === 0 ? 1 : d3.max(curDensitySelected);
  const needToResort = d3.max(curDensitySelected) === 0 ? false : true;

  let yScaleBar = d3.scaleLinear()
    .domain([0, yMax])
    .range([svgHeight - svgContPadding.bottom, svgContPadding.top + titleHeight]);

  svg.select('g.mid-content')
    .selectAll('rect.selected-bar')
    .data(curDensitySelected)
    .join('rect')
    .transition('bar')
    .duration(500)
    .attr('y', d => yScaleBar(d))
    .attr('height', d => svgHeight - svgContPadding.bottom - yScaleBar(d));

  return needToResort;
};


/**
 * Update the categorical feature plot with distributions of the selected samples.
 * @param {object} component Side bar component
 * @param {object} f Current feature
 * @param {number} selectedSampleCount Number of selected test samples
 * @param {number} totalSampleNum Number of total test samples
 * @param {number} svgHeight SVG Height
 * @param {object} svgCatPadding Padding info
 * @param {number} titleHeight Feature name text height
 */
export const updateCatFeature = (component, f, selectedSampleCount, totalSampleCount,
  svgHeight, svgCatPadding, titleHeight) => {
  let svg = d3.select(component)
    .select(`#cat-feature-svg-${f.id}`);

  let curData = f.histEdge.map((d, i) => ({
    edge: f.histEdge[i],
    count: f.histCount[i],
    selectedCount: f.histSelectedCount[i],
    selectedDensity: selectedSampleCount === 0 ? 0 : f.histSelectedCount[i] / selectedSampleCount
  }));

  // Compute the distance score
  let selectedDensity = curData.map(d => d.selectedDensity);
  let globalDensity = curData.map(d => d.count / totalSampleCount);
  f.distanceScore = l1Distance(globalDensity, selectedDensity);

  curData.sort((a, b) => b.count - a.count);

  const yMax = d3.max(curData, d => d.selectedDensity) === 0 ? 1 : d3.max(curData, d => d.selectedDensity);
  const needToResort = d3.max(curData, d => d.selectedDensity) === 0 ? false : true;

  let yScaleSelected = d3.scaleLinear()
    .domain([0, yMax])
    .range([svgHeight - svgCatPadding.bottom, svgCatPadding.top + titleHeight]);

  svg.select('g.mid-content')
    .selectAll('rect.selected-bar')
    .data(curData, d => d.edge)
    .join('rect')
    .transition('cont-bar')
    .duration(500)
    .attr('y', d => yScaleSelected(d.selectedDensity))
    .attr('height', d => svgHeight - svgCatPadding.bottom - yScaleSelected(d.selectedDensity));

  return needToResort;
};
