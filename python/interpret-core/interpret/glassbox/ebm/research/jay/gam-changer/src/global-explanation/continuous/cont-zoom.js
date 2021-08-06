import * as d3 from 'd3';
import { moveMenubar } from './cont-bbox';

export const rExtent = [2, 16];
export const zoomScaleExtent = [0.95, 30];

export const zoomStart = (state, multiMenu) => {
  if (state.selectedInfo.hasSelected) {
    d3.select(multiMenu)
      .classed('hidden', true);
  }
};

export const zoomEnd = (state, multiMenu) => {
  if (state.selectedInfo.hasSelected) {
    d3.select(multiMenu)
      .classed('hidden', false);
  }
};

/**
 * Update the view with zoom transformation
 * @param event Zoom event
 * @param xScale Scale for the x-axis
 * @param yScale Scale for the y-axis
 */
export const zoomed = (event, state, xScale, yScale, svg,
  linePathWidth, nodeStrokeWidth, yAxisWidth, lineChartWidth, lineChartHeight,
  multiMenu, component
) => {

  let svgSelect = d3.select(svg);
  let transform = event.transform;

  // Transform the axises
  let zXScale = transform.rescaleX(xScale);
  let zYScale = transform.rescaleY(yScale);

  state.curXScale = zXScale;
  state.curYScale = zYScale;
  state.curTransform = transform;

  svgSelect.select('g.x-axis')
    .call(d3.axisBottom(zXScale));

  svgSelect.select('g.y-axis')
    .call(d3.axisLeft(zYScale));

  // Transform the lines
  let lineGroup = svgSelect.selectAll('g.line-chart-line-group')
    .attr('transform', transform);

  // Rescale the stroke width a little bit
  lineGroup.style('stroke-width', linePathWidth / transform.k);

  // Transform the confidence rectangles
  svgSelect.select('g.line-chart-confidence-group')
    .attr('transform', transform);

  // Transform the nodes
  let nodeGroup = svgSelect.select('g.line-chart-node-group');

  if (transform.k <= 1 && nodeGroup.style('visibility') === 'visible') {
    nodeGroup.transition()
      .duration(300)
      .style('opacity', 0)
      .on('end', (d, i, g) => {
        d3.select(g[i])
          .style('visibility', 'hidden');
      });
  }

  if (transform.k > 1 && nodeGroup.style('visibility') === 'hidden') {
    nodeGroup.style('opacity', 0);
    nodeGroup.style('visibility', 'visible')
      .transition()
      .duration(500)
      .style('opacity', 1);
  }

  svgSelect.select('g.line-chart-node-group')
    .attr('transform', transform)
    .selectAll('circle.node')
    .attr('r', rScale(transform.k))
    .style('stroke-width', nodeStrokeWidth / transform.k);

  // Transform the density rectangles
  // Here we want to translate and scale the x axis, and keep y axis consistent
  svgSelect.select('g.hist-chart-content-group')
    .attr('transform', `translate(${yAxisWidth + transform.x},
        ${lineChartHeight})scale(${transform.k}, 1)`);

  // Transform the selection bbox if applicable
  if (state.selectedInfo.hasSelected) {
    // Here we don't use transform, because we want to keep the gap between
    // the nodes and bounding box border constant across all scales

    // We want to compute the world coordinate here
    // Need to transfer back the scale factor from the node radius
    let curPadding = rScale(state.curTransform.k) + state.bboxPadding * state.curTransform.k;

    svgSelect.select('g.line-chart-content-group')
      .selectAll('rect.select-bbox')
      .attr('x', d => state.curXScale(d.x1) - curPadding)
      .attr('y', d => state.curYScale(d.y1) - curPadding)
      .attr('width', d => {
        if (state.selectedInfo.nodeData.length === 1) {
          return state.curXScale(d.x2) - state.curXScale(d.x1) + 2 * curPadding;
        } else {
          return state.curXScale(d.x2) - state.curXScale(d.x1) + curPadding;
        }
      })
      .attr('height', d => state.curYScale(d.y2) - state.curYScale(d.y1) + 2 * curPadding);

    // Also transform the menu bar
    d3.select(multiMenu)
      .call(moveMenubar, svg, component);
  }

  // Draw/update the grid
  svgSelect.select('g.line-chart-grid-group')
    .call(drawGrid, zXScale, zYScale, lineChartWidth, lineChartHeight);

};

/**
 * Use linear interpolation to scale the node radius during zooming
 * It is actually kind of tricky, there should be better functions
 * (1) In overview, we want the radius to be small to avoid overdrawing;
 * (2) When zooming in, we want the radius to increase (slowly)
 * (3) Need to counter the zoom's scaling effect
 * @param k Scale factor
 */
export const rScale = (k) => {
  let alpha = (k - zoomScaleExtent[0]) / (zoomScaleExtent[1] - zoomScaleExtent[0]);
  alpha = d3.easeLinear(alpha);
  let target = alpha * (rExtent[1] - rExtent[0]) + rExtent[0];
  return target / k;
};

const drawGrid = (g, xScale, yScale, lineChartWidth, lineChartHeight) => {
  g.style('stroke', 'black')
    .style('stroke-opacity', 0.08);

  // Add vertical lines based on the xScale ticks
  g.call(g => g.selectAll('line.grid-line-x')
    .data(xScale.ticks(), d => d)
    .join(
      enter => enter.append('line')
        .attr('class', 'grid-line-x')
        .attr('y2', lineChartHeight),
      update => update,
      exit => exit.remove()
    )
    .attr('x1', d => 0.5 + xScale(d))
    .attr('x2', d => 0.5 + xScale(d))
  );

  // Add horizontal lines based on the yScale ticks
  return g.call(g => g.selectAll('line.grid-line-y')
    .data(yScale.ticks(), d => d)
    .join(
      enter => enter.append('line')
        .attr('class', 'grid-line-y')
        .classed('grid-line-y-0', d => d === 0)
        .attr('x2', lineChartWidth),
      update => update,
      exit => exit.remove()
    )
    .attr('y1', d => yScale(d))
    .attr('y2', d => yScale(d))
  );
};