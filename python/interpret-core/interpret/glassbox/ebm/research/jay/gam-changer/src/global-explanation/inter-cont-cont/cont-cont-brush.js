import * as d3 from 'd3';
import { SelectedInfo } from './cont-cont-class';
import { moveMenubar } from '../continuous/cont-bbox';
import { rExtent } from './cont-cont-zoom';
import { state } from './cont-cont-state';
// import { redrawOriginal, drawLastEdit } from './cont-edit';

// Need a timer to avoid the brush event call after brush.move()
let idleTimeout = null;
const idleDelay = 300;

/**
 * Reset the idleTimeout timer
 */
const idled = () => {
  idleTimeout = null;
};

/**
 * Stop animating all flowing lines
 */
const stopAnimateLine = (svg) => {
  d3.select(svg)
    .select('g.line-chart-line-group')
    .selectAll('path.additive-line-segment.flow-line')
    .interrupt()
    .attr('stroke-dasharray', '0 0')
    .classed('flow-line', false);
};

export const brushDuring = (event, svg, multiMenu) => {
  // Get the selection boundary
  let selection = event.selection;
  let svgSelect = d3.select(svg);

  if (selection === null) {
    if (idleTimeout === null) {
      return idleTimeout = setTimeout(idled, idleDelay);
    }
  } else {
    // Compute the selected data region
    // X is ordinal, we just use the view coordinate instead of data
    let xRange = [selection[0][0], selection[1][0]];
    let yRange = [state.curYScale.invert(selection[1][1]), state.curYScale.invert(selection[0][1])];

    // Clean up the previous flowing lines
    state.selectedInfo = new SelectedInfo();

    // Remove the selection bbox
    svgSelect.selectAll('g.scatter-plot-content-group g.select-bbox-group').remove();

    d3.select(multiMenu)
      .classed('hidden', true);

    // Highlight the selected dots
    svgSelect.select('g.scatter-plot-dot-group')
      .selectAll('circle.additive-dot')
      .classed('selected', d => (state.curXScale(d.x) >= xRange[0] &&
        state.curXScale(d.x) <= xRange[1] && d.y >= yRange[0] && d.y <= yRange[1]));

    // Highlight the bars associated with the selected dots
    svgSelect.select('g.scatter-plot-bar-group')
      .selectAll('rect.additive-bar')
      .classed('selected', d => (state.curXScale(d.x) >= xRange[0] &&
        state.curXScale(d.x) <= xRange[1] && d.y >= yRange[0] && d.y <= yRange[1]));

    svgSelect.select('g.scatter-plot-confidence-group')
      .selectAll('path.dot-confidence')
      .classed('selected', d => (state.curXScale(d.x) >= xRange[0] &&
        state.curXScale(d.x) <= xRange[1] && d.y >= yRange[0] && d.y <= yRange[1]));
  }
};

export const brushEndSelect = (event, svg, multiMenu, brush, component,
  resetContextMenu
) => {
  // Get the selection boundary
  let selection = event.selection;
  let svgSelect = d3.select(svg);

  if (selection === null) {
    if (idleTimeout === null) {
      // Clean up the previous flowing lines
      stopAnimateLine();
      state.selectedInfo = new SelectedInfo();

      svgSelect.select('g.line-chart-content-group g.brush rect.overlay')
        .attr('cursor', null);

      d3.select(multiMenu)
        .classed('hidden', true);

      resetContextMenu();

      // Do not save the user's change (same as clicking the cancel button)
      // Redraw the graph with original data
      // redrawOriginal(svg);

      // Redraw the last edit if possible
      if (state.additiveDataLastLastEdit !== undefined) {
        state.additiveDataLastEdit = JSON.parse(JSON.stringify(state.additiveDataLastLastEdit));
        // drawLastEdit(svg);
        // Prepare for next redrawing after recovering the last last edit graph
        state.additiveDataLastEdit = JSON.parse(JSON.stringify(state.additiveData));
      }

      // Remove the selection bbox
      svgSelect.selectAll('g.scatter-plot-content-group g.select-bbox-group').remove();

      return idleTimeout = setTimeout(idled, idleDelay);
    }
  } else {

    // Compute the selected data region
    // X is ordinal, we just use the view coordinate instead of data
    let xRange = [selection[0][0], selection[1][0]];
    let yRange = [state.curYScale.invert(selection[1][1]), state.curYScale.invert(selection[0][1])];

    // Highlight the selected dots
    svgSelect.select('g.scatter-plot-dot-group')
      .selectAll('circle.additive-dot')
      .classed('selected', d => {
        if (state.curXScale(d.x) >= xRange[0] && state.curXScale(d.x) <= xRange[1] && d.y >= yRange[0] && d.y <= yRange[1]) {
          state.selectedInfo.nodeData.push({ x: d.x, y: d.y, id: d.id });
          return true;
        } else {
          return false;
        }
      });

    // Compute the bounding box
    state.selectedInfo.computeBBox();

    let curPadding = (rExtent[0] + state.bboxPadding) * state.curTransform.k;

    let bbox = svgSelect.select('g.scatter-plot-content-group')
      .append('g')
      .attr('class', 'select-bbox-group')
      .selectAll('rect.select-bbox')
      .data(state.selectedInfo.boundingBox)
      .join('rect')
      .attr('class', 'select-bbox original-bbox')
      .attr('x', d => state.curXScale(d.x1) - curPadding)
      .attr('y', d => state.curYScale(d.y1) - curPadding)
      .attr('width', d => state.curXScale(d.x2) - state.curXScale(d.x1) + 2 * curPadding)
      .attr('height', d => state.curYScale(d.y2) - state.curYScale(d.y1) + 2 * curPadding)
      .style('stroke-width', 1)
      .style('stroke', 'hsl(230, 100%, 10%)')
      .style('stroke-dasharray', '5 3');

    bbox.clone(true)
      .classed('original-bbox', false)
      .style('stroke', 'white')
      .style('stroke-dasharray', null)
      .style('stroke-width', 1 * 3)
      .lower();

    state.selectedInfo.hasSelected = svgSelect.selectAll('g.scatter-plot-dot-group circle.additive-dot.selected').size() > 0;

    if (state.selectedInfo.hasSelected) {
      // Show the context menu near the selected region
      d3.select(multiMenu)
        .call(moveMenubar, svg, component)
        .classed('hidden', false);
    }

    // Remove the brush box
    svgSelect.select('g.scatter-plot-content-group g.brush')
      .call(brush.move, null)
      .select('rect.overlay')
      .attr('cursor', null);
  }
};
