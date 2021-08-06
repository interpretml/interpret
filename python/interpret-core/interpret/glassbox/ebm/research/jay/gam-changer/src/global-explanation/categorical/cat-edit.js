import * as d3 from 'd3';
import { rScale } from './cat-zoom';

export const drawBufferGraph = (state, svg, animated, duration, callback = () => { }) => {
  const svgSelect = d3.select(svg);

  let trans = d3.transition('buffer')
    .duration(duration)
    .ease(d3.easeCubicInOut)
    .on('end', () => {
      callback();
    });

  let nodes = svgSelect.select('g.scatter-plot-dot-group')
    .selectAll('.additive-dot');

  let bars = svgSelect.select('g.scatter-plot-bar-group.real')
    .selectAll('.additive-bar');

  // Only update, no enter or exit
  if (animated) {
    nodes.data(Object.values(state.pointDataBuffer), d => d.id)
      .transition(trans)
      .attr('cx', d => state.oriXScale(d.x))
      .attr('cy', d => state.oriYScale(d.y));

    bars.data(Object.values(state.pointDataBuffer), d => d.id)
      .transition(trans)
      .attr('y', d => d.y > 0 ? state.oriYScale(d.y) : state.oriYScale(0))
      .attr('height', d => Math.abs(state.oriYScale(d.y) - state.oriYScale(0)));
  } else {
    nodes.data(Object.values(state.pointDataBuffer), d => d.id)
      .attr('cx', d => state.oriXScale(d.x))
      .attr('cy', d => state.oriYScale(d.y));

    bars.data(Object.values(state.pointDataBuffer), d => d.id)
      .attr('y', d => d.y > 0 ? state.oriYScale(d.y) : state.oriYScale(0))
      .attr('height', d => Math.abs(state.oriYScale(d.y) - state.oriYScale(0)));
  }

  // Move the selected bbox
  let curPadding = rScale(state.curTransform.k) + state.bboxPadding * state.curTransform.k;

  if (animated) {
    svgSelect.select('g.scatter-plot-content-group g.select-bbox-group')
      .selectAll('rect.select-bbox')
      .datum(state.selectedInfo.boundingBox[0])
      .transition(trans)
      .attr('y', d => state.curYScale(d.y1) - curPadding)
      .attr('height', d => state.curYScale(d.y2) - state.curYScale(d.y1) + 2 * curPadding);
  } else {
    svgSelect.select('g.scatter-plot-content-group g.select-bbox-group')
      .selectAll('rect.select-bbox')
      .datum(state.selectedInfo.boundingBox[0])
      .attr('y', d => state.curYScale(d.y1) - curPadding)
      .attr('height', d => state.curYScale(d.y2) - state.curYScale(d.y1) + 2 * curPadding);
  }
};

export const drawLastEdit = (state, svg, barWidth) => {
  if (state.pointDataLastEdit === undefined) {
    return;
  }

  const svgSelect = d3.select(svg);

  let trans = d3.transition('lastEdit')
    .duration(400)
    .ease(d3.easeCubicInOut);

  let bars = svgSelect.select('g.scatter-plot-bar-group.last-edit-back')
    .selectAll('.additive-bar');

  let lines = svgSelect.select('g.scatter-plot-bar-group.last-edit-front')
    .selectAll('.additive-line');

  bars.data(Object.values(state.pointDataLastEdit), d => d.id)
    .join(
      enter => enter.append('rect')
        .attr('class', 'additive-bar')
        .attr('x', d => state.oriXScale(d.x) - barWidth / 2)
        .attr('y', state.oriYScale(0))
        .attr('width', barWidth)
        .attr('height', 0)
        .call(enter => enter.transition(trans)
          .attr('y', d => d.y > 0 ? state.oriYScale(d.y) : state.oriYScale(0))
          .attr('height', d => Math.abs(state.oriYScale(d.y) - state.oriYScale(0)))
        ),
      update => update.call(update => update.transition(trans)
        .attr('y', d => d.y > 0 ? state.oriYScale(d.y) : state.oriYScale(0))
        .attr('height', d => Math.abs(state.oriYScale(d.y) - state.oriYScale(0)))
      )
    );

  lines.data(Object.values(state.pointDataLastEdit), d => d.id)
    .join(
      enter => enter.append('path')
        .attr('class', 'additive-line')
        .attr('d', d => `M ${state.oriXScale(d.x) - barWidth / 2}, ${state.oriYScale(0)} l ${barWidth}, 0`)
        .call(enter => enter.transition(trans)
          .attr('d', d => `M ${state.oriXScale(d.x) - barWidth / 2}, ${state.oriYScale(d.y)} l ${barWidth}, 0`)
        ),
      update => update.call(update => update.transition(trans)
        .attr('d', d => `M ${state.oriXScale(d.x) - barWidth / 2}, ${state.oriYScale(d.y)} l ${barWidth}, 0`)
      )
    );

};

export const grayOutConfidenceLine = (state, svg) => {
  let editingIDs = new Set();
  state.selectedInfo.nodeData.forEach(d => editingIDs.add(d.id));

  d3.select(svg)
    .selectAll('.scatter-plot-confidence-group .dot-confidence')
    .filter(d => editingIDs.has(d.id))
    .classed('edited', true);
};

export const redrawOriginal = (state, svg, bounce=true, animationEndFunc=undefined) => {
  const svgSelect = d3.select(svg);

  let trans = d3.transition('restore')
    .duration(500)
    .ease(d3.easeElasticOut
      .period(0.35)
    );

  let transNoBounce = d3.transition('restoreNo')
    .duration(500)
    .ease(d3.easeLinear);

  if (!bounce) {
    trans = transNoBounce;
  }

  if (animationEndFunc !== undefined) {
    trans.on('end', animationEndFunc);
  }

  // Step 1: update the bbox info
  state.selectedInfo.updateNodeData(state.pointData);
  state.selectedInfo.computeBBox(state.pointData);

  // Step 2: redraw the nodes and bars with original data
  let nodes = svgSelect.select('g.scatter-plot-dot-group')
    .selectAll('.additive-dot');

  let bars = svgSelect.select('g.scatter-plot-bar-group.real')
    .selectAll('.additive-bar');

  // Only update, no enter or exit
  nodes.data(Object.values(state.pointData), d => d.id)
    .transition(trans)
    .attr('cx', d => state.oriXScale(d.x))
    .attr('cy', d => state.oriYScale(d.y));

  bars.data(Object.values(state.pointData), d => d.id)
    .transition(trans)
    .attr('y', d => d.y > 0 ? state.oriYScale(d.y) : state.oriYScale(0))
    .attr('height', d => Math.abs(state.oriYScale(d.y) - state.oriYScale(0)));

  // Step 4: move the selected bbox to their original place
  let curPadding = rScale(state.curTransform.k) + state.bboxPadding * state.curTransform.k;

  svgSelect.select('g.scatter-plot-content-group g.select-bbox-group')
    .selectAll('rect.select-bbox')
    .datum(state.selectedInfo.boundingBox[0])
    .transition(trans)
    .attr('y', d => state.curYScale(d.y1) - curPadding)
    .attr('height', d => state.curYScale(d.y2) - state.curYScale(d.y1) + 2 * curPadding);
};