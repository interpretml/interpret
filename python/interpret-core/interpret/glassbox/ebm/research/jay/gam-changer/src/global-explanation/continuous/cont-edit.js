import * as d3 from 'd3';
import { round } from '../../utils/utils';
import { rScale, rExtent } from './cont-zoom';
import { updateAdditiveDataBufferFromPointDataBuffer } from './cont-data';
import { SimpleLinearRegression } from '../../simple-linear-regression';

// TODO: Uniform this variable across all files (use config file)
const nodeStrokeWidth = 1;
let dragTimeout = null;

export const dragged = (e, state, svg, useTimeout, footerStore, updateEBM) => {

  const dataYChange = state.curYScale.invert(e.y) - state.curYScale.invert(e.y - e.dy);

  // Change the data based on the y-value changes, then redraw nodes (preferred method)
  state.selectedInfo.nodeData.forEach(d => {
    const id = d.id;
    const curPoint = state.pointDataBuffer[id];

    // Step 1.1: update point data
    state.pointDataBuffer[id].y += dataYChange;

    // Step 1.2: update path data

    // i's Left line
    const leftLineIndex = curPoint.leftLineIndex;
    if (leftLineIndex !== null) {
      state.additiveDataBuffer[leftLineIndex].y2 += dataYChange;
    }

    // i's Right line
    const rightLineIndex = curPoint.rightLineIndex;
    state.additiveDataBuffer[rightLineIndex].y1 += dataYChange;
    state.additiveDataBuffer[rightLineIndex].y2 += dataYChange;
    state.additiveDataBuffer[rightLineIndex].sy += dataYChange;

    // (i + 1)'s Left line
    if (curPoint.rightPointID !== null) {
      const nextLeftLineIndex = state.pointDataBuffer[curPoint.rightPointID].leftLineIndex;
      state.additiveDataBuffer[nextLeftLineIndex].y1 += dataYChange;
      state.additiveDataBuffer[nextLeftLineIndex].sy += dataYChange;
    }

  });

  // Step 1.3: update the bbox info
  state.selectedInfo.updateNodeData(state.pointDataBuffer);
  state.selectedInfo.computeBBox(state.pointDataBuffer);

  // Update the visualization with new data
  // Step 2: redraw the nodes that are changed
  drawBufferGraph(state, svg, false);

  // Update the sidebar info
  if (dragTimeout !== null) {
    clearTimeout(dragTimeout);
  }
  dragTimeout = setTimeout(() => {
    updateEBM('current');
  }, useTimeout ? 300 : 0);
  

  // Update the footer message
  footerStore.update(value => {
    value.baseline += dataYChange;
    value.state = `Scores <b>${value.baseline >= 0 ? 'increased' : 'decreased'}</b> by <b>${round(Math.abs(value.baseline), 2)}</b>`;
    return value;
  });
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

  // if (state.lastSelectedInfo !== null) {
  //   state.lastSelectedInfo.updateNodeData(state.pointData);
  //   state.lastSelectedInfo.computeBBox(state.pointData);
  // }

  // Step 2: redraw the nodes with original data
  let nodes = svgSelect.select('g.line-chart-node-group')
    .selectAll('circle.node');

  nodes.data(Object.values(state.pointData), d => d.id)
    .join(
      enter => enter.append('circle')
        .attr('class', 'node')
        .classed('selected', state.selectedInfo.hasSelected)
        .attr('id', d => `node-${d.id}`)
        .attr('cx', d => state.oriXScale(d.x))
        .attr('cy', d => state.oriYScale(d.y))
        .attr('r', rScale(state.curTransform.k))
        .style('opacity', 0)
        .style('stroke-width', nodeStrokeWidth / state.curTransform.k)
        .call(
          enter => enter.transition(transNoBounce)
            .style('opacity', 1)
        ),
      update => update.call(
        update => update.transition(trans)
          .attr('cx', d => state.oriXScale(d.x))
          .attr('cy', d => state.oriYScale(d.y))
      ),
      exit => exit.remove()
    );

  // Step 3: redraw the paths with original data
  let paths = svgSelect.select('g.line-chart-line-group.real')
    .selectAll('path.additive-line-segment');

  paths.data(state.additiveData, d => `${d.id}-${d.pos}`)
    .join(
      enter => enter.append('path')
        .attr('class', 'additive-line-segment')
        .classed('selected', state.selectedInfo.hasSelected)
        .attr('id', d => d.id)
        .attr('d', d => `M ${state.oriXScale(d.x1)}, ${state.oriYScale(d.y1)}
          L ${state.oriXScale(d.x2)} ${state.oriYScale(d.y2)}`)
        .style('opacity', 0.2)
        .call(
          enter => enter.transition(transNoBounce)
            .delay(300)
            .style('opacity', 1)
        ),
      update => update.call(
        update => update.transition(trans)
          .attr('d', d => `M ${state.oriXScale(d.x1)}, ${state.oriYScale(d.y1)}
              L ${state.oriXScale(d.x2)} ${state.oriYScale(d.y2)}`)
      ),
      exit => exit.call(
        exit => exit.style('opacity', 1)
          .transition('pre')
          .duration(300)
          .style('opacity', 0)
          .remove()
      )
    );

  // Step 4: move the selected bbox to their original place
  let curPadding = rScale(state.curTransform.k) + state.bboxPadding * state.curTransform.k;

  svgSelect.select('g.line-chart-content-group g.select-bbox-group')
    .selectAll('rect.select-bbox')
    .datum(state.selectedInfo.boundingBox[0])
    .transition(trans)
    .attr('y', d => state.curYScale(d.y1) - curPadding)
    .attr('height', d => state.curYScale(d.y2) - state.curYScale(d.y1) + 2 * curPadding);
};

export const redrawMonotone = (state, svg, isoYs, callBack) => {

  // Change the data based on the y-value changes, then redraw nodes (preferred method)
  state.selectedInfo.nodeData.forEach((d, j) => {
    const id = d.id;
    const curPoint = state.pointDataBuffer[id];

    // Step 1.1: update point data
    state.pointDataBuffer[id].y = isoYs[j];

    // Step 1.2: update path data

    // i's Left line
    const leftLineIndex = curPoint.leftLineIndex;
    if (leftLineIndex !== null) {
      state.additiveDataBuffer[leftLineIndex].y2 = isoYs[j];
    }

    // i's Right line
    const rightLineIndex = curPoint.rightLineIndex;
    state.additiveDataBuffer[rightLineIndex].y1 = isoYs[j];
    state.additiveDataBuffer[rightLineIndex].y2 = isoYs[j];
    state.additiveDataBuffer[rightLineIndex].sy = isoYs[j];

    // (i + 1)'s Left line
    if (curPoint.rightPointID !== null) {
      const nextLeftLineIndex = state.pointDataBuffer[curPoint.rightPointID].leftLineIndex;
      state.additiveDataBuffer[nextLeftLineIndex].y1 = isoYs[j];
      state.additiveDataBuffer[nextLeftLineIndex].sy = isoYs[j];
    }
  });

  // Step 1.3: update the bbox info
  state.selectedInfo.updateNodeData(state.pointDataBuffer);
  state.selectedInfo.computeBBox(state.pointDataBuffer);

  // Step 2 Update the visualization with new data
  drawBufferGraph(state, svg, true, 400, callBack);
};

export const regressionInterpolate = (state, svg, callBack) => {
  // Regression interpolation
  // Create x and y arrays
  let xs = [];
  let ys = [];
  let ws = [];

  // We use the points in between as interpolation steps
  let leftPoint = { x: Infinity, y: null, id: null };
  let rightPoint = { x: -Infinity, y: null, id: null };

  state.selectedInfo.nodeDataBuffer.forEach(d => {
    let curPoint = state.pointDataBuffer[d.id];
    if (d.x < leftPoint.x) {
      leftPoint = curPoint;
    }
    if (d.x > rightPoint.x) {
      rightPoint = curPoint;
    }
    xs.push(curPoint.x);
    ys.push(curPoint.y);
    ws.push(curPoint.count);
  });

  // Fit a linear regression and transform xs
  let model = new SimpleLinearRegression();
  model.fit(xs, ys, ws);
  let newYs = model.predict(xs);

  // Step 1: Update the point data
  state.selectedInfo.nodeDataBuffer.forEach((d, i) => {
    state.pointDataBuffer[d.id].y = newYs[i];
  });

  // Step 2: Recreate the path data using the updated point data  
  updateAdditiveDataBufferFromPointDataBuffer(state);

  // Step 3: update the bbox info
  state.selectedInfo.updateNodeData(state.pointDataBuffer);
  state.selectedInfo.computeBBox(state.pointDataBuffer);

  // Step 4: Update the graph using new data
  drawBufferGraph(state, svg, true, 800, callBack);
};

export const inplaceInterpolate = (state, svg, callBack) => {
  // Regional interpolation
  // We use the points in between as interpolation steps
  let leftPoint = { x: Infinity, y: null, id: null };
  let rightPoint = { x: -Infinity, y: null, id: null };

  state.selectedInfo.nodeDataBuffer.forEach(d => {
    if (d.x < leftPoint.x) {
      leftPoint = state.pointDataBuffer[d.id];
    }
    if (d.x > rightPoint.x) {
      rightPoint = state.pointDataBuffer[d.id];
    }
  });

  const xRange = rightPoint.x - leftPoint.x;
  const yRange = rightPoint.y - leftPoint.y;

  // Iterate the point from left to right using their pointers and modify
  // both the node data and path data

  // Step 1: Update the point data
  let curPoint = state.pointDataBuffer[leftPoint.rightPointID];

  while (curPoint.id !== rightPoint.id) {
    let alpha = (curPoint.x - leftPoint.x) / xRange;
    let newY = leftPoint.y + yRange * alpha;
    curPoint.y = newY;

    curPoint = state.pointDataBuffer[curPoint.rightPointID];
  }

  // Step 2: Recreate the path data using the updated point data  
  updateAdditiveDataBufferFromPointDataBuffer(state);

  // Step 3: update the bbox info
  state.selectedInfo.updateNodeData(state.pointDataBuffer);
  state.selectedInfo.computeBBox(state.pointDataBuffer);

  // Step 4: Update the graph using new data
  drawBufferGraph(state, svg, true, 800, callBack);
};

export const stepInterpolate = (state, svg, steps, callBack) => {
  // Step 1: Find the left and right point in the selected region
  let leftPoint = { x: Infinity, y: null, id: null };
  let rightPoint = { x: -Infinity, y: null, id: null };
  let totalCount = 0;

  state.selectedInfo.nodeDataBuffer.forEach(d => {
    if (d.x < leftPoint.x) {
      leftPoint = state.pointDataBuffer[d.id];
    }
    if (d.x > rightPoint.x) {
      rightPoint = state.pointDataBuffer[d.id];
    }
  });

  const xRange = rightPoint.x - leftPoint.x;
  const yRange = rightPoint.y - leftPoint.y;

  // Step 2: Remove all the points (if any) in between
  let curPoint = state.pointDataBuffer[leftPoint.rightPointID];

  while(curPoint.id !== rightPoint.id) {
    totalCount += curPoint.count;
    const nextID = curPoint.rightPointID;
    delete state.pointDataBuffer[curPoint.id];
    curPoint = state.pointDataBuffer[nextID];
  }

  // Step 3: Create interpolated middle points and add them to the buffer

  for (let s = 1; s < steps + 1; s++) {
    // Step 3.1: Compute interpolated values
    const alpha = s / (steps + 1);
    const curX = leftPoint.x + alpha * xRange;
    const curY = leftPoint.y + alpha * yRange;
    const curPoint = {
      x: curX,
      y: curY,
      count: totalCount / steps,
      id: `q${leftPoint.id}pxq${rightPoint.id}p-${s}`,
      ebmID: leftPoint.id + s,
      leftPointID: s == 1 ? leftPoint.id : `q${leftPoint.id}pxq${rightPoint.id}p-${s - 1}`,
      rightPointID: s == steps ? rightPoint.id : `q${leftPoint.id}pxq${rightPoint.id}p-${s + 1}`,
      leftLineIndex: null,
      rightLineIndex: null
    };

    // Step 3.2: Add it to the buffer array
    state.pointDataBuffer[`q${leftPoint.id}pxq${rightPoint.id}p-${s}`] = curPoint;
  }

  // Step 3.3: Link the left point and right point to these new points
  leftPoint.rightPointID = `q${leftPoint.id}pxq${rightPoint.id}p-${1}`;
  rightPoint.leftPointID = `q${leftPoint.id}pxq${rightPoint.id}p-${steps}`;

  // Step 3.4: Update the ebmID of all points in the buffer
  // The first point will always have ID 0
  let curBufferPoint = state.pointDataBuffer[0];
  let curEbmID = 0;
  while(curBufferPoint.rightPointID !== null) {
    state.pointDataBuffer[curBufferPoint.id].ebmID = curEbmID++;
    curBufferPoint = state.pointDataBuffer[curBufferPoint.rightPointID];
  }
  state.pointDataBuffer[curBufferPoint.id].ebmID = curEbmID++;

  // Step 4: Recreate the path data using the updated point data  
  updateAdditiveDataBufferFromPointDataBuffer(state);

  // Step 5: Update the selectedInfo to reflect new middle nodes
  curPoint = leftPoint;
  state.selectedInfo.nodeDataBuffer = [];

  while (curPoint.id !== rightPoint.id) {
    const nextID = curPoint.rightPointID;
    state.selectedInfo.nodeDataBuffer.push({
      x: curPoint.x,
      y: curPoint.y,
      id: curPoint.id,
      ebmID: curPoint.ebmID
    });
    curPoint = state.pointDataBuffer[nextID];
  }
  
  // Add the right point too
  state.selectedInfo.nodeDataBuffer.push({
    x: curPoint.x,
    y: curPoint.y,
    id: curPoint.id,
    ebmID: curPoint.ebmID
  });

  state.selectedInfo.computeBBoxBuffer(state.pointDataBuffer);

  // Step 6: Update the graph using new data
  drawBufferGraph(state, svg, true, 800, callBack);

};

export const merge = (state, svg, value, callBack) => {
  // Step 1: Find the left and right point in the selected region
  let leftPoint = { x: Infinity, y: null, id: null };
  let rightPoint = { x: -Infinity, y: null, id: null };

  let sum = 0;
  let count = 0;

  state.selectedInfo.nodeData.forEach(d => {
    if (d.x < leftPoint.x) {
      leftPoint = state.pointDataBuffer[d.id];
    }
    if (d.x > rightPoint.x) {
      rightPoint = state.pointDataBuffer[d.id];
    }
    sum += state.pointDataBuffer[d.id].y * state.pointDataBuffer[d.id].count;
    count += state.pointDataBuffer[d.id].count;
  });

  const average = sum / count;

  // Step 2: Iterate through all nodes in the region and assign left node value
  // or the given value to them
  let curPoint = state.pointDataBuffer[leftPoint.id];

  let target = 0;

  if (value === 'left') {
    target = leftPoint.y;
  } else if (value === 'right') {
    target = rightPoint.y;
  } else if (value === 'average') {
    target = average;
  } else {
    target = value;
  }

  while (curPoint.id !== rightPoint.rightPointID) {
    const nextID = curPoint.rightPointID;
    state.pointDataBuffer[curPoint.id].y = target;

    if (nextID === null) {
      break;
    }
    curPoint = state.pointDataBuffer[nextID];
  }

  // Step 3: Recreate the path data using the updated point data  
  updateAdditiveDataBufferFromPointDataBuffer(state);

  // Step 4: update the bbox info
  state.selectedInfo.updateNodeData(state.pointDataBuffer);
  state.selectedInfo.computeBBox(state.pointDataBuffer);

  // Step 5: Update the graph using new data
  drawBufferGraph(state, svg, true, 500, callBack);

  return target;
};

export const drawBufferGraph = (state, svg, animated, duration,
  callBack=() => {}) => {

  const svgSelect = d3.select(svg);

  let trans = d3.transition('buffer')
    .duration(duration)
    .ease(d3.easeCubicInOut)
    .on('end', () => {
      callBack();
    });

  // Change the node data and node position based on the y-value changes
  let nodes = svgSelect.select('g.line-chart-node-group')
    .selectAll('circle.node');

  if (animated) {
    nodes.data(Object.values(state.pointDataBuffer), d => d.id)
      .join(
        enter => enter.append('circle')
          .attr('class', 'node selected')
          .attr('id', d => `node-${d.id}`)
          .attr('cx', d => state.oriXScale(d.x))
          .attr('cy', d => state.oriYScale(d.y))
          .attr('r', rScale(state.curTransform.k))
          .style('opacity', 0)
          .style('stroke-width', nodeStrokeWidth / state.curTransform.k)
          .call(
            enter => enter.transition(trans)
              .style('opacity', 1)
          ),
        update => update.call(
          update => update.transition(trans)
            .attr('cx', d => state.oriXScale(d.x))
            .attr('cy', d => state.oriYScale(d.y))
        ),
        exit => exit.remove()
      );
  } else {
    nodes.data(Object.values(state.pointDataBuffer), d => d.id)
      .join(
        enter => enter.append('circle')
          .attr('class', 'node selected')
          .attr('id', d => `node-${d.id}`)
          .attr('cx', d => state.oriXScale(d.x))
          .attr('cy', d => state.oriYScale(d.y))
          .attr('r', rScale(state.curTransform.k))
          .style('stroke-width', nodeStrokeWidth / state.curTransform.k),
        update => update.attr('cx', d => state.oriXScale(d.x))
          .attr('cy', d => state.oriYScale(d.y)),
        exit => exit.remove()
      );
  }

  // Step 2.2: redraw the paths that are changed
  let paths = svgSelect.select('g.line-chart-line-group.real')
    .selectAll('path.additive-line-segment');

  if (animated) {
    paths.data(state.additiveDataBuffer, d => `${d.id}-${d.pos}`)
      .join(
        enter => enter.append('path')
          .attr('class', 'additive-line-segment selected')
          .attr('id', d => d.id)
          .attr('d', d => {
            // Helpful animation, let's figure out the projection of the end point
            // on the linear interpolation line, and use it as the animation
            // starting point
            let project = projectTriangleSide(d);
            let pathD = '';
            if (d.pos == 'r') {
              pathD = `M ${state.oriXScale(d.x1)}, ${state.oriYScale(d.y1)}
                L ${state.oriXScale(project.x)} ${state.oriYScale(project.y)}`;
            } else {
              pathD = `M ${state.oriXScale(project.x)}, ${state.oriYScale(project.y)}
                L ${state.oriXScale(d.x2)} ${state.oriYScale(d.y2)}`;
            }
            return pathD;
          })
          .style('opacity', 0)
          .call(enter => enter.transition('pre')
            .duration(duration / 2)
            .style('opacity', 1)
          )
          .call(
            enter => enter.transition(trans)
              .attr('d', d => `M ${state.oriXScale(d.x1)}, ${state.oriYScale(d.y1)}
                L ${state.oriXScale(d.x2)} ${state.oriYScale(d.y2)}`)
          ),
        update => update.call(
          update => update.transition(trans)
            .attr('d', d => `M ${state.oriXScale(d.x1)}, ${state.oriYScale(d.y1)}
              L ${state.oriXScale(d.x2)} ${state.oriYScale(d.y2)}`)
        ),
        exit => exit.call(
          exit => exit.style('opacity', 1)
            .transition('pre')
            .duration(duration / 2)
            .style('opacity', 0)
            .remove()
        )
      );
  } else {
    paths.data(state.additiveDataBuffer, d => `${d.id}-${d.pos}`)
      .join(
        enter => enter.append('path')
          .attr('class', 'additive-line-segment selected')
          .attr('id', d => d.id)
          .attr('d', d => `M ${state.oriXScale(d.x1)}, ${state.oriYScale(d.y1)}
                  L ${state.oriXScale(d.x2)} ${state.oriYScale(d.y2)}`),
        update => update.attr('d', d => `M ${state.oriXScale(d.x1)}, ${state.oriYScale(d.y1)}
          L ${state.oriXScale(d.x2)} ${state.oriYScale(d.y2)}`),
        exit => exit.remove()
      );
  }

  // Step 2.3: move the selected bbox
  let curPadding = rScale(state.curTransform.k) + state.bboxPadding * state.curTransform.k;

  if (animated) {
    svgSelect.select('g.line-chart-content-group g.select-bbox-group')
      .selectAll('rect.select-bbox')
      .datum(state.selectedInfo.boundingBox[0])
      .transition(trans)
      .attr('y', d => state.curYScale(d.y1) - curPadding)
      .attr('height', d => state.curYScale(d.y2) - state.curYScale(d.y1) + 2 * curPadding);
  } else {
    svgSelect.select('g.line-chart-content-group g.select-bbox-group')
      .selectAll('rect.select-bbox')
      .datum(state.selectedInfo.boundingBox[0])
      .attr('y', d => state.curYScale(d.y1) - curPadding)
      .attr('height', d => state.curYScale(d.y2) - state.curYScale(d.y1) + 2 * curPadding);
  }

};

export const drawLastEdit = (state, svg) => {

  if (state.additiveDataLastEdit === undefined) {
    return;
  }

  const svgSelect = d3.select(svg);

  let trans = d3.transition('lastEdit')
    .duration(400)
    .ease(d3.easeCubicInOut);

  // Step 2.2: redraw the paths that are changed
  let paths = svgSelect.select('g.line-chart-line-group.last-edit')
    .selectAll('path.additive-line-segment');

  paths.data(state.additiveDataLastEdit, d => `${d.id}-${d.pos}`)
    .join(
      enter => enter.append('path')
        .attr('class', 'additive-line-segment')
        .attr('id', d => d.id)
        .attr('d',  d => `M ${state.oriXScale(d.x1)}, ${state.oriYScale(d.y1)}
                L ${state.oriXScale(d.x2)} ${state.oriYScale(d.y2)}`),
      update => update.call(
        update => update.transition(trans)
          .attr('d', d => `M ${state.oriXScale(d.x1)}, ${state.oriYScale(d.y1)}
              L ${state.oriXScale(d.x2)} ${state.oriYScale(d.y2)}`)
      ),
      exit => exit.remove()
    );
};

/**
 * Project (d.tx, d.sy) onto line [(d.sx, d.sy), (d.tx, d.ty)]
 * @param {Object} d AdditiveData object
 * @returns object of the projection point coordinate
 */
const projectTriangleSide = (d) => {
  const theta = Math.atan2(d.ty, d.tx);
  const pLen = (d.tx - d.sx) * Math.cos(theta);
  const len = Math.sqrt((d.ty - d.sy) ** 2 + (d.tx - d.sx) ** 2);
  let pRatio = pLen / len;
  // let xRange = Math.abs(d.tx - d.sx);
  // let yRange = Math.abs(d.ty - d.sy);
  // pRatio = xRange / (xRange + yRange);
  const pX = d.sx + pRatio * (d.tx - d.sx);
  const pY = d.sy + pRatio * (d.ty - d.sy);
  return {x: pX, y: pY};
};

