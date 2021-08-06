import * as d3 from 'd3';
import { SelectedInfo } from './cont-class';
import { moveMenubar } from './cont-bbox';
import { rScale } from './cont-zoom';
import { redrawOriginal, drawLastEdit } from './cont-edit';
import { config } from '../../config';

// Need a timer to avoid the brush event call after brush.move()
let idleTimeout = null;
const idleDelay = 300;

// Brush zooming
const zoomTransitionTime = 700;

/**
 * Reset the idleTimeout timer
 */
const idled = () => {
  idleTimeout = null;
};

export const brushDuring = (event, state, svg, multiMenu, ebm, footerStore) => {
  // Get the selection boundary
  let selection = event.selection;
  let svgSelect = d3.select(svg);

  if (selection === null) {
    if (idleTimeout === null) {
      return idleTimeout = setTimeout(idled, idleDelay);
    }
  } else {
    // Compute the selected data region
    let xRange = [state.curXScale.invert(selection[0][0]), state.curXScale.invert(selection[1][0])];
    let yRange = [state.curYScale.invert(selection[1][1]), state.curYScale.invert(selection[0][1])];

    // Save the selectedInfo, we might need to recover EBM with it
    // state.lastSelectedInfo = JSON.parse(JSON.stringify(state.selectedInfo));
    state.lastSelectedInfo = Object.assign(Object.create(
      Object.getPrototypeOf(state.selectedInfo)), state.selectedInfo);

    // Clean up the previous flowing lines
    state.selectedInfo = new SelectedInfo();

    // Remove the selection bbox
    svgSelect.selectAll('g.line-chart-content-group g.select-bbox-group').remove();

    d3.select(multiMenu)
      .classed('hidden', true);

    // Highlight the selected dots
    let selectedBinIndexes = [];
    let selectedBinIDs = [];

    svgSelect.select('g.line-chart-node-group')
      .selectAll('circle.node')
      .classed('selected', d => {
        if (d.x >= xRange[0] && d.x <= xRange[1] && d.y >= yRange[0] && d.y <= yRange[1]) {
          selectedBinIndexes.push(d.ebmID);
          selectedBinIDs.push([d.id, d.x]);
          return true;
        } else if (d.rightPointID !== null){
          let rd = state.pointData[d.rightPointID];
          if (d.y >= yRange[0] && d.y <= yRange[1] && rd.x >= xRange[0] &&
              rd.x <= xRange[1] && rd.y >= yRange[0] && rd.y <= yRange[1]) {
            selectedBinIndexes.push(d.ebmID);
            selectedBinIDs.push([d.id, d.x]);
            return true;
          }
        } else {
          return false;
        }
      });

    selectedBinIDs.sort((a, b) => a[1] - b[1]);
      
    svgSelect.select('g.line-chart-line-group.real')
      .selectAll('path.additive-line-segment')
      .classed('selected', false);

    selectedBinIDs.forEach((pair, i) => {
      let id = pair[0];
      svgSelect.select('g.line-chart-line-group.real')
        .select(`#path-${id}-${state.pointData[id].rightPointID}-r`)
        .classed('selected', true);

      if (i !== selectedBinIDs.length - 1) {
        svgSelect.select('g.line-chart-line-group.real')
          .select(`#path-${id}-${state.pointData[id].rightPointID}-l`)
          .classed('selected', true);
      } else {
        // Last path is named as something like path-99-99-r
        svgSelect.select('g.line-chart-line-group.real')
          .select(`#path-${id}-${id}-r`)
          .classed('selected', true);
      }
    });

    // Update the footer message
    footerStore.update(value => {
      let sampleNum = ebm.getSelectedSampleNum(selectedBinIndexes);
      value.sample = `<b>${sampleNum}/${value.totalSampleNum}</b> test samples selected`;
      return value;
    });
  }
};

/**
 * Discard the current marquee selection. This function does not handle any graph
 * drawing/redrawing.
 * @param {object} svg The svg object
 * @param {object} multiMenu The multimenu object
 * @param {func} resetContextMenu Function to reset the context menu
 * @param {func} resetFeatureSidebar Function to reset the feature sidebar
 */
export const quitSelection = (svg, state, multiMenu, resetContextMenu, resetFeatureSidebar) => {
  let svgSelect = d3.select(svg);

  state.selectedInfo = new SelectedInfo();

  // De-highlight the paths associated with the selected dots
  svgSelect.select('g.line-chart-node-group')
    .selectAll('circle.node')
    .classed('selected', false);

  svgSelect.select('g.line-chart-line-group.real')
    .selectAll('path.additive-line-segment')
    .classed('selected', false);

  svgSelect.select('g.line-chart-content-group g.brush rect.overlay')
    .attr('cursor', null);

  d3.select(multiMenu)
    .classed('hidden', true);

  // End move mode
  resetContextMenu();

  // Remove the selection bbox
  svgSelect.selectAll('g.line-chart-content-group g.select-bbox-group').remove();

  // Reset the feature sidebar
  resetFeatureSidebar();
};

export const brushEndSelect = (event, state, svg, multiMenu, myContextMenu,
  bboxStrokeWidth, brush, component, resetContextMenu, sidebarStore, setEBM,
  updateEBM, updateFeatureSidebar, resetFeatureSidebar, nullifyMetrics,
  computeSelectedEffects, cancelTooltip
) => {
  // Get the selection boundary
  let selection = event.selection;
  let svgSelect = d3.select(svg);

  if (selection === null) {
    if (idleTimeout === null) {

      svgSelect.select('g.line-chart-content-group g.brush rect.overlay')
        .attr('cursor', null);

      d3.select(multiMenu)
        .classed('hidden', true);

      myContextMenu.mouseleaveHandler();

      // End move mode
      let modeInfo = resetContextMenu();

      let callBack = () => {};

      // If the current edit is interpolation, we need to recover the bin definition
      // in the EBM model
      if (modeInfo.subItemMode === 'interpolation') {
        callBack = () => {
          setEBM('current', state.pointData);
        };
        
      } else if (modeInfo.moveMode || modeInfo.subItemMode !== null) {
        
        callBack = () => {
          // We are using lastSelectedInfo here because selectedInfo is purged
          // in brushDuring
          updateEBM('recoverEBM', state.lastSelectedInfo.nodeData);
          state.lastSelectedInfo = null;
        };
      }

      // Redraw the last edit if possible
      if (modeInfo.moveMode || modeInfo.subItemMode !== null) {
        // Do not save the user's change (same as clicking the cancel button)
        // Redraw the graph with original data
        redrawOriginal(state, svg, true, callBack);

        if (state.additiveDataLastLastEdit !== undefined) {
          state.additiveDataLastEdit = JSON.parse(JSON.stringify(state.additiveDataLastLastEdit));
          drawLastEdit(state, svg);
          // Prepare for next redrawing after recovering the last last edit graph
          state.additiveDataLastEdit = JSON.parse(JSON.stringify(state.additiveData));
        }
      }

      // Recover the metrics if user is quitting context menu without committing
      sidebarStore.update(value => {
        // Svelte would trigger an update when update() is called
        // So if we want to avoid call 'recover' twice, we need to set another
        // message
        if (modeInfo.moveMode || modeInfo.subItemMode !== null) {
          value.curGroup = 'recover';
        } else {
          value.curGroup = 'no action';
        }
        return value;
      });

      // Remove the selection bbox
      svgSelect.selectAll('g.line-chart-content-group g.select-bbox-group').remove();

      // Reset the feature sidebar
      resetFeatureSidebar();

      // Nullify the metrics if in selected tab
      nullifyMetrics();

      return idleTimeout = setTimeout(idled, idleDelay);
    }
  } else {

    // Compute the selected data region
    let xRange = [state.curXScale.invert(selection[0][0]), state.curXScale.invert(selection[1][0])];
    let yRange = [state.curYScale.invert(selection[1][1]), state.curYScale.invert(selection[0][1])];

    let selectedBinIndexes = [];
    let selectedBinIDs = [];

    // Highlight the selected dots
    svgSelect.select('g.line-chart-node-group')
      .selectAll('circle.node')
      .classed('selected', d => {
        if (d.x >= xRange[0] && d.x <= xRange[1] && d.y >= yRange[0] && d.y <= yRange[1]) {
          selectedBinIndexes.push(d.ebmID);
          selectedBinIDs.push([d.id, d.x]);
          return true;
        } else if (d.rightPointID !== null) {
          let rd = state.pointData[d.rightPointID];
          if (d.y >= yRange[0] && d.y <= yRange[1] && rd.x >= xRange[0] &&
            rd.x <= xRange[1] && rd.y >= yRange[0] && rd.y <= yRange[1]) {
            selectedBinIndexes.push(d.ebmID);
            selectedBinIDs.push([d.id, d.x]);
            return true;
          }
        } else {
          return false;
        }
      });

    // Force to select the missed middle nodes
    const selectedXRange = d3.extent(selectedBinIDs.map(d => state.pointData[d[0]].x));

    // Reset the IDs here because we will select all again with the new condition
    selectedBinIndexes = [];
    selectedBinIDs = [];

    svgSelect.select('g.line-chart-node-group')
      .selectAll('circle.node')
      .classed('selected', d => {
        if (d.x >= selectedXRange[0] && d.x <= selectedXRange[1]) {
          selectedBinIndexes.push(d.ebmID);
          selectedBinIDs.push([d.id, d.x]);
          state.selectedInfo.nodeData.push({ x: d.x, y: d.y, id: d.id, ebmID: d.ebmID });
          return true;
        } else {
          return false;
        }
      });

    selectedBinIDs.sort((a, b) => a[1] - b[1]);

    svgSelect.select('g.line-chart-line-group.real')
      .selectAll('path.additive-line-segment')
      .classed('selected', false);

    selectedBinIDs.forEach((pair, i) => {
      let id = pair[0];
      svgSelect.select('g.line-chart-line-group.real')
        .select(`#path-${id}-${state.pointData[id].rightPointID}-r`)
        .classed('selected', true);

      if (i !== selectedBinIDs.length - 1) {
        svgSelect.select('g.line-chart-line-group.real')
          .select(`#path-${id}-${state.pointData[id].rightPointID}-l`)
          .classed('selected', true);
      } else {
        // Last path is named as something like path-99-99-r
        svgSelect.select('g.line-chart-line-group.real')
          .select(`#path-${id}-${id}-r`)
          .classed('selected', true);
      }
    });

    // Compute the bounding box
    state.selectedInfo.computeBBox(state.pointData);

    let curPadding = rScale(state.curTransform.k) + state.bboxPadding * state.curTransform.k;

    let bbox = svgSelect.select('g.line-chart-content-group')
      .append('g')
      .attr('class', 'select-bbox-group')
      .selectAll('rect.select-bbox')
      .data(state.selectedInfo.boundingBox)
      .join('rect')
      .attr('class', 'select-bbox original-bbox')
      .attr('x', d => state.curXScale(d.x1) - curPadding)
      .attr('y', d => state.curYScale(d.y1) - curPadding)
      .attr('width', d => {
        if (state.selectedInfo.nodeData.length === 1) {
          return state.curXScale(d.x2) - state.curXScale(d.x1) + 2 * curPadding;
        } else {
          return state.curXScale(d.x2) - state.curXScale(d.x1) + curPadding;
        }
      })
      .attr('height', d => state.curYScale(d.y2) - state.curYScale(d.y1) + 2 * curPadding)
      .style('stroke-width', bboxStrokeWidth)
      .style('stroke', 'hsl(230, 100%, 10%)')
      .style('stroke-dasharray', '5 3');

    bbox.clone(true)
      .classed('original-bbox', false)
      .style('stroke', config.colors.background)
      .style('stroke-dasharray', null)
      .style('stroke-width', bboxStrokeWidth * 3)
      .lower();

    state.selectedInfo.hasSelected = svgSelect.selectAll('g.line-chart-node-group circle.node.selected').size() > 0;

    if (state.selectedInfo.hasSelected) {
      // Show the context menu near the selected region
      d3.select(multiMenu)
        .call(moveMenubar, svg, component)
        .classed('hidden', false);

      // Trigger a counting of the feature distribution of the selected samples
      updateFeatureSidebar(selectedBinIndexes);
    }

    // Nullify the metrics if in selected tab and no selection
    nullifyMetrics();

    // Recompute the selected effects if in selected tab and we do have selection
    computeSelectedEffects();

    // Remove the brush box
    svgSelect.select('g.line-chart-content-group g.brush')
      .call(brush.move, null)
      .select('rect.overlay')
      .attr('cursor', null);
  }
};

export const selectAllBins = (svg, state, bboxStrokeWidth, multiMenu, component,
  updateFeatureSidebar, nullifyMetrics, computeSelectedEffects, brush) => {
  let xRange = [-Infinity, Infinity];
  let yRange = [-Infinity, Infinity];

  let selectedBinIndexes = [];
  let svgSelect = d3.select(svg);

  // Highlight the selected dots
  svgSelect.select('g.line-chart-node-group')
    .selectAll('circle.node')
    .classed('selected', d => {
      if (d.x >= xRange[0] && d.x <= xRange[1] && d.y >= yRange[0] && d.y <= yRange[1]) {
        selectedBinIndexes.push(d.ebmID);
        state.selectedInfo.nodeData.push({ x: d.x, y: d.y, id: d.id, ebmID: d.ebmID });
        return true;
      } else if (d.rightPointID !== null) {
        let rd = state.pointData[d.rightPointID];
        if (d.y >= yRange[0] && d.y <= yRange[1] && rd.x >= xRange[0] &&
          rd.x <= xRange[1] && rd.y >= yRange[0] && rd.y <= yRange[1]) {
          selectedBinIndexes.push(d.ebmID);
          state.selectedInfo.nodeData.push({ x: d.x, y: d.y, id: d.id, ebmID: d.ebmID });
          return true;
        }
      } else {
        return false;
      }
    });

  svgSelect.select('g.line-chart-line-group.real')
    .selectAll('path')
    .classed('selected', true);

  svgSelect.select('g.line-chart-node-group')
    .selectAll('circle.node')
    .classed('selected', true);

  // Compute the bounding box
  state.selectedInfo.computeBBox(state.pointData);

  let curPadding = rScale(state.curTransform.k) + state.bboxPadding * state.curTransform.k;

  let bbox = svgSelect.select('g.line-chart-content-group')
    .append('g')
    .attr('class', 'select-bbox-group')
    .selectAll('rect.select-bbox')
    .data(state.selectedInfo.boundingBox)
    .join('rect')
    .attr('class', 'select-bbox original-bbox')
    .attr('x', d => state.curXScale(d.x1) - curPadding)
    .attr('y', d => state.curYScale(d.y1) - curPadding)
    .attr('width', d => state.curXScale(d.x2) - state.curXScale(d.x1) + curPadding)
    .attr('height', d => state.curYScale(d.y2) - state.curYScale(d.y1) + 2 * curPadding)
    .style('stroke-width', bboxStrokeWidth)
    .style('stroke', 'hsl(230, 100%, 10%)')
    .style('stroke-dasharray', '5 3');

  bbox.clone(true)
    .classed('original-bbox', false)
    .style('stroke', config.colors.background)
    .style('stroke-dasharray', null)
    .style('stroke-width', bboxStrokeWidth * 3)
    .lower();

  state.selectedInfo.hasSelected = svgSelect.selectAll('g.line-chart-node-group circle.node.selected').size() > 0;

  if (state.selectedInfo.hasSelected) {
    // Show the context menu near the selected region
    d3.select(multiMenu)
      .call(moveMenubar, svg, component)
      .classed('hidden', false);

    // Trigger a counting of the feature distribution of the selected samples
    updateFeatureSidebar(selectedBinIndexes);
  }

  // Nullify the metrics if in selected tab and no selection
  nullifyMetrics();

  // Recompute the selected effects if in selected tab and we do have selection
  computeSelectedEffects();

  // Remove the brush box
  svgSelect.select('g.line-chart-content-group g.brush')
    .call(brush.move, null)
    .select('rect.overlay')
    .attr('cursor', null);
};

export const brushEndZoom = (event, xScale, yScale, initXDomain, initYDomain, svg, brush) => {
  // Get the selection boundary
  let selection = event.selection;

  // If there is no selection, return to the initial stage
  // Double click returns to the initial stage
  if (selection === null) {
    if (idleTimeout === null) {
      return idleTimeout = setTimeout(idled, idleDelay);
    }

    xScale.domain(initXDomain);
    yScale.domain(initYDomain);
  } else {
    // Rescale the x and y axises
    xScale.domain([xScale.invert(selection[0][0]), xScale.invert(selection[1][0])]);
    yScale.domain([yScale.invert(selection[1][1]), yScale.invert(selection[0][1])]);

    // Remove the brush box
    d3.select(svg)
      .select('g.line-chart-content-group g.brush')
      .call(brush.move, null);
  }

  // Zoom in to the new selection
  brushZoom(xScale, yScale);
};

const brushZoom = (xScale, yScale, svg) => {

  // Create a common transition
  let svgSelect = d3.select(svg);
  let trans = svgSelect.transition('zoom')
    .duration(zoomTransitionTime);

  // Update the axises
  svgSelect.select('g.x-axis')
    .transition(trans)
    .call(d3.axisBottom(xScale));

  svgSelect.select('g.y-axis')
    .transition(trans)
    .call(d3.axisLeft(yScale));

  // Redraw the lines using the new scale
  svgSelect.select('g.line-chart-line-group')
    .selectAll('path.additive-line-segment')
    .transition(trans)
    .attr('d', d => {
      return `M ${xScale(d.x1)}, ${yScale(d.y1)} L ${xScale(d.x2)} ${yScale(d.y2)}`;
    });

};