import * as d3 from 'd3';

/**
 * Use the selection bbox to compute where to put the context menu bar
*/
export const moveMenubar = (menubar, svg, component) => {
  const bbox = d3.select(svg)
    .select('g.select-bbox-group rect.select-bbox');

  if (bbox.node() === null) return;

  const bboxPosition = bbox.node().getBoundingClientRect();
  const panelBboxPosition = component.getBoundingClientRect();

  const menuBarBbox = menubar.node().getBoundingClientRect();
  const menuWidth = +menubar.select('.menu-wrapper').attr('data-max-width');
  const menuHeight = menuBarBbox.height;

  let left = bboxPosition.x - panelBboxPosition.x + bboxPosition.width / 2 - menuWidth / 2;
  let top = bboxPosition.y - panelBboxPosition.y - menuHeight - 40;

  // Do not move the bar out of its parent
  left = Math.max(0, left);
  top = Math.max(0, top);

  menubar.style('left', `${left}px`)
    .style('top', `${top}px`);
};