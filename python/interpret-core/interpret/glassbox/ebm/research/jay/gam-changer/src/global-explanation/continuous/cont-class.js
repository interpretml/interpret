import * as d3 from 'd3';

export class SelectedInfo {
  constructor() {
    this.hasSelected = false;
    this.nodeData = [];
    this.boundingBox = [];
    this.nodeDataBuffer = null;
  }

  computeBBox(pointData) {
    if (this.nodeData.length > 0) {
      // Get the right x value
      let rightPoint = pointData[this.nodeData.reduce((a, b) => a.x > b.x ? a : b).id];
      // If the right point is the last point, then the right value should be the
      // largest x value from the training set (registered as maxX in the last point)
      // We add 2 as a dirty way to work around the counter-scaling of bbox in zoom
      // and make the bbox's right edge on the right of the whole graph
      // The correct way is to conditionally draw the bbox
      let rightX = rightPoint.rightPointID === null ? rightPoint.maxX : pointData[rightPoint.rightPointID].x;

      this.boundingBox = [{
        x1: d3.min(this.nodeData.map(d => d.x)),
        y1: d3.max(this.nodeData.map(d => d.y)),
        x2: rightX,
        y2: d3.min(this.nodeData.map(d => d.y))
      }];
    } else {
      this.boundingBox = [];
    }
  }

  computeBBoxBuffer() {
    if (this.nodeDataBuffer.length > 0) {
      this.boundingBox = [{
        x1: d3.min(this.nodeDataBuffer.map(d => d.x)),
        y1: d3.max(this.nodeDataBuffer.map(d => d.y)),
        x2: d3.max(this.nodeDataBuffer.map(d => d.x)),
        y2: d3.min(this.nodeDataBuffer.map(d => d.y))
      }];
    } else {
      this.boundingBox = [];
    }
  }

  updateNodeData(pointData) {
    for (let i = 0; i < this.nodeData.length; i++) {
      this.nodeData[i].x = pointData[this.nodeData[i].id].x;
      this.nodeData[i].y = pointData[this.nodeData[i].id].y;
    }
  }
}