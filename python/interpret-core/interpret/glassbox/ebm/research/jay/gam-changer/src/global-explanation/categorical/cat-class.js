import * as d3 from 'd3';

export class SelectedInfo {
  constructor() {
    this.hasSelected = false;
    this.nodeData = [];
    this.boundingBox = [];
    this.nodeDataBuffer = null;
  }

  computeBBox() {
    if (this.nodeData.length > 0) {
      let minIDIndex = -1;
      let maxIDIndex = -1;
      let minID = Infinity;
      let maxID = -Infinity;
      this.nodeData.forEach((d, i) => {
        if (d.id > maxID) {
          maxID = d.id;
          maxIDIndex = i;
        }

        if (d.id < minID) {
          minID = d.id;
          minIDIndex = i;
        }
      });

      this.boundingBox = [{
        x1: this.nodeData[minIDIndex].x,
        y1: d3.max(this.nodeData.map(d => d.y)),
        x2: this.nodeData[maxIDIndex].x,
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