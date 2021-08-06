
/**
 * Create rectangles in SVG path format tracing the standard deviations at each
 * point in the model.
 * @param featureData
 */
export const createConfidenceData = (featureData) => {

  let confidenceData = [];

  for (let i = 0; i < featureData.additive.length; i++) {
    let curValue = featureData.additive[i];
    let curError = featureData.error[i];

    confidenceData.push({
      x1: featureData.binEdge[i],
      y1: curValue + curError,
      x2: featureData.binEdge[i + 1],
      y2: curValue - curError
    });
  }

  // Right bound
  let rightValue = featureData.additive[featureData.additive.length - 1];
  let rightError = featureData.error[featureData.additive.length - 1];

  confidenceData.push({
    x1: featureData.binEdge[featureData.additive.length - 1],
    y1: rightValue + rightError,
    x2: featureData.binEdge[featureData.additive.length - 1],
    y2: rightValue - rightError
  });

  return confidenceData;
};

/**
 * Create line segments (path) to trace the additive term at each bin in the
 * model.
 * @param featureData
 */
export const createAdditiveData = (featureData) => {
  let additiveData = [];

  for (let i = 0; i < featureData.additive.length - 1; i++) {

    // Compute the source point and the target point
    let sx = featureData.binEdge[i];
    let sy = featureData.additive[i];
    let tx = featureData.binEdge[i + 1];
    let ty = featureData.additive[i + 1];

    // Add line segments (need two segments to connect two points)
    // We separate these two lines so it is easier to drag
    additiveData.push({
      x1: sx,
      y1: sy,
      x2: tx,
      y2: sy,
      id: `path-${i}-${i+1}-r`,
      pos: 'r',
      sx: sx,
      sy: sy,
      tx: tx,
      ty: ty
    });

    additiveData.push({
      x1: tx,
      y1: sy,
      x2: tx,
      y2: ty,
      id: `path-${i}-${i + 1}-l`,
      pos: 'l',
      sx: sx,
      sy: sy,
      tx: tx,
      ty: ty
    });
  }

  // Connect the last two points (because max point has no additive value, it
  // does not have a left edge)
  additiveData.push({
    x1: featureData.binEdge[featureData.additive.length - 1],
    y1: featureData.additive[featureData.additive.length - 1],
    x2: featureData.binEdge[featureData.additive.length],
    y2: featureData.additive[featureData.additive.length - 1],
    id: `path-${featureData.additive.length - 1}-${featureData.additive.length - 1}-r`,
    pos: 'r',
    sx: featureData.binEdge[featureData.additive.length - 1],
    sy: featureData.additive[featureData.additive.length - 1],
    tx: featureData.binEdge[featureData.additive.length],
    ty: featureData.additive[featureData.additive.length - 1]
  });

  return additiveData;
};

/**
 * Create nodes where each step function begins
 * @param featureData
 */
export const createPointData = (featureData) => {
  let pointData = {};

  for (let i = 0; i < featureData.additive.length; i++) {
    pointData[i] = {
      x: featureData.binEdge[i],
      y: featureData.additive[i],
      count: featureData.count[i],
      id: i,
      ebmID: i,
      leftPointID: i == 0 ? null : i - 1,
      rightPointID: i == featureData.additive.length - 1 ? null : i + 1,
      leftLineIndex: null,
      rightLineIndex: null
    };
  }

  // Since the last point has no additive value, it is not included in the
  // point data array, we need to separately record its x value
  // We register it to the last point
  pointData[featureData.additive.length - 1].maxX = featureData.binEdge[featureData.additive.length];

  return pointData;
};

export const linkPointToAdditive = (pointData, additiveData) => {
  additiveData.forEach( (d, i) => {
    if (d.pos === 'r') {
      let curID = d.id.replace(/path-(\d+)-(\d+)-[lr]/, '$1');
      pointData[curID].rightLineIndex = i;
    } else {
      let curID = d.id.replace(/path-(\d+)-(\d+)-[lr]/, '$2');
      pointData[curID].leftLineIndex = i;
    }
  });
};

/**
 * Create a new additiveDataBuffer array from the current pointDataBuffer object
 * This function modifies the state in-place
 * This function also update the leftLineIndex/rightLineIndex in the pointDataBuffer
 */
export const updateAdditiveDataBufferFromPointDataBuffer = (state) => {
  let newAdditiveData = [];

  // Find the start point of all graph
  let curPoint = state.pointDataBuffer[Object.keys(state.pointDataBuffer)[0]];
  while (curPoint.leftPointID !== null) {
    curPoint = state.pointDataBuffer[curPoint.leftPointID];
  }

  // Iterate through all the points from the starting point
  let curLineIndex = 0;
  let nextPoint = state.pointDataBuffer[curPoint.rightPointID];
  while (curPoint.rightPointID !== null) {

    newAdditiveData.push({
      x1: curPoint.x,
      y1: curPoint.y,
      x2: nextPoint.x,
      y2: curPoint.y,
      id: `path-${curPoint.id}-${nextPoint.id}-r`,
      pos: 'r',
      sx: curPoint.x,
      sy: curPoint.y,
      tx: nextPoint.x,
      ty: nextPoint.y
    });

    curPoint.rightLineIndex = curLineIndex;
    curLineIndex++;

    newAdditiveData.push({
      x1: nextPoint.x,
      y1: curPoint.y,
      x2: nextPoint.x,
      y2: nextPoint.y,
      id: `path-${curPoint.id}-${nextPoint.id}-l`,
      pos: 'l',
      sx: curPoint.x,
      sy: curPoint.y,
      tx: nextPoint.x,
      ty: nextPoint.y
    });

    nextPoint.leftLineIndex = curLineIndex;
    curLineIndex++;

    curPoint = nextPoint;
    nextPoint = state.pointDataBuffer[curPoint.rightPointID];
  }

  // Connect the last two points (because max point has no additive value, it
  // does not have a left edge)
  newAdditiveData.push({
    x1: curPoint.x,
    y1: curPoint.y,
    x2: state.oriXScale.domain()[1],
    y2: curPoint.y,
    id: `path-${curPoint.id}-${curPoint.id}-r`,
    pos: 'r',
    sx: curPoint.x,
    sy: curPoint.y,
    tx: state.oriXScale.domain()[1],
    ty: curPoint.y
  });

  curPoint.rightLineIndex = curLineIndex;

  state.additiveDataBuffer = newAdditiveData;
};