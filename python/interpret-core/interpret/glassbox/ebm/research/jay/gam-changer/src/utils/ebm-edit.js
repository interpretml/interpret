/**
 * Overwrite the edge definition in the EBM WASM model.
 * @param {object} curNodeData Node data in `state`
 */
export const getBinEdgeScore = (curNodeData) => {

  // Update the complete bin edge definition in the EBM model
  let newBinEdges = [];
  let newScores = [];

  // The left point will always have index 0
  let curPoint = curNodeData[0];
  let curEBMID = 0;

  while (curPoint.rightPointID !== null) {
    // Collect x and y
    newBinEdges.push(curPoint.x);
    newScores.push(curPoint.y);

    // Update the new ID so we can map them to bin indexes later (needed for
    // selection to check sample number)
    curPoint.ebmID = curEBMID;
    curEBMID++;

    curPoint = curNodeData[curPoint.rightPointID];
  }

  // Add the right node
  newBinEdges.push(curPoint.x);
  newScores.push(curPoint.y);
  curPoint.ebmID = curEBMID;

  return {newBinEdges: newBinEdges, newScores: newScores};
};