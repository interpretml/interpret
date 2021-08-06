const initDummyEBM = (_featureData, _sampleData, _editingFeature, _isClassification) => {

  class EBM {
    // Store an instance of WASM EBM
    ebm;
    editingFeatureName;

    constructor(featureData, sampleData, editingFeature, isClassification) {

      // Store values for JS object
      this.isClassification = isClassification;
      this.ebm = {};
      this.isDummy = true;
    }

    destroy() {
      this.ebm = {};
    }

    printData() {
      return;
    }

    getProb() {
      return [];
    }

    getScore() {
      return [];
    }

    getPrediction() {
      return [];
    }

    getSelectedSampleNum(binIndexes) {
      return 0;
    }

    getSelectedSampleDist(binIndexes) {
      return [[]];
    }

    getHistBinCounts() {
      return [[]];
    }

    updateModel(changedBinIndexes, changedScores) {
      return;
    }

    setModel(newBinEdges, newScores) {
      return;
    }

    getMetrics() {

      /**
       * (1) regression: [[[RMSE, MAE]]]
       * (2) binary classification: [roc 2D points, [confusion matrix 1D],
       *  [[accuracy, roc auc, balanced accuracy]]]
       */

      // Unpack the return value from getMetrics()
      let metrics = {};
      if (!this.isClassification) {
        metrics.rmse = null;
        metrics.mae = null;
      } else {
        metrics.rocCurve = [];
        metrics.confusionMatrix = [null, null, null, null];
        metrics.accuracy = null;
        metrics.rocAuc = null;
        metrics.balancedAccuracy = null;
      }

      return metrics;
    }

    getMetricsOnSelectedBins(binIndexes) {

      /**
       * (1) regression: [[[RMSE, MAE]]]
       * (2) binary classification: [roc 2D points, [confusion matrix 1D],
       *  [[accuracy, roc auc, balanced accuracy]]]
       */

      // Unpack the return value from getMetrics()
      let metrics = {};
      if (!this.isClassification) {
        metrics.rmse = null;
        metrics.mae = null;
      } else {
        metrics.rocCurve = [];
        metrics.confusionMatrix = [null, null, null, null];
        metrics.accuracy = null;
        metrics.rocAuc = null;
        metrics.balancedAccuracy = null;
      }

      return metrics;
    }

    getMetricsOnSelectedSlice() {
      // Unpack the return value from getMetrics()
      let metrics = {};
      if (!this.isClassification) {
        metrics.rmse = null;
        metrics.mae = null;
      } else {
        metrics.rocCurve = [];
        metrics.confusionMatrix = [null, null, null, null];
        metrics.accuracy = null;
        metrics.rocAuc = null;
        metrics.balancedAccuracy = null;
      }

      return metrics;
    }

    setSliceData(featureID, featureLevel) {
      return;
    }

    /**
     * Change the currently editing feature. If this feature has not been edited
     * before, EBM wasm internally creates a bin-sample mapping for it.
     * Need to call this function before update() or set() ebm on any feature.
     * @param {string} featureName Name of the editing feature
     */
    setEditingFeature(featureName) {
      this.editingFeatureName = featureName;
    }
  }

  let model = new EBM(_featureData, _sampleData, _editingFeature, _isClassification);
  return model;

};

export { initDummyEBM };