import loader from '@assemblyscript/loader';
import { ConsoleImport } from 'as-console/imports.esm.js';

const Console = new ConsoleImport();
const imports = {...Console.wasmImports};


export const initEBM = (_featureData, _sampleData, _editingFeature, _isClassification) => {
  return loader.instantiate(
    fetch('/build/optimized.wasm').then((result) => result.arrayBuffer()),
    imports
  ).then(({ exports }) => {
    Console.wasmExports = exports;
    const wasm = exports;
    const __pin = wasm.__pin;
    const __unpin = wasm.__unpin;
    const __newArray = wasm.__newArray;
    const __getArray = wasm.__getArray;
    const __newString = wasm.__newString;
    const __getString = wasm.__getString;

    /**
     * Convert a JS string array to pointer of string pointers in AS
     * @param {[string]} strings String array
     * @returns Pointer to the array of string pointers
     */
    const __createStringArray = (strings) => {
      let stringPtrs = strings.map(str => __pin(__newString(str)));
      let stringArrayPtr = __pin(__newArray(wasm.StringArray_ID, stringPtrs));
      stringPtrs.forEach(ptr => __unpin(ptr));

      return stringArrayPtr;
    };

    /**
     * Utility function to free a 2D array
     * @param {[[object]]} array2d 2D array
     */
    const __unpin2DArray = (array2d) => {
      for (let i = 0; i < array2d.length; i++) {
        __unpin(array2d[i]);
      }
      __unpin(array2d);
    };

    /**
     * Utility function to free a 3D array
     * @param {[[[object]]]} array2d 3D array
     */
    const __unpin3DArray = (array3d) => {
      for (let i = 0; i < array3d.length; i++) {
        for (let j = 0; j < array3d[i].length; j++) {
          __unpin(array3d[i][j]);
        }
        __unpin(array3d[i]);
      }
      __unpin(array3d);
    };

    class EBM {
      // Store an instance of WASM EBM
      ebm;
      sampleDataNameMap;
      editingFeatureIndex;
      editingFeatureName;

      constructor(featureData, sampleData, editingFeature, isClassification) {

        // Store values for JS object
        this.isClassification = isClassification;

        /**
         * Pre-process the feature data
         *
         * Feature data includes the main effect and also the interaction effect, and
         * we want to split those two.
         */

        // Step 1: For the main effect, we only need bin edges and scores stored with the same order
        // of `featureNames` and `featureTypes`.

        // Create an index map from feature name to their index in featureData
        let featureDataNameMap = new Map();
        featureData.features.forEach((d, i) => featureDataNameMap.set(d.name, i));

        this.sampleDataNameMap = new Map();
        sampleData.featureNames.forEach((d, i) => this.sampleDataNameMap.set(d, i));

        let featureNamesPtr = __createStringArray(sampleData.featureNames);
        let featureTypesPtr = __createStringArray(sampleData.featureTypes);

        let editingFeatureIndex = this.sampleDataNameMap.get(editingFeature);
        this.editingFeatureIndex = editingFeatureIndex;
        this.editingFeatureName = editingFeature;

        // Create two 2D arrays for binEdge ([feature, bin]) and score ([feature, bin]) respectively
        // We mix continuous and categorical together (assume the categorical features
        // have been encoded)
        let binEdges = [];
        let scores = [];


        // We also pass the histogram edges (defined by InterpretML) to WASM. We use
        // WASM EBM to count bin size based on the test set, so that we only iterate
        // the test data once.
        let histBinEdges = [];

        // This loop won't encounter interaction terms
        for (let i = 0; i < sampleData.featureNames.length; i++) {
          let curName = sampleData.featureNames[i];
          let curIndex = featureDataNameMap.get(curName);

          let curScore = featureData.features[curIndex].additive.slice();
          let curBinEdge;
          let curHistBinEdge;

          if (sampleData.featureTypes[i] === 'categorical') {
            curBinEdge = featureData.features[curIndex].binLabel.slice();
            curHistBinEdge = featureData.features[curIndex].histEdge.slice();
          } else {
            curBinEdge = featureData.features[curIndex].binEdge.slice(0, -1);
            curHistBinEdge = featureData.features[curIndex].histEdge.slice(0, -1);
          }

          // Pin the inner 1D arrays
          let curBinEdgePtr = __pin(__newArray(wasm.Float64Array_ID, curBinEdge));
          let curScorePtr = __pin(__newArray(wasm.Float64Array_ID, curScore));
          let curHistBinEdgesPtr = __pin(__newArray(wasm.Float64Array_ID, curHistBinEdge));

          binEdges.push(curBinEdgePtr);
          scores.push(curScorePtr);
          histBinEdges.push(curHistBinEdgesPtr);
        }

        // Pin the 2D arrays
        const binEdgesPtr = __pin(__newArray(wasm.Float64Array2D_ID, binEdges));
        const scoresPtr = __pin(__newArray(wasm.Float64Array2D_ID, scores));
        const histBinEdgesPtr = __pin(__newArray(wasm.Float64Array2D_ID, histBinEdges));

        /**
         * Step 2: For the interaction effect, we want to store the feature
         * indexes and the score.
         *
         * Here we store arrays of indexes(2D), edges(3D), and scores(3D)
         */
        let interactionIndexes = [];
        let interactionScores = [];
        let interactionBinEdges = [];

        featureData.features.forEach((d) => {
          if (d.type === 'interaction') {
            // Parse the feature name
            let index1 = sampleData.featureNames.indexOf(d.name1);
            let index2 = sampleData.featureNames.indexOf(d.name2);

            let curIndexesPtr = __pin(__newArray(wasm.Int32Array_ID, [index1, index2]));
            interactionIndexes.push(curIndexesPtr);

            // Collect two bin edges
            let binEdge1Ptr;
            let binEdge2Ptr;

            // Have to skip the max edge if it is continuous
            if (sampleData.featureTypes[index1] === 'categorical') {
              binEdge1Ptr = __pin(__newArray(wasm.Float64Array_ID, d.binLabel1.slice()));
            } else {
              binEdge1Ptr = __pin(__newArray(wasm.Float64Array_ID, d.binLabel1.slice(0, -1)));
            }

            if (sampleData.featureTypes[index2] === 'categorical') {
              binEdge2Ptr = __pin(__newArray(wasm.Float64Array_ID, d.binLabel2.slice()));
            } else {
              binEdge2Ptr = __pin(__newArray(wasm.Float64Array_ID, d.binLabel2.slice(0, -1)));
            }

            let curBinEdgesPtr = __pin(__newArray(wasm.Float64Array2D_ID, [binEdge1Ptr, binEdge2Ptr]));

            interactionBinEdges.push(curBinEdgesPtr);

            // Add the scores
            let curScore2D = d.additive.map((a) => {
              let aPtr = __pin(__newArray(wasm.Float64Array_ID, a));
              return aPtr;
            });
            let curScore2DPtr = __pin(__newArray(wasm.Float64Array2D_ID, curScore2D));
            interactionScores.push(curScore2DPtr);
          }
        });

        // Create 3D arrays
        let interactionIndexesPtr = __pin(__newArray(wasm.Int32Array2D_ID, interactionIndexes));
        let interactionBinEdgesPtr = __pin(__newArray(wasm.Float64Array3D_ID, interactionBinEdges));
        let interactionScoresPtr = __pin(__newArray(wasm.Float64Array3D_ID, interactionScores));

        /**
         * Step 3: Pass the sample data to WASM. We directly transfer this 2D float
         * array to WASM (assume categorical features are encoded already)
         */
        let samples = sampleData.samples.map((d) => __pin(__newArray(wasm.Float64Array_ID, d)));
        let samplesPtr = __pin(__newArray(wasm.Float64Array2D_ID, samples));
        let labelsPtr = __pin(__newArray(wasm.Float64Array_ID, sampleData.labels));

        /**
         * Step 4: Initialize the EBM in WASM
         */
        this.ebm = wasm.__EBM(
          featureNamesPtr,
          featureTypesPtr,
          binEdgesPtr,
          scoresPtr,
          histBinEdgesPtr,
          featureData.intercept,
          interactionIndexesPtr,
          interactionBinEdgesPtr,
          interactionScoresPtr,
          samplesPtr,
          labelsPtr,
          editingFeatureIndex,
          isClassification
        );
        __pin(this.ebm);

        /**
         * Step 5: free the arrays created to communicate with WASM
         */
        __unpin(labelsPtr);
        __unpin2DArray(samplesPtr);
        __unpin3DArray(interactionScoresPtr);
        __unpin3DArray(interactionBinEdgesPtr);
        __unpin2DArray(interactionIndexesPtr);
        __unpin2DArray(histBinEdgesPtr);
        __unpin2DArray(scoresPtr);
        __unpin2DArray(binEdgesPtr);
        __unpin(featureTypesPtr);
        __unpin(featureNamesPtr);
      }

      /**
       * Free the ebm wasm memory.
       */
      destroy() {
        __unpin(this.ebm);
        this.ebm = null;
      }

      printData() {
        let namePtr = this.ebm.printName();
        let name = __getString(namePtr);
        console.log('Editing: ', name);
      }

      /**
       * Get the current predicted probabilities
       * @returns Predicted probabilities
       */
      getProb() {
        let predProbs = __getArray(this.ebm.getPrediction());
        return predProbs;
      }

      /**
       * Get the current predictions (logits for classification or continuous values
       * for regression)
       * @returns predictions
       */
      getScore() {
        return __getArray(this.ebm.predLabels);
      }

      /**
       * Get the binary classification results
       * @returns Binary predictions
       */
      getPrediction() {
        if (this.isClassification) {
          let predProbs = __getArray(this.ebm.getPrediction());
          return predProbs.map(d => (d >= 0.5 ? 1 : 0));
        }
        return __getArray(this.ebm.getPrediction());
      }

      /**
       * Get the number of test samples affected by the given binIndexes
       * @param {[int]} binIndexes Indexes of bins
       * @returns {int} number of samples
       */
      getSelectedSampleNum(binIndexes) {
        let binIndexesPtr = __pin(__newArray(wasm.Int32Array_ID, binIndexes));
        let count = this.ebm.getSelectedSampleNum(binIndexesPtr);
        __unpin(binIndexesPtr);
        return count;
      }

      /**
       * Get the distribution of the samples affected by the given inIndexes
       * @param {[int]} binIndexes Indexes of bins
       * @returns [[int]] distributions of different bins
       */
      getSelectedSampleDist(binIndexes) {
        let binIndexesPtr = __pin(__newArray(wasm.Int32Array_ID, binIndexes));
        let histBinCounts = __getArray(this.ebm.getSelectedSampleDist(binIndexesPtr));
        histBinCounts = histBinCounts.map(p => __getArray(p));
        __unpin(binIndexesPtr);
        return histBinCounts;
      }

      /**
       * Get the histogram from the training data (from EBM python code)
       * @returns histogram of all bins
       */
      getHistBinCounts() {
        let histBinCounts = __getArray(this.ebm.histBinCounts);
        histBinCounts = histBinCounts.map(p => __getArray(p));
        return histBinCounts;
      }

      /**
       * Change the currently editing feature. If this feature has not been edited
       * before, EBM wasm internally creates a bin-sample mapping for it.
       * Need to call this function before update() or set() ebm on any feature.
       * @param {string} featureName Name of the editing feature
       */
      setEditingFeature(featureName) {
        let featureIndex = this.sampleDataNameMap.get(featureName);
        this.ebm.setEditingFeature(featureIndex);
        this.editingFeatureName = featureName;
        this.editingFeatureIndex = this.sampleDataNameMap.get(featureName);
      }

      /**
       * Change the scores of some bins of a feature.
       * This function assumes setEditingFeature(featureName) has been called once
       * @param {[int]} changedBinIndexes Indexes of bins
       * @param {[float]} changedScores Target scores for these bins
       * @param {string} featureName Name of the feature to update
       */
      updateModel(changedBinIndexes, changedScores, featureName = undefined) {
        // Get the feature index based on the feature name if it is specified
        let featureIndex = this.editingFeatureIndex;
        if (featureName !== undefined) {
          featureIndex = this.sampleDataNameMap.get(featureName);
        }

        let changedBinIndexesPtr = __pin(__newArray(wasm.Int32Array_ID, changedBinIndexes));
        let changedScoresPtr = __pin(__newArray(wasm.Float64Array_ID, changedScores));

        this.ebm.updateModel(changedBinIndexesPtr, changedScoresPtr, featureIndex);

        __unpin(changedBinIndexesPtr);
        __unpin(changedScoresPtr);
      }

      /**
       * Overwrite the whole bin definition for some continuous feature
       * This function assumes setEditingFeature(featureName) has been called once
       * @param {[int]} changedBinIndexes Indexes of all new bins
       * @param {[float]} changedScores Target scores for these bins
       * @param {string} featureName Name of the feature to overwrite
       */
      setModel(newBinEdges, newScores, featureName = undefined) {
        // Get the feature index based on the feature name if it is specified
        let featureIndex = this.editingFeatureIndex;
        if (featureName !== undefined) {
          featureIndex = this.sampleDataNameMap.get(featureName);
        }

        let newBinEdgesPtr = __pin(__newArray(wasm.Float64Array_ID, newBinEdges));
        let newScoresPtr = __pin(__newArray(wasm.Float64Array_ID, newScores));

        this.ebm.setModel(newBinEdgesPtr, newScoresPtr, featureIndex);

        __unpin(newBinEdgesPtr);
        __unpin(newScoresPtr);
      }

      /**
       * Get the metrics
       * @returns {object}
       */
      getMetrics() {

        /**
         * (1) regression: [[[RMSE, MAE]]]
         * (2) binary classification: [roc 2D points, [confusion matrix 1D],
         *  [[accuracy, roc auc, balanced accuracy]]]
         */

        // Unpack the return value from getMetrics()
        let metrics = {};
        if (!this.isClassification) {
          let result3D = __getArray(this.ebm.getMetrics());
          let result3DPtr = __pin(result3D);

          let result2D = __getArray(result3D[0]);
          let result2DPtr = __pin(result2D);

          let result1D = __getArray(result2D[0]);
          let result1DPtr = __pin(result1D);

          metrics.rmse = result1D[0];
          metrics.mae = result1D[1];

          __unpin(result1DPtr);
          __unpin(result2DPtr);
          __unpin(result3DPtr);
        } else {
          // Unpack ROC curves
          let result3D = __getArray(this.ebm.getMetrics());
          let result3DPtr = __pin(result3D);

          let result1DPtrs = [];
          let roc2D = __getArray(result3D[0]);
          let result2DPtr = __pin(roc2D);

          let rocPoints = roc2D.map(d => {
            let point = __getArray(d);
            result1DPtrs.push(__pin(point));
            return point;
          });

          metrics.rocCurve = rocPoints;
          result1DPtrs.map(d => __unpin(d));
          __unpin(result2DPtr);

          // Unpack PR curves
          // result1DPtrs = [];
          // let pr2D = __getArray(result3D[1]);
          // result2DPtr = __pin(roc2D);

          // let prPoints = pr2D.map(d => {
          //   let point = __getArray(d);
          //   result1DPtrs.push(__pin(point));
          //   return point;
          // });

          // metrics.prCurve = prPoints;
          // result1DPtrs.map(d => __unpin(d));
          // __unpin(result2DPtr);

          // Unpack confusion matrix
          let result2D = __getArray(result3D[1]);
          result2DPtr = __pin(result2D);

          let result1D = __getArray(result2D[0]);
          let result1DPtr = __pin(result1D);

          metrics.confusionMatrix = result1D;

          __unpin(result1DPtr);
          __unpin(result2DPtr);

          // Unpack summary statistics
          result2D = __getArray(result3D[2]);
          result2DPtr = __pin(result2D);

          result1D = __getArray(result2D[0]);
          result1DPtr = __pin(result1D);

          metrics.accuracy = result1D[0];
          metrics.rocAuc = result1D[1];
          metrics.balancedAccuracy = result1D[2];

          __unpin(result1DPtr);
          __unpin(result2DPtr);

          __unpin(result3DPtr);
        }

        // let metrics = __getArray(this.ebm.getMetrics());
        return metrics;
      }

      /**
       * Get the metrics on the selected bins
       * @param {[int]} binIndexes Indexes of selected bins
       * @returns {object}
       */
      getMetricsOnSelectedBins(binIndexes) {
        let binIndexesPtr = __pin(__newArray(wasm.Int32Array_ID, binIndexes));

        /**
         * (1) regression: [[[RMSE, MAE]]]
         * (2) binary classification: [roc 2D points, [confusion matrix 1D],
         *  [[accuracy, roc auc, balanced accuracy]]]
         */

        // Unpack the return value from getMetrics()
        let metrics = {};
        if (!this.isClassification) {
          let result3D = __getArray(this.ebm.getMetricsOnSelectedBins(binIndexesPtr));
          let result3DPtr = __pin(result3D);

          let result2D = __getArray(result3D[0]);
          let result2DPtr = __pin(result2D);

          let result1D = __getArray(result2D[0]);
          let result1DPtr = __pin(result1D);

          metrics.rmse = result1D[0];
          metrics.mae = result1D[1];

          __unpin(result1DPtr);
          __unpin(result2DPtr);
          __unpin(result3DPtr);
        } else {
          // Unpack ROC curves
          let result3D = __getArray(this.ebm.getMetricsOnSelectedBins(binIndexesPtr));
          let result3DPtr = __pin(result3D);

          let result1DPtrs = [];
          let roc2D = __getArray(result3D[0]);
          let result2DPtr = __pin(roc2D);

          let rocPoints = roc2D.map(d => {
            let point = __getArray(d);
            result1DPtrs.push(__pin(point));
            return point;
          });

          metrics.rocCurve = rocPoints;
          result1DPtrs.map(d => __unpin(d));
          __unpin(result2DPtr);

          // Unpack confusion matrix
          let result2D = __getArray(result3D[1]);
          result2DPtr = __pin(result2D);

          let result1D = __getArray(result2D[0]);
          let result1DPtr = __pin(result1D);

          metrics.confusionMatrix = result1D;

          __unpin(result1DPtr);
          __unpin(result2DPtr);

          // Unpack summary statistics
          result2D = __getArray(result3D[2]);
          result2DPtr = __pin(result2D);

          result1D = __getArray(result2D[0]);
          result1DPtr = __pin(result1D);

          metrics.accuracy = result1D[0];
          metrics.rocAuc = result1D[1];
          metrics.balancedAccuracy = result1D[2];

          __unpin(result1DPtr);
          __unpin(result2DPtr);
          __unpin(result3DPtr);
        }
        __unpin(binIndexesPtr);

        return metrics;
      }

      /**
       * Get the metrics on the selected slice
       * This function assumes setSliceData() has been called
       * @returns {object}
       */
      getMetricsOnSelectedSlice() {
        // Unpack the return value from getMetrics()
        let metrics = {};
        if (!this.isClassification) {
          let result3D = __getArray(this.ebm.getMetricsOnSelectedSlice());
          let result3DPtr = __pin(result3D);

          let result2D = __getArray(result3D[0]);
          let result2DPtr = __pin(result2D);

          let result1D = __getArray(result2D[0]);
          let result1DPtr = __pin(result1D);

          metrics.rmse = result1D[0];
          metrics.mae = result1D[1];

          __unpin(result1DPtr);
          __unpin(result2DPtr);
          __unpin(result3DPtr);
        } else {
          // Unpack ROC curves
          let result3D = __getArray(this.ebm.getMetricsOnSelectedSlice());
          let result3DPtr = __pin(result3D);

          let result1DPtrs = [];
          let roc2D = __getArray(result3D[0]);
          let result2DPtr = __pin(roc2D);

          let rocPoints = roc2D.map(d => {
            let point = __getArray(d);
            result1DPtrs.push(__pin(point));
            return point;
          });

          metrics.rocCurve = rocPoints;
          result1DPtrs.map(d => __unpin(d));
          __unpin(result2DPtr);

          // Unpack confusion matrix
          let result2D = __getArray(result3D[1]);
          result2DPtr = __pin(result2D);

          let result1D = __getArray(result2D[0]);
          let result1DPtr = __pin(result1D);

          metrics.confusionMatrix = result1D;

          __unpin(result1DPtr);
          __unpin(result2DPtr);

          // Unpack summary statistics
          result2D = __getArray(result3D[2]);
          result2DPtr = __pin(result2D);

          result1D = __getArray(result2D[0]);
          result1DPtr = __pin(result1D);

          metrics.accuracy = result1D[0];
          metrics.rocAuc = result1D[1];
          metrics.balancedAccuracy = result1D[2];

          __unpin(result1DPtr);
          __unpin(result2DPtr);
          __unpin(result3DPtr);
        }

        return metrics;
      }


      /**
       * Set the current sliced data (a level of a categorical feature)
       * @param {int} featureID The index of the categorical feature
       * @param {int} featureLevel The integer encoding of the variable level
       * @returns {int} Number of test samples in this slice
       */
      setSliceData(featureID, featureLevel) {
        return this.ebm.setSliceData(featureID, featureLevel);
      }

    }

    let model = new EBM(_featureData, _sampleData, _editingFeature, _isClassification);
    return model;

  });
};
