/**
 * The entry file of the WebAssembly module.
 * 
 * Author: Jay Wang (jayw@gatech.edu)
 */

import { console } from 'as-console';
import { rootMeanSquaredError, meanAbsoluteError, countByThreshold, getROCCurve,
  getPRCurve, getROCAuc, getAveragePrecision, getAccuracy, getConfusionMatrix,
  getBalancedAccuracy
} from './metrics';

/**
 * Find the lower bound of a pair between where inserting `value` into `sorted`
 * would keep `sorted` in order.
 * @param sorted a sorted array (ascending order)
 * @param value a number to insert into `sorted`
 * @returns the lower bound index in the sorted array to insert
 */
export function searchSortedLowerIndex(sorted: Array<f64>, value: f64): i32 {
  let left: i32 = 0;
  let right: i32 = sorted.length - 1;

  while (right - left > 1) {
    let i: i32 = left + Math.floor((right - left) / 2) as i32;

    if (value > sorted[i]) {
      left = i;
    } else if (value < sorted[i]) {
      right = i;
    } else {
      return i;
    }
  }

  // Handle out of bound issue
  if (value >= sorted[right]) {
    return right;
  }
  if (value < sorted[left]) {
    return left;
  }
  return right - 1;
}

function round(num: f64, decimal: i32): f64 {
  return Math.round((num + 2e-16) * (10 ** decimal)) / (10 ** decimal);
};

function sigmoid(logit: f64): f64 {
  let odd = Math.exp(logit);

  // Round the prob for more stable ROC AUC computation
  return round(odd / (1 + odd), 3);
}

export class __EBM {

  // --- Initialization values ---

  // Feature information
  featureNames: Array<string>;
  featureTypes: Array<string>;

  // Feature bin edges and additive score
  binEdges: Array<Array<f64>>;
  scores: Array<Array<f64>>;
  histBinEdges: Array<Array<f64>>;
  intercept: f64;

  // Bin edges and scores of interaction terms
  interactionIndexes: Array<Array<i32>>;
  interactionScores: Array<Array<Array<f64>>>;
  interactionBinEdges: Array<Array<Array<f64>>>;

  // The test dataset
  samples: Array<Array<f64>>;
  labels: Array<f64>;

  editingFeatureIndex: i32;
  isClassification: bool;

  // --- Values needed to be computed ---
  
  // Current prediction
  predLabels: Array<f64>;
  predProbs: Array<f64>;
  editingFeatureSampleMap: Array<Array<i32>>;
  editingFeatureSampleMaps: Map<i32, Array<Array<i32>>>;
  histBinCounts: Array<Array<i32>>;

  // Track the sample IDs of the selected slice
  sliceSampleIDs: Array<i32>;

  /**
   * 
   * @param featureNames Feature names
   * @param featureTypes Feature types ('continuous', 'categorical')
   * @param binEdges Bin left point (continuous) or labels (categorical)
   * @param scores Bin additive score
   * @param intercept The intercept score
   * @param interactionIndexes Feature indexes of each interaction pair
   * @param interactionBinEdges Array of pairs of bin edges for each interaction pair
   * @param interactionScores Array of 2D additive scores for each interaction pair
   * @param samples The data matrix [# of samples, # of features]
   * @param labels The data labels [# of samples]
   */
  constructor(
    featureNames: Array<string>,
    featureTypes: Array<string>,
    binEdges: Array<Array<f64>>,
    scores: Array<Array<f64>>,
    histBinEdges: Array<Array<f64>>,
    intercept: f64,
    interactionIndexes: Array<Array<i32>>,
    interactionBinEdges: Array<Array<Array<f64>>>,
    interactionScores: Array<Array<Array<f64>>>,
    samples: Array<Array<f64>>,
    labels: Array<f64>,
    editingFeatureIndex: i32,
    isClassification: bool
  ) {

    // Step 1: Initialize properties from the arguments
    this.featureNames = featureNames;
    this.featureTypes = featureTypes;
    this.binEdges = binEdges;
    this.scores = scores;
    this.histBinEdges = histBinEdges;
    this.intercept = intercept;
    this.interactionIndexes = interactionIndexes;
    this.interactionBinEdges = interactionBinEdges;
    this.interactionScores = interactionScores;
    this.samples = samples;
    this.labels = labels;
    this.editingFeatureIndex = editingFeatureIndex;
    this.isClassification = isClassification;

    /**
     * Step 2: Iterate through the sample data to initialize
     * - Current prediction
     * - Editing feature's bin bucket info
     * - Histogram counts
     */
    this.predLabels = new Array<f64>(this.labels.length).fill(this.intercept);
    this.predProbs = new Array<f64>(this.predLabels.length).fill(0.0);

    this.editingFeatureSampleMap = new Array<Array<i32>>(this.binEdges[this.editingFeatureIndex].length);

    this.histBinCounts = new Array<Array<i32>>(this.histBinEdges.length);

    // Initialize the editing feature map
    for (let b = 0; b < this.binEdges[this.editingFeatureIndex].length; b++) {
      this.editingFeatureSampleMap[b] = new Array<i32>();
    }

    // We use editingFeatureSampleMaps to track all the feature maps
    this.editingFeatureSampleMaps = new Map<i32, Array<Array<i32>>>();
    this.editingFeatureSampleMaps.set(this.editingFeatureIndex, this.editingFeatureSampleMap);

    // Initialize the hist bin counts
    for (let b = 0; b < this.binEdges.length; b++) {
      this.histBinCounts[b] = new Array<i32>(this.histBinEdges[b].length).fill(0);
    }

    for (let i = 0; i < this.samples.length; i++) {
      // Add main effect scores
      for (let j = 0; j < this.samples[0].length; j++) {
        
        let curFeatureName = this.featureNames[j];
        let curFeatureType = this.featureTypes[j];
        let curFeature = this.samples[i][j];

        // Use the feature value to find the corresponding bin
        let binIndex: i32;
        let binScore: f64;
        let histBinIndex: i32;

        if (curFeatureType == 'continuous') {
          binIndex = searchSortedLowerIndex(this.binEdges[j], curFeature);
          binScore = this.scores[j][binIndex];
          histBinIndex = searchSortedLowerIndex(this.histBinEdges[j], curFeature);
        } else {
          binIndex = this.binEdges[j].indexOf(curFeature);
          if (binIndex < 0) {
            // Unseen level during training => use 0 as score instead
            console.log(`[WASM] Unseen categorical level: ${curFeatureName}, ${i}, ${j}, ${curFeature}`);
            binScore = 0
          } else {
            binScore = this.scores[j][binIndex];
          }

          histBinIndex = this.histBinEdges[j].indexOf(curFeature);
          if (histBinIndex < 0) {
            // Unseen level during training => use 0 as score instead
            // console.log(`[WASM] Unseen categorical level in histogram: ${curFeatureName}, ${i}, ${j}, ${curFeature}`);
            histBinIndex = 0
          }
        }

        // Add the current score to prediction
        this.predLabels[i] += binScore;

        // If we encounter the editing feature, we also want to collect the sample
        // IDs for each bin
        if (j == this.editingFeatureIndex && binIndex >= 0) {
          this.editingFeatureSampleMap[binIndex].push(i);
        }

        // Add the histogram count
        this.histBinCounts[j][histBinIndex] ++;
      }

      // Add interaction effect scores
      for (let j = 0; j < this.interactionIndexes.length; j++) {
        let curIndexes = this.interactionIndexes[j];
        let type1 = this.featureTypes[curIndexes[0]];
        let type2 = this.featureTypes[curIndexes[1]];

        let value1 = this.samples[i][curIndexes[0]];
        let value2 = this.samples[i][curIndexes[1]];

        // Figure out which bin to query along two dimensions
        let binIndex1: i32;
        let binIndex2: i32;

        if (type1 == 'continuous') {
          binIndex1 = searchSortedLowerIndex(this.interactionBinEdges[j][0], value1);
        } else {
          binIndex1 = this.interactionBinEdges[j][0].indexOf(value1);
        }

        if (type2 == 'continuous') {
          binIndex2 = searchSortedLowerIndex(this.interactionBinEdges[j][1], value2);
        } else {
          binIndex2 = this.interactionBinEdges[j][1].indexOf(value2);
        }

        // Query the bin scores
        let binScore: f64;

        if (binIndex1 < 0 || binIndex2 < 0) {
          binScore = 0;
        } else {
          binScore = this.interactionScores[j][binIndex1][binIndex2];
        }

        // Add the current score to prediction
        this.predLabels[i] += binScore;
      }

      // If it is a classifier, then we have to convert the logit to prob
      if (this.isClassification) {
        this.predProbs[i] = sigmoid(this.predLabels[i]);
      }
    }

    /**
     * Step 3: properties that will be initialized later
     */

    this.sliceSampleIDs = [];
  }

  /**
   * Compute the number of test samples in the given bins
   * @param binIndexes Bin indexes of a interested region
   */
  getSelectedSampleNum(binIndexes: Array<i32>): i32 {
    let count = 0;

    for (let i = 0; i < binIndexes.length; i++) {
      count += this.editingFeatureSampleMap[binIndexes[i]].length;
    }

    return count;
  }

  /**
   * Count the hist bin counts for each feature for each affected samples.
   * @param binIndexes Bin indexes of a interested region
   */
  getSelectedSampleDist(binIndexes: Array<i32>): Array<Array<i32>> {
    // Initialize bin counts
    let binCounts = new Array<Array<i32>>(this.histBinEdges.length);
    for (let b = 0; b < this.histBinEdges.length; b++) {
      binCounts[b] = new Array<i32>(this.histBinEdges[b].length).fill(0);
    }

    // Iterate through all the affected samples
    for (let i = 0; i < binIndexes.length; i++) {
      let ids = this.editingFeatureSampleMap[binIndexes[i]];

      for (let ids_i = 0; ids_i < ids.length; ids_i++) {
        let s = ids[ids_i];

        // Iterate through all features
        for (let j = 0; j < this.histBinEdges.length; j++) {
          let curFeatureType = this.featureTypes[j];
          let curFeature = this.samples[s][j];
          let histBinIndex: i32;

          if (curFeatureType == 'continuous') {
            histBinIndex = searchSortedLowerIndex(this.histBinEdges[j], curFeature);
          } else {
            histBinIndex = this.histBinEdges[j].indexOf(curFeature);
            if (histBinIndex < 0) {
              // Unseen level during training => use 0 as score instead
              console.log(`[WASM] Unseen categorical level in histogram: ${s}, ${j}, ${curFeature}`);
              histBinIndex = 0
            }
          }

          // Update the count
          binCounts[j][histBinIndex] ++;
        }
      }
    }

    return binCounts;
  }

  updateModel(changedBinIndexes: Array<i32>, changedScores: Array<f64>, featureIndex: i32): void {
    // Update the bin scores
    let scoreDiffs = new Array<f64>(changedScores.length);

    for (let i = 0; i < changedBinIndexes.length; i++) {
      // Keep track the score difference for later faster prediction
      let b = changedBinIndexes[i];
      scoreDiffs[i] = changedScores[i] - this.scores[featureIndex][b];

      this.scores[featureIndex][b] = changedScores[i];
    }

    // Update the prediction
    this.updatePredictionPartial(changedBinIndexes, scoreDiffs, featureIndex);
  }

  updatePredictionPartial(changedBinIndexes: Array<i32>, scoreDiffs: Array<f64>, featureIndex: i32): void {
    // We know which bin has been changed and which samples are affected, so we
    // only need to update their predictions
    for (let i = 0; i < changedBinIndexes.length; i++) {
      let b = changedBinIndexes[i];
      let affectedSampleIDs = this.editingFeatureSampleMaps.get(featureIndex)[b];

      for (let j = 0; j < affectedSampleIDs.length; j++) {
        let s = affectedSampleIDs[j];
        this.predLabels[s] += scoreDiffs[i];

        // Update the prob if it is a classifier
        if (this.isClassification) {
          this.predProbs[s] = sigmoid(this.predLabels[s]);
        }
      }
    }
  }

  /**
   * Overwrite the bin definition and scores of the current editing feature.
   * This function assumes the feature has been set before (existing
   * editingFeatureSampleMaps entry, setEditingFeature() has been called)
   * @param newBinEdges New bin edges.
   * @param newScores New bin scores.
   * @param featureIndex Index of the feature to edit.
   */
  setModel(newBinEdges: Array<f64>, newScores: Array<f64>, featureIndex: i32): void {

    // Step 1: Remove the effect of this feature from logits (re-compute prob later)
    for (let b = 0; b < this.binEdges[featureIndex].length; b++) {
      let affectedSampleIDs = this.editingFeatureSampleMaps.get(featureIndex)[b];
      let curBinScore = this.scores[featureIndex][b];

      for (let i = 0; i < affectedSampleIDs.length; i++) {
        let s = affectedSampleIDs[i];
        this.predLabels[s] -= curBinScore;
      }
    }

    // Step 2: overwrite the bin edges and scores
    this.binEdges[featureIndex] = newBinEdges;
    this.scores[featureIndex] = newScores;

    // Step 3: Re-indexing the sample IDs & Add the new score to logits
    let curEditingFeatureSampleMap = new Array<Array<i32>>(newBinEdges.length);

    // Initialize the editing feature map
    for (let b = 0; b < newBinEdges.length; b++) {
      curEditingFeatureSampleMap[b] = new Array<i32>();
    }

    for (let s = 0; s < this.samples.length; s++) {
      let curFeature = this.samples[s][featureIndex];

      // Use the feature value to find the corresponding bin
      let binIndex: i32;
      let binScore: f64;

      if (this.featureTypes[featureIndex] == 'continuous') {
        binIndex = searchSortedLowerIndex(newBinEdges, curFeature);
        binScore = newScores[binIndex];
      } else {
        binIndex = newBinEdges.indexOf(curFeature);
        if (binIndex < 0) {
          // Unseen level during training => use 0 as score instead
          console.log(`>> Unseen feature: ${this.featureNames[featureIndex]}, ${s}, ${curFeature}`);
          binScore = 0
        } else {
          binScore = newScores[binIndex];
        }
      }

      // Add the new score to logits
      this.predLabels[s] += binScore;

      // Update the prob if it is a classifier
      if (this.isClassification) {
        this.predProbs[s] = sigmoid(this.predLabels[s]);
      }

      // Track the sample ID in the index
      // console.log([s, binIndex, curFeature]);
      if (binIndex >= 0) {
        curEditingFeatureSampleMap[binIndex].push(s);
      }
    }

    // Update the featureSampleMap in our record
    this.editingFeatureSampleMaps.set(featureIndex, curEditingFeatureSampleMap);
    this.editingFeatureSampleMap = this.editingFeatureSampleMaps.get(featureIndex);
  }

  getPrediction(): Array<f64> {
    if (this.isClassification) {
      return this.predProbs;
    } else {
      return this.predLabels;
    }
  }

  getMetrics(): Array<Array<Array<f64>>> {
    let output = new Array<Array<Array<f64>>>();

    if (!this.isClassification) {
      let curResult = new Array<f64>();
      curResult.push(rootMeanSquaredError(this.labels, this.predLabels));
      curResult.push(meanAbsoluteError(this.labels, this.predLabels));
      output.push([curResult]);
    } else {
      // Compute ROC curves
      let countResult = countByThreshold(this.labels, this.predProbs);
      let rocPoints = getROCCurve(countResult);
      // let prPoints = getPRCurve(countResult);

      output.push(rocPoints);
      // output.push(prPoints);

      // Compute confusion matrix
      let confusionMatrix = getConfusionMatrix(this.labels, this.predProbs);

      output.push([confusionMatrix]);

      // Compute summary statistics
      let rocAuc = getROCAuc(rocPoints);
      // let averagePrecision = getAveragePrecision(prPoints);
      let accuracy = getAccuracy(this.labels, this.predProbs);
      let balancedAccuracy = getBalancedAccuracy(confusionMatrix);

      output.push([[accuracy, rocAuc, balancedAccuracy]]);
    }

    return output;
  }

  getMetricsOnSelectedSamples(sampleIDs: Array<i32>): Array<Array<Array<f64>>> {
    let output = new Array<Array<Array<f64>>>();

    if (!this.isClassification) {
      // Filter the affected labels and their predictions
      let curResult = new Array<f64>();
      let affectedLabels = new Array<f64>(sampleIDs.length);
      let affectedPredLabels = new Array<f64>(sampleIDs.length);

      for (let s = 0; s < sampleIDs.length; s++) {
        affectedLabels[s] = this.labels[sampleIDs[s]];
        affectedPredLabels[s] = this.predLabels[sampleIDs[s]];
      }

      curResult.push(rootMeanSquaredError(affectedLabels, affectedPredLabels));
      curResult.push(meanAbsoluteError(affectedLabels, affectedPredLabels));
      output.push([curResult]);
    } else {
      // Filter the affected labels and their predictions
      let affectedLabels = new Array<f64>(sampleIDs.length);
      let affectedPredLProbs = new Array<f64>(sampleIDs.length);

      for (let s = 0; s < sampleIDs.length; s++) {
        affectedLabels[s] = this.labels[sampleIDs[s]];
        affectedPredLProbs[s] = this.predProbs[sampleIDs[s]];
      }

      // Compute ROC curves
      let countResult = countByThreshold(affectedLabels, affectedPredLProbs);
      let rocPoints = getROCCurve(countResult);

      output.push(rocPoints);

      // Compute confusion matrix
      let confusionMatrix = getConfusionMatrix(affectedLabels, affectedPredLProbs);

      output.push([confusionMatrix]);

      // Compute summary statistics
      let rocAuc = getROCAuc(rocPoints);
      let accuracy = getAccuracy(affectedLabels, affectedPredLProbs);
      let balancedAccuracy = getBalancedAccuracy(confusionMatrix);

      output.push([[accuracy, rocAuc, balancedAccuracy]]);
    }

    return output;
  }

  /**
   * Compute the performance metrics on the affected samples of the selected bins.
   * @param binIndexes Bin indexes of a interested region
   */
  getMetricsOnSelectedBins(binIndexes: Array<i32>): Array<Array<Array<f64>>> {

    // Get the affected sample IDs
    let affectedIDs = new Array<i32>();
    for (let i = 0; i < binIndexes.length; i++) {
      let ids = this.editingFeatureSampleMap[binIndexes[i]];

      for (let ids_i = 0; ids_i < ids.length; ids_i++) {
        affectedIDs.push(ids[ids_i]);
      }
    }

    return this.getMetricsOnSelectedSamples(affectedIDs);
  }

  /**
   * Compute the performance metrics on the affected slice of samples.
   * This function assumes this.sliceSampleIDs has been set correctly.
   */
  getMetricsOnSelectedSlice(): Array<Array<Array<f64>>> {
    return this.getMetricsOnSelectedSamples(this.sliceSampleIDs);
  }

  /**
   * Set this.sliceSampleIDs with the selected categorical level.
   * this.sliceSampleIDs tracks all test samples have the given categorical level.
   * @param featureID Index of the interested categorical variable
   * @param featureLevel Integer encoding for the interested categorical level (value)
   */
  setSliceData(featureID: i32, featureLevel: i32): i32 {
    if (this.featureTypes[featureID] == 'continuous') {
      trace('[WASM] Cannot slice continuous variable ' + this.featureNames[featureID]);
      return -1;
    }

    this.sliceSampleIDs = [];

    // Iterate through all the test samples to collect the ones having the level
    for (let i = 0; i < this.samples.length; i++) {
      let curFeature = this.samples[i][featureID];
      if (curFeature == featureLevel) {
        this.sliceSampleIDs.push(i);
      }
    }

    return this.sliceSampleIDs.length;
  }

  /**
   * Change the current editing feature. If this feature has not been edited before,
   * this function creates a new editingFeatureSampleMap for it
   * @param featureID Index of the interested categorical variable
   */
  setEditingFeature(featureID: i32): void {

    this.editingFeatureIndex = featureID;

    // Creates a new feature sample map
    if (!this.editingFeatureSampleMaps.has(featureID)) {
      this.editingFeatureSampleMap = new Array<Array<i32>>(this.binEdges[this.editingFeatureIndex].length);

      // Initialize the editing feature map
      for (let b = 0; b < this.binEdges[this.editingFeatureIndex].length; b++) {
        this.editingFeatureSampleMap[b] = new Array<i32>();
      }

      // Populate the map
      for (let i = 0; i < this.samples.length; i++) {

        let curFeatureType = this.featureTypes[featureID];
        let curFeature = this.samples[i][featureID];

        // Use the feature value to find the corresponding bin
        let binIndex: i32;

        if (curFeatureType == 'continuous') {
          binIndex = searchSortedLowerIndex(this.binEdges[featureID], curFeature);
        } else {
          binIndex = this.binEdges[featureID].indexOf(curFeature);
        }

        if (binIndex >= 0) {
          this.editingFeatureSampleMap[binIndex].push(i);
        }
      }

      // Keep track the map
      this.editingFeatureSampleMaps.set(featureID, this.editingFeatureSampleMap);
    }

    this.editingFeatureSampleMap = this.editingFeatureSampleMaps.get(featureID);
  }

  // Use this function to test assembly script stuff
  printName(): string {
    trace('editing', 1, this.editingFeatureIndex);

    let test3dArray = new Array<Array<Array<i32>>>();
    let test2dArray = new Array<Array<i32>>();
    test3dArray.push(test2dArray);

    let test1dArray = [1, 2, 3];
    test2dArray.push(test1dArray);

    console.log(test2dArray[0]);

    let temp = test2dArray[0];
    temp[0] = 10;
    console.log(test2dArray[0]);

    temp = [0, 1];
    temp[0] = 20;
    console.log(test2dArray[0]);


    return this.featureTypes[this.editingFeatureIndex];
  };

}

// Export the metrics functions to JS for testing
export { rootMeanSquaredError, meanAbsoluteError, countByThreshold, getROCCurve,
  getPRCurve, getROCAuc, getAveragePrecision, getAccuracy, getConfusionMatrix,
  getBalancedAccuracy
};

// We need unique array id so we can allocate them in JS
export const Int32Array_ID = idof<Array<i32>>();
export const Int32Array2D_ID = idof<Array<Array<i32>>>();

export const Float64Array_ID = idof<Array<f64>>();
export const Float64Array2D_ID = idof<Array<Array<f64>>>();
export const Float64Array3D_ID = idof<Array<Array<Array<f64>>>>();

export const StringArray_ID = idof<Array<string>>();
export const StringArray2D_ID = idof<Array<Array<string>>>();
