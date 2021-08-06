/* eslint-disable eqeqeq */

/**
 * Common ML metrics implemented in WASM.
 *
 * Author: Jay Wang (jayw@gatech.edu)
 */

import { console } from 'as-console';

/**
 * Compute the root mean squared error (RMSE)
 * @param yTrue True values
 * @param yPred Predicted values
 * @returns score
 */
export function rootMeanSquaredError(yTrue: Array<f64>, yPred: Array<f64>): f64 {
  let SESum = 0.0;
  for (let i = 0; i < yTrue.length; i++) {
    SESum += (yPred[i] - yTrue[i]) ** 2;
  }
  return Math.sqrt(SESum / yTrue.length);
}

/**
 * Compute the root mean squared error (RMSE)
 * @param yTrue True values
 * @param yPred Predicted values
 * @returns score
 */
export function meanAbsoluteError(yTrue: Array<f64>, yPred: Array<f64>): f64 {
  let AESum = 0.0;
  for (let i = 0; i < yTrue.length; i++) {
    AESum += Math.abs(yTrue[i] - yPred[i]);
  }
  return AESum / yTrue.length;
}

/**
 * Compute the threshold along with true/false positives for the given prediction
 * scores and true labels.
 *
 * @param {[int]} yTrue Array of true labels. 1 for positive and 0 for negative
 * @param {[float]} yScore Array of prediction scores.
 */
export function countByThreshold(yTrue: Array<f64>, yScore: Array<f64>): Array<Array<f64>> {
  let thresholds = new Array<f64>();
  let fps = new Array<f64>();
  let tps = new Array<f64>();

  // Sort yTrue and yScore together, by sorting yScore decreasingly
  let yMerged = new Array<Array<f64>>();
  let sortedYTrue = new Array<f64>();
  let sortedYScore = new Array<f64>();

  for (let i = 0; i < yTrue.length; i++) {
    yMerged.push([yTrue[i], yScore[i]]);
  }

  yMerged.sort((a, b) => {
    if (b[1] - a[1] > 0) {
      return 1;
    } else if (b[1] - a[1] < 0) {
      return -1;
    } else {
      return 0;
    }
  });

  for (let i = 0; i < yMerged.length; i++) {
    sortedYTrue.push(yMerged[i][0]);
    sortedYScore.push(yMerged[i][1]);
  }

  // Construct the threshold list. Thresholds are the unique score values with a
  // descending order.
  thresholds.push(sortedYScore[0]);
  for (let i = 1; i < sortedYScore.length; i++) {
    if (thresholds[thresholds.length - 1] !== sortedYScore[i]) {
      thresholds.push(sortedYScore[i]);
    }
  }

  // Count FP and TP based on different thresholds. Since thresholds are sorted
  // descending, the score passing current threshold will definitely pass the
  // next threshold. Therefore, FP and TP are actually cumulative.
  // My algorithm can be further optimized.
  let cumFP = 0;
  let cumTP = 0;

  for (let ti = 0; ti < thresholds.length; ti++) {
    let t = thresholds[ti];
    let i = 0;
    for (i; i < sortedYScore.length; i++) {
      // Get the prediction
      let curPredict = sortedYScore[i] >= t ? 1 : 0;

      // If the prediction is N, then we break, because we are not counting
      // FN or TN, and the scores are sorted
      if (curPredict === 0) {
        break;
      }

      // Update FP, TP count for this threshold
      if (curPredict === sortedYTrue[i]) {
        cumTP += 1;
      } else if (curPredict !== sortedYTrue[i]) {
        cumFP += 1;
      }
    }
    // Store FP and TP for this threshold
    fps.push(cumFP);
    tps.push(cumTP);

    // Slice the score array so no need to revisit larger values again
    sortedYScore = sortedYScore.slice(i, sortedYScore.length);
    sortedYTrue = sortedYTrue.slice(i, sortedYTrue.length);
  }

  return [fps, tps, thresholds];
}

/**
 * Function to get ROC Curve points.
 *
 * @returns array of points (point[0] is tpr, point[1] is fpr)
 */
export function getROCCurve(thresholdResult: Array<Array<f64>>): Array<Array<f64>> {
  // Get threshold along with TPs and FPs
  let fps = thresholdResult[0];
  let tps = thresholdResult[1];

  // Could drop thresholds where corresponding point is on the connected
  // line of the previous and next points
  let optimalFPs = [fps[0]];
  let optimalTPs = [tps[0]];

  for (let i = 0; i < tps.length - 2; i++) {
    let fpDiff1 = fps[i + 2] - fps[i + 1];
    let fpDiff2 = fps[i + 1] - fps[i];
    let fpDiff = fpDiff1 - fpDiff2;

    let tpDiff1 = tps[i + 2] - tps[i + 1];
    let tpDiff2 = tps[i + 1] - tps[i];
    let tpDiff = tpDiff1 - tpDiff2;

    if (tpDiff != 0 || fpDiff != 0) {
      optimalFPs.push(fps[i + 1]);
      optimalTPs.push(tps[i + 1]);
    }
  }

  optimalFPs.push(fps[fps.length - 1]);
  optimalTPs.push(tps[tps.length - 1]);

  // Remember the last entry of TPs is the total number of positive labels, and
  // the last entry of FPs is the total number of negative labels
  let totalN = optimalFPs[optimalFPs.length - 1];
  let totalP = optimalTPs[optimalTPs.length - 1];

  // Add (0, 0) as the starting point
  let rocPoints = [[0.0, 0.0]];

  for (let i = 0; i < optimalTPs.length; i++) {
    rocPoints.push(
      [
        optimalTPs[i] / totalP,
        optimalFPs[i] / totalN
      ]
    );
  }

  return rocPoints;
}

/**
 * Function to get precision-recall Curve points.
 *
 * @param thresholdResult Results from counting the FPs and TPs
 *
 * @returns array of points ([0] is precision, [1] is recall)
 */
export function getPRCurve(thresholdResult: Array<Array<f64>>): Array<Array<f64>> {
  // Get threshold along with TPs and FPs
  let fps = thresholdResult[0];
  let tps = thresholdResult[1];

  // Remember the last entry of TPs is the total number of positive labels.
  let totalP = tps[tps.length - 1];

  // Add (1, 0) as the starting point
  let prPoints = [[1.0, 0.0]];

  for (let i = 0; i < tps.length; i++) {
    prPoints.push(
      [
        tps[i] / (tps[i] + fps[i]),
        tps[i] / totalP
      ]
    );

    // Stop when full recall is achieved
    if (tps[i] / totalP == 1.0) {
      break;
    }
  }

  return prPoints;
}

/**
 * Compute area under the ROC curve. Only using trapezoidal
 * approximation (no interested region correction).
 *
 * @param rocPoints Points on the ROC curve
 */
export function getROCAuc(rocPoints: Array<Array<f64>>): f64 {
  // Reverse the array, so x is decreasing
  let rocPointsReverse = rocPoints.slice(0).reverse();

  // Use Trapezoidal rule to approximate area under the curve
  let trapeArea = 0.0;

  for (let i = 0; i < rocPointsReverse.length - 1; i++) {
    trapeArea += (rocPointsReverse[i][1] - rocPointsReverse[i + 1][1]) *
      (rocPointsReverse[i][0] + rocPointsReverse[i + 1][0]) / 2.0;
  }

  return trapeArea;
}

/**
 * Function to compute average precision.
 *
 * @param prPoints Points on the precision recall curve
 */
export function getAveragePrecision(prPoints: Array<Array<f64>>): f64 {
  // Compute average precision
  // AP = ∑_n (R_n - R_{n-1}) * P_n
  let averagePrecision = 0.0;

  // Reverse the array, so recall is decreasing
  let prPointsReverse = prPoints.slice(0).reverse();
  for (let i = 0; i < prPointsReverse.length - 1; i++) {
    averagePrecision += (prPointsReverse[i][1] - prPointsReverse[i + 1][1]) *
      prPointsReverse[i][0];
  }
  return averagePrecision;
}

/**
 * Function to compute the accuracy score.
 * @param yTrue True labels
 * @param yProb Predicted scores
 * @returns Accuracy score
 */
export function getAccuracy(yTrue: Array<f64>, yProb: Array<f64>): f64 {
  let correctNum = 0.0;
  for (let i = 0; i < yTrue.length; i++) {
    let curLabel = yProb[i] >= 0.5 ? 1.0 : 0.0;
    if (curLabel == yTrue[i]) {
      correctNum++;
    }
  }
  return correctNum / yTrue.length;
}

/**
 * Function to compute the confusion matrix.
 * @param yTrue True labels
 * @param yProb Predicted scores
 * @param threshold Threshold to determine the positive prediction (default 0.5)
 * @returns [tn, fp, fn, tp]
 */
export function getConfusionMatrix(yTrue: Array<f64>, yProb: Array<f64>, threshold: f64 = 0.5): Array<f64> {
  let confusionMatrix = [0.0, 0.0, 0.0, 0.0];

  for (let i = 0; i < yTrue.length; i++) {
    let curLabel = yProb[i] >= threshold ? 1.0 : 0.0;
    if (curLabel == 1) {
      if (curLabel == yTrue[i]) {
        confusionMatrix[3]++;
      } else {
        confusionMatrix[1]++;
      }
    } else if (curLabel == yTrue[i]) {
      confusionMatrix[0]++;
    } else {
      confusionMatrix[2]++;
    }
  }

  return confusionMatrix;
}

/**
 * Function to compute the balanced accuracy score (average of recalls on all classes)
 * @param confusionMatrix [tn, fp, fn, tp]
 * @returns balanced accuracy score
 */
export function getBalancedAccuracy(confusionMatrix: Array<f64>): f64 {

  // Compute the recall of positive case
  let recallP = confusionMatrix[3] / (confusionMatrix[3] + confusionMatrix[2]);

  // Compute the recall of negative case
  let recallN = confusionMatrix[0] / (confusionMatrix[0] + confusionMatrix[1]);

  return (recallP + recallN) / 2;
}
