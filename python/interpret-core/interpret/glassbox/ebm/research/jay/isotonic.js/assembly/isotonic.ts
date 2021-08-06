/**
 * The entry file of the WebAssembly module.
 * 
 * This file implements the Pool Adjacent Violators Algorithm (PAVA) to solve
 * isotonic regression. It is a port of the scikit-learn's implementation.
 * 
 * Author: Jay Wang (jayw@gatech.edu)
 */

/**
 * Sort x, y, w in place with the same order, and uses x as the first sorting key
 * and y as the secondary sorting key.
 * @param x x array
 * @param y y array
 * @param w weight array
 * @param increasing if sorting (x, y, z) with an increasing order
 */
export function lexsort(x: Array<f64>, y: Array<f64>, w: Array<f64>, increasing: bool=true) :void {
  // Bundle three arrays into one before sorting
  let combinedArray = new Array<Array<f64>>(x.length);

  for (let i = 0; i < x.length; i++) {
    combinedArray[i] = [x[i], y[i], w[i]];
  }

  // The builtin sort() function is not stable, so we need to manually sort on
  // the secondary key to break ties
  // There is no closure in AS (we have no access to increasing arg in the
  // comparison function, so we need to use if statement to create two sorts)
  if (increasing) {
    combinedArray.sort((a: Array<f64>, b: Array<f64>) => {
      if (a[0] < b[0]) {
        return -1;
      } else if (a[0] > b[0]) {
        return 1;
      } else {
        // Breaking tie using the secondary key (y)
        return a[1] - b[1] as i32;
      }
    });
  } else {
    combinedArray.sort((a: Array<f64>, b: Array<f64>) => {
      if (a[0] < b[0]) {
        return 1;
      } else if (a[0] > b[0]) {
        return -1;
      } else {
        // Breaking tie using the secondary key (y)
        return b[1] - a[1] as i32;
      }
    });
  }

  // Update the values in x, y, and w
  for (let i = 0; i < x.length; i++) {
    x[i] = combinedArray[i][0];
    y[i] = combinedArray[i][1];
    w[i] = combinedArray[i][2];
  }
}

/**
 * Drop the duplicate x, replace their y's with weighted average, and replace
 * their w's with weight sum. This function assumes that x is sorted.
 * @param x x array
 * @param y y array
 * @param w weight array
 * @returns [unique x array, unique y array, unique weight array]
 */
export function makeUnique(x: Array<f64>, y: Array<f64>, w: Array<f64>): Array<Array<f64>> {

  // Count the unique values in x
  let xUniqueSet = new Set<f64>();
  for (let i = 0; i < x.length; i++) {
    xUniqueSet.add(x[i]);
  }

  // Create output arrays
  let xOut = new Array<f64>(xUniqueSet.size);
  let yOut = new Array<f64>(xUniqueSet.size);
  let wOut = new Array<f64>(xUniqueSet.size);

  // Iterate through the x, y, z arrays and compute weighted average for the y's
  // of duplicating x's
  let curX = x[0];
  let curY :f64 = 0;
  let curW: f64 = 0;

  const eps = 1e-6;
  let i = 0;

  for (let j = 0; j < x.length; j++) {
    let xj = x[j];

    if (Math.abs(xj - curX) >= eps) {
      // xj is different from curX, we take average of the accumulated y's
      xOut[i] = curX;
      yOut[i] = curY / curW;
      wOut[i] = curW;
      i ++;

      // Move to the new unique value, init y and w
      curX = xj;
      curY = y[j] * w[j];
      curW = w[j];
    } else {
      // xj is the same as curX, we accumulate the weighted y value
      curY += y[j] * w[j];
      curW += w[j];
    }
  }

  // Add the last values
  xOut[i] = curX;
  yOut[i] = curY / curW;
  wOut[i] = curW;

  assert(xUniqueSet.size == i + 1);

  // let output = new Array<Array<f64>>(3);
  // for (let i = 0; i < 3; i++) {
  //   output[i] = new Array<f64>(xOut.length);
  //   for (let j = 0; j < xOut.length; j++) {
  //     if (i == 0) {
  //       output[i][j] = xOut[j];
  //     } else if (i == 1) {
  //       output[i][j] = yOut[j];
  //     } else if (i == 2) {
  //       output[i][j] = wOut[j];
  //     }
  //   }
  // }

  return [xOut, yOut, wOut];
};

/**
 * Fit the isotonic regression on y with weight using the Pool Adjacent
 * Violators Algorithm (PAVA).
 * This function changes the y's value in-place.
 * @param y y array
 * @param w weight array
 */
export function inplaceIsotonicY(y: Array<f64>, w: Array<f64>): void {

  // `target` is an array of indexes pointing to the end/start of the decreasing
  // sub-range (to right/left).
  // The sub-range grows from left to right, right-pointing index points to the
  // longest decreasing range, and left-pointing index means an end of a smaller
  // decreasing range
  // For each small decreasing range, there is one y value (computed by the weighted
  // average of y's in that range) and one w value (sum of w's in that range)
  // We init this array with the index at each entry
  let target = new Array<i32>(y.length).map<i32>((d, i) => i as i32);

  // Left to right iteration to find decreasing sub-ranges
  let i = 0;
  while (i < y.length) {
    // Compare the left pointer and right pointer
    let k = target[i] + 1;
    if (k == y.length) {
      break;
    }
    
    let yLeft = y[i];
    let yRight = y[k];

    // End of the current range, move left pointer forward
    if (yLeft < yRight) {
      i = k;
      continue;
    }

    // yRight is still smaller than yLeft
    // Re-set the accumulating y and w with yLeft
    let ySum = yLeft * w[i];
    let wSum = w[i];

    // Move the right pointer forward until it no longer decreases
    while (true) {
      let yPreRight = y[k];
      ySum += yPreRight * w[k];
      wSum += w[k];

      k = target[k] + 1;

      // Repeat until the current k points to a non-decreasing value
      if (k == y.length || y[k] >= yPreRight) {
        // End of the current sub-range
        // Resolve the y value and w value for this sub-range, and store them
        // at the left pointer
        y[i] = ySum / wSum;
        w[i] = wSum;

        // Update the target array to mark this sub-range (left anr right)
        target[i] = k - 1;    // Right pointing at the start
        target[k - 1] = i;    // Left pointing at the end

        // Back track to the last left pointer if possible
        if (i > 0) {
          i = target[i - 1];
        }
        break;
      }
    }
  }

  // Fill the values between left and right pointers with y values at the
  // left pointer (create step functions)
  i = 0;
  while (i < y.length) {
    // Find the current right pointer, impute values in between
    for (let j = i + 1; j < target[i] + 1; j++) {
      y[j] = y[i];
    }
    i = target[i] + 1;
  }

};

/**
 * Find the index where inserting `value` into `sorted` would keep `sorted` in order.
 * @param sorted a sorted array (ascending order)
 * @param value a number to insert into `sorted`
 * @returns index to insert `value` ito `sorted`
 */
export function searchsorted(sorted: Array<f64>, value: f64): i32 {
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
  if (value > sorted[right]) {
    return right + 1;
  }
  if (value < sorted[left]) {
    return left;
  }
  return right;
}

export class __IsotonicRegression {
  yMin: f64;
  yMax: f64;
  xMin: f64;
  xMax: f64;

  increasing: bool;
  clipOutOfBound: bool;

  xThresholds: Array<f64>;
  yThresholds: Array<f64>;

  buildY: Array<f64>;
  buildF: (x: Array<f64>) => Array<f64>;

  /**
   * Constructor for the class IsotonicRegression
   * @param yMin minimum value of y
   * @param yMax maximum value of y
   * @param increasing if true, fit an increasing isotonic regression
   * @param clipOutOfBound if true, clip the out of bound x; otherwise predict null
   */
  constructor(yMin: f64, yMax: f64, increasing: bool, clipOutOfBound: bool) {
    this.yMin = yMin;
    this.yMax = yMax;
    this.increasing = increasing;
    this.clipOutOfBound = clipOutOfBound;

    // Have to initialize all properties
    this.xThresholds = [];
    this.yThresholds = [];
    this.buildY = [];
    this.buildF = (x: Array<f64>) => x;
    this.xMin = Infinity;
    this.xMax = -Infinity;
  };

  /**
   * Fit an isotonic regression.
   * @param x x array
   * @param y y array
   * @param w weight array
   */
  fit(x: Array<f64>, y: Array<f64>, w: Array<f64>): void {
    // Sort the x, y, w arrays by x and y
    lexsort(x, y, w);

    // Deduplicate x by computing weighted average of y's
    let result = makeUnique(x, y, w);
    let uniqueX = result[0];
    let uniqueY = result[1];
    let uniqueW = result[2];

    // Reverse y and w if we are fitting a decreasing isotonic regression
    if (!this.increasing) {
      uniqueY.reverse();
      uniqueW.reverse();
    }

    // Fit isotonic regression on y and w
    inplaceIsotonicY(uniqueY, uniqueW);

    // Clip the fitted y by [yMin, yMax]
    for (let i = 0; i < uniqueY.length; i++) {
      if (uniqueY[i] < this.yMin) {
        uniqueY[i] = this.yMin;
      }
      if (uniqueY[i] > this.yMax) {
        uniqueY[i] = this.yMax;
      }
    }

    // If user is fitting a decreasing isotonic regression, we need to flip back
    // the y and w arrays
    if (!this.increasing) {
      uniqueY.reverse();
    }

    // Store the bounds of x arrays
    for (let i = 0; i < uniqueX.length; i++) {
      if (uniqueX[i] > this.xMax) {
        this.xMax = uniqueX[i];
      }
      if (uniqueX[i] < this.xMin) {
        this.xMin = uniqueX[i];
      }
    }

    // Remove unnecessary points after fitting (y values that are equal to the
    // one before and the one after it, except the 1st and last y)
    let cleanedUniqueX = new Array<f64>();
    let cleanedUniqueY = new Array<f64>();

    cleanedUniqueX.push(uniqueX[0]);
    cleanedUniqueY.push(uniqueY[0]);

    const esp = 1e-6;
    
    for (let i = 1; i < uniqueY.length - 1; i++) {
      if (Math.abs(uniqueY[i] - uniqueY[i - 1]) > esp || Math.abs(uniqueY[i] - uniqueY[i + 1]) > esp) {
        cleanedUniqueX.push(uniqueX[i]);
        cleanedUniqueY.push(uniqueY[i]);
      }
    }

    cleanedUniqueX.push(uniqueX[uniqueX.length - 1]);
    cleanedUniqueY.push(uniqueY[uniqueY.length - 1]);

    // Store the fitted values
    this.xThresholds = cleanedUniqueX;
    this.yThresholds = cleanedUniqueY;
  };

  /**
   * Use the trained isotonic regression model to predict on the new data
   * @param newX new data array
   * @returns predictions, same size as `newX`
   */
  predict(newX: Array<f64>): Float64Array {

    let predictions = new Float64Array(newX.length);

    for (let i = 0; i < newX.length; i++) {
      // Find the corresponding ranges in xThresholds that newX should fall into
      let curIndex = searchsorted(this.xThresholds, newX[i]);

      // Clip the insert index if `clipOutOfBound`
      if (curIndex < 1) {
        if (this.clipOutOfBound) {
          // Use the min y threshold for out of bound x
          predictions[i] = this.yThresholds[0];
        } else {
          predictions[i] = NaN;
        }
        continue;
      }

      if (curIndex > this.xThresholds.length - 1) {
        if (this.clipOutOfBound) {
          // Use the max y threshold for out of bound x
          predictions[i] = this.yThresholds[this.yThresholds.length - 1];
        } else {
          predictions[i] = NaN;
        }
        continue;
      }

      // Calculate the prediction using linear interpolation between thresholds
      let xLow = this.xThresholds[curIndex - 1];
      let xHigh = this.xThresholds[curIndex];
      let yLow = this.yThresholds[curIndex - 1];
      let yHigh = this.yThresholds[curIndex];

      let slope = (yHigh - yLow) / (xHigh - xLow);

      predictions[i] = yLow + slope * (newX[i] - xLow);
    }

    return predictions;
  };

  /**
   * Reinitialize this model
   */
  reset(): void {
    this.xThresholds = [];
    this.yThresholds = [];
    this.buildY = [];
    this.buildF = (x: Array<f64>) => x;
    this.xMin = Infinity;
    this.xMax = -Infinity;
  };
}

// We need unique array id so we can allocate them in JS
export const xArrayID = idof<Array<f64>>();
export const yArrayID = idof<Array<f64>>();
export const wArrayID = idof<Array<f64>>();
export const newXArrayID = idof<Array<f64>>();
