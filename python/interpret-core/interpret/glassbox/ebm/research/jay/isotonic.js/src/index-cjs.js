const fs = require('fs');

const loader = require('@assemblyscript/loader');

const imports = { /* imports go here */ };
const wasmModule = loader.instantiateSync(
  // Use untouched.wasm for development, use optimized.wasm for distribution/external use
  fs.readFileSync('build/untouched.wasm'),
  imports
);

// Import the helper functions
const wasm = wasmModule.exports;
const __pin = wasm.__pin;
const __unpin = wasm.__unpin;
const __newArray = wasm.__newArray;
const __getArray = wasm.__getArray;

/**
 * JS wrapper for the lexsort() from WASM, which sorts x, y, w with the same order
 * and uses x as the first sorting key and y as the secondary sorting key.
 * This function is only used to expose lexsort() for unit testing.
 * @param {[number]} x x array
 * @param {[number]} y y array
 * @param {[number]} w weight array
 * @param {bool} increasing sort to increasing order
 * @returns sorted [x, y, w] array
 */
const __lexsort = (x, y, w, increasing) => {
  // Create a new array in JS
  let xPtr = __pin(__newArray(wasm.xArrayID, x));
  let yPtr = __pin(__newArray(wasm.yArrayID, y));
  let wPtr = __pin(__newArray(wasm.wArrayID, w));

  // Lexsort using WASM
  wasm.lexsort(xPtr, yPtr, wPtr, increasing);

  // Convert pointer to JS arrays
  let xArray = __getArray(xPtr);
  let yArray = __getArray(yPtr);
  let wArray = __getArray(wPtr);

  // Unpin the pointers so they can get collected
  __unpin(xPtr);
  __unpin(yPtr);
  __unpin(wPtr);

  // console.log('x', xArray, 'y', yArray, 'w', wArray);
  return [xArray, yArray, wArray];
};

/**
 * JS wrapper for the makeUnique() from WASM, which drops the duplicate x,
 * replace their y's with weighted average, and replace their w's with weight
 * sum. This function assumes that x is sorted. This function is only used to
 * expose lexsort() for unit testing.
 * @param {[number]} x x array
 * @param {[number]} y y array
 * @param {[number]} w weight array
 * @returns [unique x array, unique y array, unique weight array]
 */
const __makeUnique = (x, y, w) => {
  // Create a new array in JS
  let xPtr = __pin(__newArray(wasm.xArrayID, x));
  let yPtr = __pin(__newArray(wasm.yArrayID, y));
  let wPtr = __pin(__newArray(wasm.wArrayID, w));

  let resultPtr = __pin(wasm.makeUnique(xPtr, yPtr, wPtr));

  // Resolve the 2D pointers
  let result = __getArray(resultPtr);
  result = result.map((d) => __getArray(d));

  // Unpin the pointers so they can get collected
  __unpin(xPtr);
  __unpin(yPtr);
  __unpin(wPtr);
  __unpin(resultPtr);

  return result;
};

/**
 * JS wrapper for the inplaceIsotonicY() from WASM, which Fit the isotonic
 * regression on y with weight using the Pool Adjacent Violators Algorithm
 * (PAVA). Internally the array is updated in-place, but the JS y array is
 * not changed.
 * @param {[number]} y y array
 * @param {[number]} w weight array
 * @returns Fitted y array
 */
const __inplaceIsotonicY = (y, w) => {
  let yPtr = __pin(__newArray(wasm.yArrayID, y));
  let wPtr = __pin(__newArray(wasm.wArrayID, w));

  wasm.inplaceIsotonicY(yPtr, wPtr);

  let yArray = __getArray(yPtr);

  // Unpin the pointers so they can get collected
  __unpin(yPtr);
  __unpin(wPtr);

  return yArray;
};

/**
 * JS wrapper for the searchsorted() from WASM, which finds the index where
 * inserting `value` into `sorted` would keep `sorted` in order.
 * @param {[number]} sorted a sorted array (ascending order)
 * @param {number} value a number to insert into `sorted`
 * @returns index to insert `value` ito `sorted`
 */
const __searchsorted = (sorted, value) => {
  let xPtr = __pin(__newArray(wasm.xArrayID, sorted));
  let index = wasm.searchsorted(xPtr, value);
  __unpin(xPtr);
  return index;
};

class IsotonicRegression {
  // Store an instance of WASM IsotonicRegression
  iso;

  /**
   * Constructor for the class IsotonicRegression
   * @param {object} param0 Model configuration. It can have 4 fields:
   * 1. yMin {number} minimum value of y, default = -Infinity
   * 2. yMax {number} maximum value of y, default = Infinity
   * 3. increasing {bool} if true, fit an increasing isotonic regression, default = true
   * 4. clipOutOfBound {bool} if true, clip the out of bound x; otherwise predict null, default = true
   */
  constructor({yMin = -Infinity, yMax = Infinity, increasing = true, clipOutOfBound = true} = {}) {
    this.iso = new wasm.__IsotonicRegression(yMin, yMax, increasing, clipOutOfBound);

    // Important to pin any WASM object created in JS
    // Since the runtime on the Wasm end does not know that a JS object keeps
    // the Wasm object alive
    __pin(this.iso);
  }

  /**
   * Fit an isotonic regression on the given x, y, w data.
   * @param {[number]} x x array
   * @param {[number]} y y array
   * @param {[number]} w weight array
   */
  fit(x, y, w = undefined) {
    // If sample weight is not given, replace them with [1, 1 ... 1]
    let sampleWeight = w;
    if (w === undefined) {
      sampleWeight = new Array(x.length).fill(1.0);
    }

    // Check the parameters
    this.__checkFitParam(x, y, sampleWeight);

    // Create arrays in WASM memory
    let xPtr = __pin(__newArray(wasm.xArrayID, x));
    let yPtr = __pin(__newArray(wasm.yArrayID, y));
    let wPtr = __pin(__newArray(wasm.wArrayID, sampleWeight));

    // Fit the Isotonic regression using WASM
    this.iso.fit(xPtr, yPtr, wPtr);

    // Unpin the pointers so they can get collected
    __unpin(xPtr);
    __unpin(yPtr);
    __unpin(wPtr);
  }


  /**
   * Use the trained isotonic regression model to predict on the new data
   * @param {[number]} newX new data array
   * @returns predictions, same size as `newX`
   */
  predict(newX) {
    // Pass newX to WASM to predict
    let newXPtr = __pin(__newArray(wasm.newXArrayID, newX));
    let predictedXPtr = this.iso.predict(newXPtr);
    let predictedXArray = __getArray(predictedXPtr);

    __unpin(newXPtr);
    return predictedXArray;
  }

  /**
   * Reset the learned weights of this isotonic regression model.
   */
  reset() {
    this.iso.reset();
  }

  /**
   * Run this function when the model is no longer needed. It is necessary because
   * WASM won't garbage collect the model until we manually __unpin() it from JS
   * (memory leak)
   */
  destroy() {
    __unpin(this.iso);
  }

  get xThresholds() {
    return __getArray(this.iso.xThresholds);
  }

  get yThresholds() {
    return __getArray(this.iso.yThresholds);
  }

  get xMin() {
    return this.iso.xMin;
  }

  get xMax() {
    return this.iso.xMax;
  }

  __checkFitParam(x, y, w) {
    if (x.length <= 1 || y.length <= 1 || w.length <= 1) {
      throw new Error('The length of input arrays should be greater than 1.');
    }

    if (x.length !== y.length) {
      throw new Error('The x array and y array should have the same length.');
    }
  }
}

module.exports = wasmModule.exports;

// Add new functions
module.exports.__lexsort = __lexsort;
module.exports.__makeUnique = __makeUnique;
module.exports.__inplaceIsotonicY = __inplaceIsotonicY;
module.exports.__searchsorted = __searchsorted;

// Overwrite the WASM IsotonicRegression with JS IsotonicRegression wrapper
module.exports.IsotonicRegression = IsotonicRegression;

