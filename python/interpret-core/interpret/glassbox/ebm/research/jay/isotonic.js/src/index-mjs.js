import loader from '@assemblyscript/loader';
// import { sample } from '../build/optimized.wasm';

// export const testWASM = () => {
//   sample({}).then((result) => {
//     console.log(result);
//   });
// };

export const initIsotonicRegression = () => {
  return loader.instantiate(
    fetch('/build/optimized.wasm').then((result) => result.arrayBuffer()),
    {}
  ).then(({ exports }) => {
    const wasm = exports;
    const __pin = wasm.__pin;
    const __unpin = wasm.__unpin;
    const __newArray = wasm.__newArray;
    const __getArray = wasm.__getArray;

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
      constructor({ yMin = -Infinity, yMax = Infinity, increasing = true, clipOutOfBound = true } = {}) {
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

    let model = new IsotonicRegression();
    return model;

  });
};

