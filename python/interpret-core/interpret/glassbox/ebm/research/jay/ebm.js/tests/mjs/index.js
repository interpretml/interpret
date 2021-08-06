// Runtime header offsets
const ID_OFFSET = -8;
const SIZE_OFFSET = -4;

// Runtime ids
const ARRAYBUFFER_ID = 0;
const STRING_ID = 1;
// const ARRAYBUFFERVIEW_ID = 2;

// Runtime type information
const ARRAYBUFFERVIEW = 1 << 0;
const ARRAY = 1 << 1;
const STATICARRAY = 1 << 2;
// const SET = 1 << 3;
// const MAP = 1 << 4;
const VAL_ALIGN_OFFSET = 6;
// const VAL_ALIGN = 1 << VAL_ALIGN_OFFSET;
const VAL_SIGNED = 1 << 11;
const VAL_FLOAT = 1 << 12;
// const VAL_NULLABLE = 1 << 13;
const VAL_MANAGED = 1 << 14;
// const KEY_ALIGN_OFFSET = 15;
// const KEY_ALIGN = 1 << KEY_ALIGN_OFFSET;
// const KEY_SIGNED = 1 << 20;
// const KEY_FLOAT = 1 << 21;
// const KEY_NULLABLE = 1 << 22;
// const KEY_MANAGED = 1 << 23;

// Array(BufferView) layout
const ARRAYBUFFERVIEW_BUFFER_OFFSET = 0;
const ARRAYBUFFERVIEW_DATASTART_OFFSET = 4;
const ARRAYBUFFERVIEW_DATALENGTH_OFFSET = 8;
const ARRAYBUFFERVIEW_SIZE = 12;
const ARRAY_LENGTH_OFFSET = 12;
const ARRAY_SIZE = 16;

const BIGINT = typeof BigUint64Array !== "undefined";
const THIS = Symbol();

const STRING_SMALLSIZE = 192; // break-even point in V8
const STRING_CHUNKSIZE = 1024; // mitigate stack overflow
const utf16 = new TextDecoder("utf-16le", { fatal: true }); // != wtf16

/** Gets a string from memory. */
function getStringImpl(buffer, ptr) {
  let len = new Uint32Array(buffer)[ptr + SIZE_OFFSET >>> 2] >>> 1;
  const wtf16 = new Uint16Array(buffer, ptr, len);
  if (len <= STRING_SMALLSIZE) return String.fromCharCode(...wtf16);
  try {
    return utf16.decode(wtf16);
  } catch {
    let str = "", off = 0;
    while (len - off > STRING_CHUNKSIZE) {
      str += String.fromCharCode(...wtf16.subarray(off, off += STRING_CHUNKSIZE));
    }
    return str + String.fromCharCode(...wtf16.subarray(off));
  }
}

/** Prepares the base module prior to instantiation. */
function preInstantiate(imports) {
  const extendedExports = {};

  function getString(memory, ptr) {
    if (!memory) return "<yet unknown>";
    return getStringImpl(memory.buffer, ptr);
  }

  // add common imports used by stdlib for convenience
  const env = (imports.env = imports.env || {});
  env.abort = env.abort || function abort(msg, file, line, colm) {
    const memory = extendedExports.memory || env.memory; // prefer exported, otherwise try imported
    throw Error(`abort: ${getString(memory, msg)} at ${getString(memory, file)}:${line}:${colm}`);
  };
  env.trace = env.trace || function trace(msg, n, ...args) {
    const memory = extendedExports.memory || env.memory;
    console.log(`trace: ${getString(memory, msg)}${n ? " " : ""}${args.slice(0, n).join(", ")}`);
  };
  env.seed = env.seed || Date.now;
  imports.Math = imports.Math || Math;
  imports.Date = imports.Date || Date;

  return extendedExports;
}

const E_NOEXPORTRUNTIME = "Operation requires compiling with --exportRuntime";
const F_NOEXPORTRUNTIME = function() { throw Error(E_NOEXPORTRUNTIME); };

/** Prepares the final module once instantiation is complete. */
function postInstantiate(extendedExports, instance) {
  const exports = instance.exports;
  const memory = exports.memory;
  const table = exports.table;
  const __new = exports.__new || F_NOEXPORTRUNTIME;
  const __pin = exports.__pin || F_NOEXPORTRUNTIME;
  const __unpin = exports.__unpin || F_NOEXPORTRUNTIME;
  const __collect = exports.__collect || F_NOEXPORTRUNTIME;
  const __rtti_base = exports.__rtti_base;
  const getRttiCount = __rtti_base
    ? function (arr) { return arr[__rtti_base >>> 2]; }
    : F_NOEXPORTRUNTIME;

  extendedExports.__new = __new;
  extendedExports.__pin = __pin;
  extendedExports.__unpin = __unpin;
  extendedExports.__collect = __collect;

  /** Gets the runtime type info for the given id. */
  function getInfo(id) {
    const U32 = new Uint32Array(memory.buffer);
    const count = getRttiCount(U32);
    if ((id >>>= 0) >= count) throw Error(`invalid id: ${id}`);
    return U32[(__rtti_base + 4 >>> 2) + id * 2];
  }

  /** Gets and validate runtime type info for the given id for array like objects */
  function getArrayInfo(id) {
    const info = getInfo(id);
    if (!(info & (ARRAYBUFFERVIEW | ARRAY | STATICARRAY))) throw Error(`not an array: ${id}, flags=${info}`);
    return info;
  }

  /** Gets the runtime base id for the given id. */
  function getBase(id) {
    const U32 = new Uint32Array(memory.buffer);
    const count = getRttiCount(U32);
    if ((id >>>= 0) >= count) throw Error(`invalid id: ${id}`);
    return U32[(__rtti_base + 4 >>> 2) + id * 2 + 1];
  }

  /** Gets the runtime alignment of a collection's values. */
  function getValueAlign(info) {
    return 31 - Math.clz32((info >>> VAL_ALIGN_OFFSET) & 31); // -1 if none
  }

  /** Gets the runtime alignment of a collection's keys. */
  // function getKeyAlign(info) {
  //   return 31 - Math.clz32((info >>> KEY_ALIGN_OFFSET) & 31); // -1 if none
  // }

  /** Allocates a new string in the module's memory and returns its pointer. */
  function __newString(str) {
    if (str == null) return 0;
    const length = str.length;
    const ptr = __new(length << 1, STRING_ID);
    const U16 = new Uint16Array(memory.buffer);
    for (var i = 0, p = ptr >>> 1; i < length; ++i) U16[p + i] = str.charCodeAt(i);
    return ptr;
  }

  extendedExports.__newString = __newString;

  /** Reads a string from the module's memory by its pointer. */
  function __getString(ptr) {
    if (!ptr) return null;
    const buffer = memory.buffer;
    const id = new Uint32Array(buffer)[ptr + ID_OFFSET >>> 2];
    if (id !== STRING_ID) throw Error(`not a string: ${ptr}`);
    return getStringImpl(buffer, ptr);
  }

  extendedExports.__getString = __getString;

  /** Gets the view matching the specified alignment, signedness and floatness. */
  function getView(alignLog2, signed, float) {
    const buffer = memory.buffer;
    if (float) {
      switch (alignLog2) {
        case 2: return new Float32Array(buffer);
        case 3: return new Float64Array(buffer);
      }
    } else {
      switch (alignLog2) {
        case 0: return new (signed ? Int8Array : Uint8Array)(buffer);
        case 1: return new (signed ? Int16Array : Uint16Array)(buffer);
        case 2: return new (signed ? Int32Array : Uint32Array)(buffer);
        case 3: return new (signed ? BigInt64Array : BigUint64Array)(buffer);
      }
    }
    throw Error(`unsupported align: ${alignLog2}`);
  }

  /** Allocates a new array in the module's memory and returns its pointer. */
  function __newArray(id, values) {
    const info = getArrayInfo(id);
    const align = getValueAlign(info);
    const length = values.length;
    const buf = __new(length << align, info & STATICARRAY ? id : ARRAYBUFFER_ID);
    let result;
    if (info & STATICARRAY) {
      result = buf;
    } else {
      __pin(buf);
      const arr = __new(info & ARRAY ? ARRAY_SIZE : ARRAYBUFFERVIEW_SIZE, id);
      __unpin(buf);
      const U32 = new Uint32Array(memory.buffer);
      U32[arr + ARRAYBUFFERVIEW_BUFFER_OFFSET >>> 2] = buf;
      U32[arr + ARRAYBUFFERVIEW_DATASTART_OFFSET >>> 2] = buf;
      U32[arr + ARRAYBUFFERVIEW_DATALENGTH_OFFSET >>> 2] = length << align;
      if (info & ARRAY) U32[arr + ARRAY_LENGTH_OFFSET >>> 2] = length;
      result = arr;
    }
    const view = getView(align, info & VAL_SIGNED, info & VAL_FLOAT);
    if (info & VAL_MANAGED) {
      for (let i = 0; i < length; ++i) {
        const value = values[i];
        view[(buf >>> align) + i] = value;
      }
    } else {
      view.set(values, buf >>> align);
    }
    return result;
  }

  extendedExports.__newArray = __newArray;

  /** Gets a live view on an array's values in the module's memory. Infers the array type from RTTI. */
  function __getArrayView(arr) {
    const U32 = new Uint32Array(memory.buffer);
    const id = U32[arr + ID_OFFSET >>> 2];
    const info = getArrayInfo(id);
    const align = getValueAlign(info);
    let buf = info & STATICARRAY
      ? arr
      : U32[arr + ARRAYBUFFERVIEW_DATASTART_OFFSET >>> 2];
    const length = info & ARRAY
      ? U32[arr + ARRAY_LENGTH_OFFSET >>> 2]
      : U32[buf + SIZE_OFFSET >>> 2] >>> align;
    return getView(align, info & VAL_SIGNED, info & VAL_FLOAT).subarray(buf >>>= align, buf + length);
  }

  extendedExports.__getArrayView = __getArrayView;

  /** Copies an array's values from the module's memory. Infers the array type from RTTI. */
  function __getArray(arr) {
    const input = __getArrayView(arr);
    const len = input.length;
    const out = new Array(len);
    for (let i = 0; i < len; i++) out[i] = input[i];
    return out;
  }

  extendedExports.__getArray = __getArray;

  /** Copies an ArrayBuffer's value from the module's memory. */
  function __getArrayBuffer(ptr) {
    const buffer = memory.buffer;
    const length = new Uint32Array(buffer)[ptr + SIZE_OFFSET >>> 2];
    return buffer.slice(ptr, ptr + length);
  }

  extendedExports.__getArrayBuffer = __getArrayBuffer;

  /** Copies a typed array's values from the module's memory. */
  function getTypedArray(Type, alignLog2, ptr) {
    return new Type(getTypedArrayView(Type, alignLog2, ptr));
  }

  /** Gets a live view on a typed array's values in the module's memory. */
  function getTypedArrayView(Type, alignLog2, ptr) {
    const buffer = memory.buffer;
    const U32 = new Uint32Array(buffer);
    const bufPtr = U32[ptr + ARRAYBUFFERVIEW_DATASTART_OFFSET >>> 2];
    return new Type(buffer, bufPtr, U32[bufPtr + SIZE_OFFSET >>> 2] >>> alignLog2);
  }

  /** Attach a set of get TypedArray and View functions to the exports. */
  function attachTypedArrayFunctions(ctor, name, align) {
    extendedExports[`__get${name}`] = getTypedArray.bind(null, ctor, align);
    extendedExports[`__get${name}View`] = getTypedArrayView.bind(null, ctor, align);
  }

  [
    Int8Array,
    Uint8Array,
    Uint8ClampedArray,
    Int16Array,
    Uint16Array,
    Int32Array,
    Uint32Array,
    Float32Array,
    Float64Array
  ].forEach(ctor => {
    attachTypedArrayFunctions(ctor, ctor.name, 31 - Math.clz32(ctor.BYTES_PER_ELEMENT));
  });

  if (BIGINT) {
    [BigUint64Array, BigInt64Array].forEach(ctor => {
      attachTypedArrayFunctions(ctor, ctor.name.slice(3), 3);
    });
  }

  /** Tests whether an object is an instance of the class represented by the specified base id. */
  function __instanceof(ptr, baseId) {
    const U32 = new Uint32Array(memory.buffer);
    let id = U32[ptr + ID_OFFSET >>> 2];
    if (id <= getRttiCount(U32)) {
      do {
        if (id == baseId) return true;
        id = getBase(id);
      } while (id);
    }
    return false;
  }

  extendedExports.__instanceof = __instanceof;

  // Pull basic exports to extendedExports so code in preInstantiate can use them
  extendedExports.memory = extendedExports.memory || memory;
  extendedExports.table  = extendedExports.table  || table;

  // Demangle exports and provide the usual utility on the prototype
  return demangle(exports, extendedExports);
}

function isResponse(src) {
  return typeof Response !== "undefined" && src instanceof Response;
}

function isModule(src) {
  return src instanceof WebAssembly.Module;
}

/** Asynchronously instantiates an AssemblyScript module from anything that can be instantiated. */
async function instantiate(source, imports = {}) {
  if (isResponse(source = await source)) return instantiateStreaming(source, imports);
  const module = isModule(source) ? source : await WebAssembly.compile(source);
  const extended = preInstantiate(imports);
  const instance = await WebAssembly.instantiate(module, imports);
  const exports = postInstantiate(extended, instance);
  return { module, instance, exports };
}

/** Synchronously instantiates an AssemblyScript module from a WebAssembly.Module or binary buffer. */
function instantiateSync(source, imports = {}) {
  const module = isModule(source) ? source : new WebAssembly.Module(source);
  const extended = preInstantiate(imports);
  const instance = new WebAssembly.Instance(module, imports);
  const exports = postInstantiate(extended, instance);
  return { module, instance, exports };
}

/** Asynchronously instantiates an AssemblyScript module from a response, i.e. as obtained by `fetch`. */
async function instantiateStreaming(source, imports = {}) {
  if (!WebAssembly.instantiateStreaming) {
    return instantiate(
      isResponse(source = await source)
        ? source.arrayBuffer()
        : source,
      imports
    );
  }
  const extended = preInstantiate(imports);
  const result = await WebAssembly.instantiateStreaming(source, imports);
  const exports = postInstantiate(extended, result.instance);
  return { ...result, exports };
}

/** Demangles an AssemblyScript module's exports to a friendly object structure. */
function demangle(exports, extendedExports = {}) {
  const setArgumentsLength = exports["__argumentsLength"]
    ? length => { exports["__argumentsLength"].value = length; }
    : exports["__setArgumentsLength"] || exports["__setargc"] || (() => { /* nop */ });
  for (let internalName in exports) {
    if (!Object.prototype.hasOwnProperty.call(exports, internalName)) continue;
    const elem = exports[internalName];
    let parts = internalName.split(".");
    let curr = extendedExports;
    while (parts.length > 1) {
      let part = parts.shift();
      if (!Object.prototype.hasOwnProperty.call(curr, part)) curr[part] = {};
      curr = curr[part];
    }
    let name = parts[0];
    let hash = name.indexOf("#");
    if (hash >= 0) {
      const className = name.substring(0, hash);
      const classElem = curr[className];
      if (typeof classElem === "undefined" || !classElem.prototype) {
        const ctor = function(...args) {
          return ctor.wrap(ctor.prototype.constructor(0, ...args));
        };
        ctor.prototype = {
          valueOf() { return this[THIS]; }
        };
        ctor.wrap = function(thisValue) {
          return Object.create(ctor.prototype, { [THIS]: { value: thisValue, writable: false } });
        };
        if (classElem) Object.getOwnPropertyNames(classElem).forEach(name =>
          Object.defineProperty(ctor, name, Object.getOwnPropertyDescriptor(classElem, name))
        );
        curr[className] = ctor;
      }
      name = name.substring(hash + 1);
      curr = curr[className].prototype;
      if (/^(get|set):/.test(name)) {
        if (!Object.prototype.hasOwnProperty.call(curr, name = name.substring(4))) {
          let getter = exports[internalName.replace("set:", "get:")];
          let setter = exports[internalName.replace("get:", "set:")];
          Object.defineProperty(curr, name, {
            get() { return getter(this[THIS]); },
            set(value) { setter(this[THIS], value); },
            enumerable: true
          });
        }
      } else {
        if (name === 'constructor') {
          (curr[name] = (...args) => {
            setArgumentsLength(args.length);
            return elem(...args);
          }).original = elem;
        } else { // instance method
          (curr[name] = function(...args) { // !
            setArgumentsLength(args.length);
            return elem(this[THIS], ...args);
          }).original = elem;
        }
      }
    } else {
      if (/^(get|set):/.test(name)) {
        if (!Object.prototype.hasOwnProperty.call(curr, name = name.substring(4))) {
          Object.defineProperty(curr, name, {
            get: exports[internalName.replace("set:", "get:")],
            set: exports[internalName.replace("get:", "set:")],
            enumerable: true
          });
        }
      } else if (typeof elem === "function" && elem !== setArgumentsLength) {
        (curr[name] = (...args) => {
          setArgumentsLength(args.length);
          return elem(...args);
        }).original = elem;
      } else {
        curr[name] = elem;
      }
    }
  }
  return extendedExports;
}

var loader = {
  instantiate,
  instantiateSync,
  instantiateStreaming,
  demangle
};

class ConsoleImport {
    
    constructor() {
        
        this._exports = null;

        this.wasmImports = {
            consoleBindings: {
                   _log: (message) => {
        
                    console.log(this._exports.__getString(message));
        
                }
            }
        };
    }

    get wasmExports() {
		return this._exports
	}
	set wasmExports(e) {
		this._exports = e;
	}

	getFn(fnIndex) {
		if (!this.wasmExports)
			throw new Error(
				'Make sure you set .wasmExports after instantiating the Wasm module but before running the Wasm module.',
			)
		return this._exports.table.get(fnIndex)
	}
}

const Console = new ConsoleImport();
const imports = {...Console.wasmImports};


const initEBM = (_featureData, _sampleData, _editingFeature, _isClassification) => {
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

export { initEBM };
