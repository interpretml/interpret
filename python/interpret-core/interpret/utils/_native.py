# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

# TODO: Add unit tests for internal EBM interfacing
import platform
import ctypes as ct
import numpy as np
import os
import struct
import logging
from contextlib import AbstractContextManager

log = logging.getLogger(__name__)

class Native:

    # BoostFlags
    BoostFlags_Default              = 0x00000000
    BoostFlags_DisableNewtonGain    = 0x00000001
    BoostFlags_DisableNewtonUpdate  = 0x00000002
    BoostFlags_GradientSums         = 0x00000004
    BoostFlags_RandomSplits         = 0x00000008

    # InteractionFlags
    InteractionFlags_Default        = 0x00000000
    InteractionFlags_Pure           = 0x00000001

    # TraceLevel
    _Trace_Off = 0
    _Trace_Error = 1
    _Trace_Warning = 2
    _Trace_Info = 3
    _Trace_Verbose = 4

    _native = None
    # if we supported win32 32-bit functions then this would need to be WINFUNCTYPE
    _LogCallbackType = ct.CFUNCTYPE(None, ct.c_int32, ct.c_char_p)

    def __init__(self):
        # Do not call "Native()".  Call "Native.get_native_singleton()" instead
        pass

    @staticmethod
    def get_native_singleton(is_debug=False):
        if Native._native is None:
            log.info("EBM lib loading.")
            native = Native()
            native._initialize(is_debug=is_debug)
            Native._native = native
        return Native._native

    @staticmethod
    def _make_pointer(array, dtype, ndim=1, is_null_allowed=False):
        # using ndpointer creates cyclic garbage references which could clog up
        # our memory and require extra garbage collections.  This function avoids that

        if array is None:
            if not is_null_allowed:  # pragma: no cover
                raise ValueError("array cannot be None")
            return None

        if not isinstance(array, np.ndarray):  # pragma: no cover
            raise ValueError("array should be an ndarray")

        if array.dtype.type is not dtype:  # pragma: no cover
            raise ValueError(f"array should be an ndarray of type {dtype}")

        if not array.flags.c_contiguous:  # pragma: no cover
            raise ValueError("array should be a contiguous ndarray")

        if array.ndim != ndim:  # pragma: no cover
            raise ValueError(f"array should have {ndim} dimensions")

        return array.ctypes.data

    @staticmethod
    def _get_native_exception(error_code, native_function):  # pragma: no cover
        if error_code == -1:
            return Exception(f'Out of memory in {native_function}')
        elif error_code == -2:
            return Exception(f'Unexpected internal error in {native_function}')
        elif error_code == -3:
            return Exception(f'Illegal native parameter value in {native_function}')
        elif error_code == -4:
            return Exception(f'User native parameter value error in {native_function}')
        elif error_code == -5:
            return Exception(f'Thread start failed in {native_function}')
        elif error_code == -10:
            return Exception(f'Loss constructor native exception in {native_function}')
        elif error_code == -11:
            return Exception(f'Loss parameter unknown')
        elif error_code == -12:
            return Exception(f'Loss parameter value malformed')
        elif error_code == -13:
            return Exception(f'Loss parameter value out of range')
        elif error_code == -14:
            return Exception(f'Loss parameter mismatch')
        elif error_code == -15:
            return Exception(f'Unrecognized loss type')
        elif error_code == -16:
            return Exception(f'Illegal loss registration name')
        elif error_code == -17:
            return Exception(f'Illegal loss parameter name')
        elif error_code == -18:
            return Exception(f'Duplicate loss parameter name')
        else:
            return Exception(f'Unrecognized native return code {error_code} in {native_function}')

    @staticmethod
    def get_count_scores_c(n_classes):
        # this should reflect how the C code represents scores
        if n_classes < 0 or 2 == n_classes:
            return 1
        elif 2 < n_classes:
            return n_classes
        else:
            return 0

    def set_logging(self, level=None):
        # NOTE: Not part of code coverage. It runs in tests, but isn't registered for some reason.
        def native_log(trace_level, message):  # pragma: no cover
            try:
                message = message.decode("ascii")

                if trace_level == self._Trace_Error:
                    log.error(message)
                elif trace_level == self._Trace_Warning:
                    log.warning(message)
                elif trace_level == self._Trace_Info:
                    log.info(message)
                elif trace_level == self._Trace_Verbose:
                    log.debug(message)
            except:  # pragma: no cover
                # we're being called from C, so we can't raise exceptions
                pass

        if level is None:
            root = logging.getLogger("interpret")
            level = root.getEffectiveLevel()

        level_dict = {
            logging.DEBUG: self._Trace_Verbose,
            logging.INFO: self._Trace_Info,
            logging.WARNING: self._Trace_Warning,
            logging.ERROR: self._Trace_Error,
            logging.CRITICAL: self._Trace_Error,
            logging.NOTSET: self._Trace_Off,
            "DEBUG": self._Trace_Verbose,
            "INFO": self._Trace_Info,
            "WARNING": self._Trace_Warning,
            "ERROR": self._Trace_Error,
            "CRITICAL": self._Trace_Error,
            "NOTSET": self._Trace_Off,
        }

        trace_level = level_dict[level]
        if self._log_callback_func is None and trace_level != self._Trace_Off:
            # it's critical that we put _LogCallbackType(native_log) into 
            # self._log_callback_func, otherwise it will be garbage collected
            self._log_callback_func = self._LogCallbackType(native_log)
            self._unsafe.SetLogCallback(self._log_callback_func)

        self._unsafe.SetTraceLevel(trace_level)

    def clean_float(self, val):
        # the EBM spec does not allow subnormal floats to be in the model definition, so flush them to zero
        val_array = np.array([val], np.float64)
        self._unsafe.CleanFloats(len(val_array), Native._make_pointer(val_array, np.float64))
        return val_array[0]


    def create_rng(self, random_state):
        if random_state is None:
            return None # non-deterministic

        if random_state < -2147483648 or 2147483647 < random_state:
            msg = f"random_state of \"{random_state}\" must be cleaned to be a 32-bit signed integer before calling create_rng"
            log.error(msg)
            raise Exception(msg)

        n_bytes = self._unsafe.MeasureRNG()
        rng = np.empty(n_bytes, np.ubyte)

        self._unsafe.InitRNG(random_state, Native._make_pointer(rng, np.ubyte))
        return rng

    def copy_rng(self, rng):
        if rng is None:
            return None # non-deterministic

        n_bytes = self._unsafe.MeasureRNG()
        rngCopy = np.empty(n_bytes, np.ubyte)

        self._unsafe.CopyRNG(
            Native._make_pointer(rng, np.ubyte), 
            Native._make_pointer(rngCopy, np.ubyte)
        )
        return rngCopy

    def branch_rng(self, rng):
        if rng is None:
            return None # non-deterministic

        n_bytes = self._unsafe.MeasureRNG()
        rngBranch = np.empty(n_bytes, np.ubyte)

        self._unsafe.BranchRNG(
            Native._make_pointer(rng, np.ubyte), 
            Native._make_pointer(rngBranch, np.ubyte)
        )
        return rngBranch

    def generate_seed(self, rng):
        # Unlike our other functions, this will generate a 32-bit seed even if rng is None
        seed = ct.c_int32(0)
        return_code = self._unsafe.GenerateSeed(
            Native._make_pointer(rng, np.ubyte, is_null_allowed=True), 
            ct.byref(seed)
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "GenerateSeed")

        return seed.value

    def generate_gaussian_random(self, rng, stddev, count):
        random_numbers = np.empty(count, dtype=np.float64, order="C")
        return_code = self._unsafe.GenerateGaussianRandom(
            Native._make_pointer(rng, np.ubyte, is_null_allowed=True),
            stddev,
            count,
            Native._make_pointer(random_numbers, np.float64)
        )

        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "GenerateGaussianRandom")

        return random_numbers

    def get_histogram_cut_count(self, X_col):
        return self._unsafe.GetHistogramCutCount(X_col.shape[0], Native._make_pointer(X_col, np.float64))

    def cut_uniform(self, X_col, max_cuts):
        if max_cuts < 0:
            raise Exception(f"max_cuts can't be negative: {max_cuts}.")

        cuts = np.empty(max_cuts, dtype=np.float64, order="C")
        count_cuts = self._unsafe.CutUniform(
            X_col.shape[0],
            Native._make_pointer(X_col, np.float64),
            max_cuts,
            Native._make_pointer(cuts, np.float64),
        )
        return cuts[:count_cuts]

    def cut_quantile(self, X_col, min_samples_bin, is_rounded, max_cuts):
        if max_cuts < 0:
            raise Exception(f"max_cuts can't be negative: {max_cuts}.")

        cuts = np.empty(max_cuts, dtype=np.float64, order="C")
        count_cuts = ct.c_int64(max_cuts)
        return_code = self._unsafe.CutQuantile(
            X_col.shape[0],
            Native._make_pointer(X_col, np.float64),
            min_samples_bin,
            is_rounded,
            ct.byref(count_cuts),
            Native._make_pointer(cuts, np.float64),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "CutQuantile")

        return cuts[:count_cuts.value]

    def cut_winsorized(self, X_col, max_cuts):
        if max_cuts < 0:
            raise Exception(f"max_cuts can't be negative: {max_cuts}.")

        cuts = np.empty(max_cuts, dtype=np.float64, order="C")
        count_cuts = ct.c_int64(max_cuts)
        return_code = self._unsafe.CutWinsorized(
            X_col.shape[0],
            Native._make_pointer(X_col, np.float64),
            ct.byref(count_cuts),
            Native._make_pointer(cuts, np.float64),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "CutWinsorized")

        return cuts[:count_cuts.value]

    def suggest_graph_bounds(self, cuts, min_feature_val=np.nan, max_feature_val=np.nan):
        # This function will never return NaN values for low_graph_bound or high_graph_bound
        # It can however return -inf values for low_graph_bound and +inf for high_graph_bound
        # if the min_feature_val or max_feature_val are +-inf or if there's an overflow when
        # extending the bounds.
        # A possible dangerous return value of -inf or +inf for both low_graph_bound and high_graph_bound
        # can occur if one of the min/max values is missing (NaN), or if the dataset consists
        # entirely of +inf or -inf values which makes both min_feature_val and max_feature_val the same
        # extreme value.  In this case the caller should probably check if low_graph_bound == high_graph_bound
        # to avoid subtracting them, which would lead to a NaN value for the difference.
        # Also high_graph_bound - low_graph_bound can be +inf even if both low_graph_bound and high_graph_bound are 
        # normal numbers.  An example of this occuring would be if min was the lowest float
        # and max was the highest float.

        low_graph_bound = ct.c_double(np.nan)
        high_graph_bound = ct.c_double(np.nan)
        return_code = self._unsafe.SuggestGraphBounds(
            len(cuts),
            cuts[0] if 0 < len(cuts) else np.nan,
            cuts[-1] if 0 < len(cuts) else np.nan,
            min_feature_val,
            max_feature_val,
            ct.byref(low_graph_bound),
            ct.byref(high_graph_bound),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "SuggestGraphBounds")

        return low_graph_bound.value, high_graph_bound.value

    def discretize(self, X_col, cuts):
        # TODO: for speed and efficiency, we should instead accept in the bin_indexes array
        bin_indexes = np.empty(X_col.shape[0], dtype=np.int64, order="C")
        return_code = self._unsafe.Discretize(
            X_col.shape[0],
            Native._make_pointer(X_col, np.float64),
            cuts.shape[0],
            Native._make_pointer(cuts, np.float64),
            Native._make_pointer(bin_indexes, np.int64),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "Discretize")

        return bin_indexes

    def measure_dataset_header(self, n_features, n_weights, n_targets):
        n_bytes = self._unsafe.MeasureDataSetHeader(n_features, n_weights, n_targets)
        if n_bytes < 0:  # pragma: no cover
            raise Native._get_native_exception(n_bytes, "MeasureDataSetHeader")
        return n_bytes

    def measure_feature(self, n_bins, is_missing, is_unknown, is_nominal, bin_indexes):
        n_bytes = self._unsafe.MeasureFeature(
            n_bins, 
            is_missing, 
            is_unknown, 
            is_nominal, 
            len(bin_indexes), 
            Native._make_pointer(bin_indexes, np.int64),
        )
        if n_bytes < 0:  # pragma: no cover
            raise Native._get_native_exception(n_bytes, "MeasureFeature")
        return n_bytes

    def measure_weight(self, weights):
        n_bytes = self._unsafe.MeasureWeight(
            len(weights), 
            Native._make_pointer(weights, np.float64),
        )
        if n_bytes < 0:  # pragma: no cover
            raise Native._get_native_exception(n_bytes, "MeasureWeight")
        return n_bytes

    def measure_classification_target(self, n_classes, targets):
        n_bytes = self._unsafe.MeasureClassificationTarget(
            n_classes, 
            len(targets), 
            Native._make_pointer(targets, np.int64),
        )
        if n_bytes < 0:  # pragma: no cover
            raise Native._get_native_exception(n_bytes, "MeasureClassificationTarget")
        return n_bytes

    def measure_regression_target(self, targets):
        n_bytes = self._unsafe.MeasureRegressionTarget(
            len(targets), 
            Native._make_pointer(targets, np.float64),
        )
        if n_bytes < 0:  # pragma: no cover
            raise Native._get_native_exception(n_bytes, "MeasureRegressionTarget")
        return n_bytes

    def fill_dataset_header(self, n_features, n_weights, n_targets, dataset):
        return_code = self._unsafe.FillDataSetHeader(
            n_features, 
            n_weights, 
            n_targets, 
            dataset.nbytes,
            Native._make_pointer(dataset, np.ubyte),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "FillDataSetHeader")

    def fill_feature(self, n_bins, is_missing, is_unknown, is_nominal, bin_indexes, dataset):
        return_code = self._unsafe.FillFeature(
            n_bins, 
            is_missing, 
            is_unknown, 
            is_nominal, 
            len(bin_indexes), 
            Native._make_pointer(bin_indexes, np.int64),
            dataset.nbytes, 
            Native._make_pointer(dataset, np.ubyte),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "FillFeature")

    def fill_weight(self, weights, dataset):
        return_code = self._unsafe.FillWeight(
            len(weights), 
            Native._make_pointer(weights, np.float64),
            dataset.nbytes, 
            Native._make_pointer(dataset, np.ubyte),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "FillWeight")

    def fill_classification_target(self, n_classes, targets, dataset):
        return_code = self._unsafe.FillClassificationTarget(
            n_classes, 
            len(targets), 
            Native._make_pointer(targets, np.int64),
            dataset.nbytes, 
            Native._make_pointer(dataset, np.ubyte),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "FillClassificationTarget")

    def fill_regression_target(self, targets, dataset):
        return_code = self._unsafe.FillRegressionTarget(
            len(targets), 
            Native._make_pointer(targets, np.float64),
            dataset.nbytes, 
            Native._make_pointer(dataset, np.ubyte),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "FillRegressionTarget")

    def check_dataset(self, dataset):
        return_code = self._unsafe.CheckDataSet(
            dataset.nbytes, 
            Native._make_pointer(dataset, np.ubyte),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "CheckDataSet")

    def extract_dataset_header(self, dataset):
        n_samples = ct.c_int64(-1)
        n_features = ct.c_int64(-1)
        n_weights = ct.c_int64(-1)
        n_targets = ct.c_int64(-1)

        return_code = self._unsafe.ExtractDataSetHeader(
            Native._make_pointer(dataset, np.ubyte),
            ct.byref(n_samples),
            ct.byref(n_features),
            ct.byref(n_weights),
            ct.byref(n_targets),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "ExtractDataSetHeader")

        return n_samples.value, n_features.value, n_weights.value, n_targets.value

    def extract_bin_counts(self, dataset, n_features):
        bin_counts = np.empty(n_features, dtype=np.int64, order="C")

        return_code = self._unsafe.ExtractBinCounts(
            Native._make_pointer(dataset, np.ubyte),
            n_features, 
            Native._make_pointer(bin_counts, np.int64),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "ExtractBinCounts")

        return bin_counts

    def extract_target_classes(self, dataset, n_targets):
        class_counts = np.empty(n_targets, dtype=np.int64, order="C")

        return_code = self._unsafe.ExtractTargetClasses(
            Native._make_pointer(dataset, np.ubyte),
            n_targets, 
            Native._make_pointer(class_counts, np.int64),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "ExtractTargetClasses")

        return class_counts

    def sample_without_replacement(
        self, 
        rng, 
        count_training_samples,
        count_validation_samples
    ):
        count_samples = count_training_samples + count_validation_samples

        bag = np.empty(count_samples, dtype=np.int8, order="C")

        return_code = self._unsafe.SampleWithoutReplacement(
            Native._make_pointer(rng, np.ubyte, is_null_allowed=True),
            count_training_samples,
            count_validation_samples,
            Native._make_pointer(bag, np.int8),
        )

        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "SampleWithoutReplacement")

        return bag

    def sample_without_replacement_stratified(
        self, 
        rng, 
        n_classes,
        count_training_samples,
        count_validation_samples,
        targets
    ):
        count_samples = count_training_samples + count_validation_samples

        if len(targets) != count_samples:
            raise ValueError("count_training_samples + count_validation_samples should be equal to len(targets)")

        bag = np.empty(count_samples, dtype=np.int8, order="C")

        return_code = self._unsafe.SampleWithoutReplacementStratified(
            Native._make_pointer(rng, np.ubyte, is_null_allowed=True),
            n_classes,
            count_training_samples,
            count_validation_samples,
            Native._make_pointer(targets, np.int64),
            Native._make_pointer(bag, np.int8),
        )

        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "SampleWithoutReplacementStratified")

        return bag


    @staticmethod
    def _get_ebm_lib_path(debug=False):
        """ Returns filepath of core EBM library.

        Returns:
            A string representing filepath.
        """
        bitsize = struct.calcsize("P") * 8
        is_64_bit = bitsize == 64

        script_path = os.path.dirname(os.path.abspath(__file__))
        package_path = os.path.join(script_path, "..")

        debug_str = "_debug" if debug else ""
        log.info("Loading native on {0} | debug = {1}".format(platform.system(), debug))
        if platform.system() == "Linux" and platform.machine() == 'x86_64' and is_64_bit:  # pragma: no cover
            return os.path.join(
                package_path, "lib", "lib_ebm_native_linux_x64{0}.so".format(debug_str)
            )
        elif platform.system() == "Windows" and platform.machine() == 'AMD64' and is_64_bit:  # pragma: no cover
            return os.path.join(
                package_path, "lib", "lib_ebm_native_win_x64{0}.dll".format(debug_str)
            )
        elif platform.system() == "Darwin" and platform.machine() == 'x86_64' and is_64_bit:  # pragma: no cover
            return os.path.join(
                package_path, "lib", "lib_ebm_native_mac_x64{0}.dylib".format(debug_str)
            )
        elif platform.system() == "Darwin" and platform.machine() == 'arm64' and is_64_bit:  # pragma: no cover
            return os.path.join(
                package_path, "lib", "lib_ebm_native_mac_arm{0}.dylib".format(debug_str)
            )
        else:  # pragma: no cover
            msg = "System {0}, platform {1}, bitsize {2} not supported for EBM".format(
                platform.system(), platform.machine(), bitsize
            )
            log.error(msg)
            raise Exception(msg)

    def _initialize(self, is_debug):
        self.is_debug = is_debug

        self._log_callback_func = None
        self._unsafe = ct.cdll.LoadLibrary(Native._get_ebm_lib_path(debug=is_debug))

        self._unsafe.SetLogCallback.argtypes = [
            # void (* LogCallbackFunction)(int32_t traceLevel, const char * message) logCallbackFunction
            self._LogCallbackType
        ]
        self._unsafe.SetLogCallback.restype = None

        self._unsafe.SetTraceLevel.argtypes = [
            # int32 traceLevel
            ct.c_int32
        ]
        self._unsafe.SetTraceLevel.restype = None

        self._unsafe.CleanFloats.argtypes = [
            # int64_t count
            ct.c_int64,
            # double * valsInOut
            ct.c_void_p,
        ]
        self._unsafe.CleanFloats.restype = None


        self._unsafe.MeasureRNG.argtypes = [
        ]
        self._unsafe.MeasureRNG.restype = ct.c_int64

        self._unsafe.InitRNG.argtypes = [
            # int32_t seed
            ct.c_int32,
            # void * rngOut
            ct.c_void_p,
        ]
        self._unsafe.InitRNG.restype = None

        self._unsafe.CopyRNG.argtypes = [
            # void * rng
            ct.c_void_p,
            # void * rngOut
            ct.c_void_p,
        ]
        self._unsafe.CopyRNG.restype = None

        self._unsafe.BranchRNG.argtypes = [
            # void * rng
            ct.c_void_p,
            # void * rngOut
            ct.c_void_p,
        ]
        self._unsafe.BranchRNG.restype = None
        
        self._unsafe.GenerateSeed.argtypes = [
            # void * rng
            ct.c_void_p,
            # SeedEbm * seedOut
            ct.POINTER(ct.c_int32),
        ]
        self._unsafe.GenerateSeed.restype = ct.c_int32

        self._unsafe.GenerateGaussianRandom.argtypes = [
            # void * rng
            ct.c_void_p,
            # double stddev
            ct.c_double,
            # int64_t count
            ct.c_int64,
            # double * randomOut
            ct.c_void_p,
        ]
        self._unsafe.GenerateGaussianRandom.restype = ct.c_int32


        self._unsafe.GetHistogramCutCount.argtypes = [
            # int64_t countSamples
            ct.c_int64,
            # double * featureVals
            ct.c_void_p,
        ]
        self._unsafe.GetHistogramCutCount.restype = ct.c_int64

        self._unsafe.CutUniform.argtypes = [
            # int64_t countSamples
            ct.c_int64,
            # double * featureVals
            ct.c_void_p,
            # int64_t countDesiredCuts
            ct.c_int64,
            # double * cutsLowerBoundInclusiveOut
            ct.c_void_p,
        ]
        self._unsafe.CutUniform.restype = ct.c_int64

        self._unsafe.CutQuantile.argtypes = [
            # int64_t countSamples
            ct.c_int64,
            # double * featureVals
            ct.c_void_p,
            # int64_t minSamplesBin
            ct.c_int64,
            # int32_t isRounded
            ct.c_int32,
            # int64_t * countCutsInOut
            ct.POINTER(ct.c_int64),
            # double * cutsLowerBoundInclusiveOut
            ct.c_void_p,
        ]
        self._unsafe.CutQuantile.restype = ct.c_int32

        self._unsafe.CutWinsorized.argtypes = [
            # int64_t countSamples
            ct.c_int64,
            # double * featureVals
            ct.c_void_p,
            # int64_t * countCutsInOut
            ct.POINTER(ct.c_int64),
            # double * cutsLowerBoundInclusiveOut
            ct.c_void_p,
        ]
        self._unsafe.CutWinsorized.restype = ct.c_int32


        self._unsafe.SuggestGraphBounds.argtypes = [
            # int64_t countCuts
            ct.c_int64,
            # double lowestCut
            ct.c_double,
            # double highestCut
            ct.c_double,
            # double minFeatureVal
            ct.c_double,
            # double maxFeatureVal
            ct.c_double,
            # double * lowGraphBoundOut
            ct.POINTER(ct.c_double),
            # double * highGraphBoundOut
            ct.POINTER(ct.c_double),
        ]
        self._unsafe.SuggestGraphBounds.restype = ct.c_int32


        self._unsafe.Discretize.argtypes = [
            # int64_t countSamples
            ct.c_int64,
            # double * featureVals
            ct.c_void_p,
            # int64_t countCuts
            ct.c_int64,
            # double * cutsLowerBoundInclusive
            ct.c_void_p,
            # int64_t * binIndexesOut
            ct.c_void_p,
        ]
        self._unsafe.Discretize.restype = ct.c_int32


        self._unsafe.MeasureDataSetHeader.argtypes = [
            # int64_t countFeatures
            ct.c_int64,
            # int64_t countWeights
            ct.c_int64,
            # int64_t countTargets
            ct.c_int64,
        ]
        self._unsafe.MeasureDataSetHeader.restype = ct.c_int64

        self._unsafe.MeasureFeature.argtypes = [
            # int64_t countBins
            ct.c_int64,
            # int32_t isMissing
            ct.c_int32,
            # int32_t isUnknown
            ct.c_int32,
            # int32_t isNominal
            ct.c_int32,
            # int64_t countSamples
            ct.c_int64,
            # int64_t * binIndexes
            ct.c_void_p,
        ]
        self._unsafe.MeasureFeature.restype = ct.c_int64

        self._unsafe.MeasureWeight.argtypes = [
            # int64_t countSamples
            ct.c_int64,
            # double * weights
            ct.c_void_p,
        ]
        self._unsafe.MeasureWeight.restype = ct.c_int64

        self._unsafe.MeasureClassificationTarget.argtypes = [
            # int64_t countClasses
            ct.c_int64,
            # int64_t countSamples
            ct.c_int64,
            # int64_t * targets
            ct.c_void_p,
        ]
        self._unsafe.MeasureClassificationTarget.restype = ct.c_int64

        self._unsafe.MeasureRegressionTarget.argtypes = [
            # int64_t countSamples
            ct.c_int64,
            # double * targets
            ct.c_void_p,
        ]
        self._unsafe.MeasureRegressionTarget.restype = ct.c_int64


        self._unsafe.FillDataSetHeader.argtypes = [
            # int64_t countFeatures
            ct.c_int64,
            # int64_t countWeights
            ct.c_int64,
            # int64_t countTargets
            ct.c_int64,
            # int64_t countBytesAllocated
            ct.c_int64,
            # void * fillMem
            ct.c_void_p,
        ]
        self._unsafe.FillDataSetHeader.restype = ct.c_int32

        self._unsafe.FillFeature.argtypes = [
            # int64_t countBins
            ct.c_int64,
            # int32_t isMissing
            ct.c_int32,
            # int32_t isUnknown
            ct.c_int32,
            # int32_t isNominal
            ct.c_int32,
            # int64_t countSamples
            ct.c_int64,
            # int64_t * binIndexes
            ct.c_void_p,
            # int64_t countBytesAllocated
            ct.c_int64,
            # void * fillMem
            ct.c_void_p,
        ]
        self._unsafe.FillFeature.restype = ct.c_int32

        self._unsafe.FillWeight.argtypes = [
            # int64_t countSamples
            ct.c_int64,
            # double * weights
            ct.c_void_p,
            # int64_t countBytesAllocated
            ct.c_int64,
            # void * fillMem
            ct.c_void_p,
        ]
        self._unsafe.FillWeight.restype = ct.c_int32

        self._unsafe.FillClassificationTarget.argtypes = [
            # int64_t countClasses
            ct.c_int64,
            # int64_t countSamples
            ct.c_int64,
            # int64_t * targets
            ct.c_void_p,
            # int64_t countBytesAllocated
            ct.c_int64,
            # void * fillMem
            ct.c_void_p,
        ]
        self._unsafe.FillClassificationTarget.restype = ct.c_int32

        self._unsafe.FillRegressionTarget.argtypes = [
            # int64_t countSamples
            ct.c_int64,
            # double * targets
            ct.c_void_p,
            # int64_t countBytesAllocated
            ct.c_int64,
            # void * fillMem
            ct.c_void_p,
        ]
        self._unsafe.FillRegressionTarget.restype = ct.c_int32

        self._unsafe.CheckDataSet.argtypes = [
            # int64_t countBytesAllocated
            ct.c_int64,
            # void * dataSet
            ct.c_void_p,
        ]
        self._unsafe.CheckDataSet.restype = ct.c_int32

        self._unsafe.ExtractDataSetHeader.argtypes = [
            # void * dataSet
            ct.c_void_p,
            # int64_t * countSamplesOut
            ct.POINTER(ct.c_int64),
            # int64_t * countFeaturesOut
            ct.POINTER(ct.c_int64),
            # int64_t * countWeightsOut
            ct.POINTER(ct.c_int64),
            # int64_t * countTargetsOut
            ct.POINTER(ct.c_int64),
        ]
        self._unsafe.ExtractDataSetHeader.restype = ct.c_int32

        self._unsafe.ExtractBinCounts.argtypes = [
            # void * dataSet
            ct.c_void_p,
            # int64_t countFeaturesVerify
            ct.c_int64,
            # int64_t * binCountsOut
            ct.c_void_p,
        ]
        self._unsafe.ExtractBinCounts.restype = ct.c_int32

        self._unsafe.ExtractTargetClasses.argtypes = [
            # void * dataSet
            ct.c_void_p,
            # int64_t countTargetsVerify
            ct.c_int64,
            # int64_t * classCountsOut
            ct.c_void_p,
        ]
        self._unsafe.ExtractTargetClasses.restype = ct.c_int32


        self._unsafe.SampleWithoutReplacement.argtypes = [
            # void * rng
            ct.c_void_p,
            # int64_t countTrainingSamples
            ct.c_int64,
            # int64_t countValidationSamples
            ct.c_int64,
            # int8_t * bagOut
            ct.c_void_p,
        ]
        self._unsafe.SampleWithoutReplacement.restype = ct.c_int32

        self._unsafe.SampleWithoutReplacementStratified.argtypes = [
            # void * rng
            ct.c_void_p,
            # int64_t countClasses
            ct.c_int64,
            # int64_t countTrainingSamples
            ct.c_int64,
            # int64_t countValidationSamples
            ct.c_int64,
            # int64_t * targets
            ct.c_void_p,
            # int8_t * bagOut
            ct.c_void_p,
        ]
        self._unsafe.SampleWithoutReplacementStratified.restype = ct.c_int32


        self._unsafe.CreateBooster.argtypes = [
            # void * rng
            ct.c_void_p,
            # void * dataSet
            ct.c_void_p,
            # int8_t * bag
            ct.c_void_p,
            # double * initScores
            ct.c_void_p,
            # int64_t countTerms
            ct.c_int64,
            # int64_t * dimensionCounts
            ct.c_void_p,
            # int64_t * featureIndexes
            ct.c_void_p,
            # int64_t countInnerBags
            ct.c_int64,
            # double * experimentalParams
            ct.c_void_p,
            # BoosterHandle * boosterHandleOut
            ct.POINTER(ct.c_void_p),
        ]
        self._unsafe.CreateBooster.restype = ct.c_int32

        self._unsafe.FreeBooster.argtypes = [
            # void * boosterHandle
            ct.c_void_p
        ]
        self._unsafe.FreeBooster.restype = None

        self._unsafe.GenerateTermUpdate.argtypes = [
            # void * rng
            ct.c_void_p,
            # void * boosterHandle
            ct.c_void_p,
            # int64_t indexTerm
            ct.c_int64,
            # BoostFlags flags 
            ct.c_int32,
            # double learningRate
            ct.c_double,
            # int64_t minSamplesLeaf
            ct.c_int64,
            # int64_t * leavesMax
            ct.c_void_p,
            # double * avgGainOut
            ct.POINTER(ct.c_double),
        ]
        self._unsafe.GenerateTermUpdate.restype = ct.c_int32

        self._unsafe.GetTermUpdateSplits.argtypes = [
            # void * boosterHandle
            ct.c_void_p,
            # int64_t indexDimension
            ct.c_int64,
            # int64_t * countSplitsInOut
            ct.POINTER(ct.c_int64),
            # int64_t * splitIndexesOut
            ct.c_void_p,
        ]
        self._unsafe.GetTermUpdateSplits.restype = ct.c_int32

        self._unsafe.GetTermUpdate.argtypes = [
            # void * boosterHandle
            ct.c_void_p,
            # double * updateScoresTensorOut
            ct.c_void_p,
        ]
        self._unsafe.GetTermUpdate.restype = ct.c_int32

        self._unsafe.SetTermUpdate.argtypes = [
            # void * boosterHandle
            ct.c_void_p,
            # int64_t indexTerm
            ct.c_int64,
            # double * updateScoresTensor
            ct.c_void_p,
        ]
        self._unsafe.SetTermUpdate.restype = ct.c_int32

        self._unsafe.ApplyTermUpdate.argtypes = [
            # void * boosterHandle
            ct.c_void_p,
            # double * avgValidationMetricOut
            ct.POINTER(ct.c_double),
        ]
        self._unsafe.ApplyTermUpdate.restype = ct.c_int32

        self._unsafe.GetBestTermScores.argtypes = [
            # void * boosterHandle
            ct.c_void_p,
            # int64_t indexTerm
            ct.c_int64,
            # double * termScoresTensorOut
            ct.c_void_p,
        ]
        self._unsafe.GetBestTermScores.restype = ct.c_int32

        self._unsafe.GetCurrentTermScores.argtypes = [
            # void * boosterHandle
            ct.c_void_p,
            # int64_t indexTerm
            ct.c_int64,
            # double * termScoresTensorOut
            ct.c_void_p,
        ]
        self._unsafe.GetCurrentTermScores.restype = ct.c_int32


        self._unsafe.CreateInteractionDetector.argtypes = [
            # void * dataSet
            ct.c_void_p,
            # int8_t * bag
            ct.c_void_p,
            # double * initScores
            ct.c_void_p,
            # double * experimentalParams
            ct.c_void_p,
            # InteractionHandle * interactionHandleOut
            ct.POINTER(ct.c_void_p),
        ]
        self._unsafe.CreateInteractionDetector.restype = ct.c_int32

        self._unsafe.FreeInteractionDetector.argtypes = [
            # void * interactionHandle
            ct.c_void_p
        ]
        self._unsafe.FreeInteractionDetector.restype = None

        self._unsafe.CalcInteractionStrength.argtypes = [
            # void * interactionHandle
            ct.c_void_p,
            # int64_t countDimensions
            ct.c_int64,
            # int64_t * featureIndexes
            ct.c_void_p,
            # InteractionFlags flags 
            ct.c_int32,
            # int64_t minSamplesLeaf
            ct.c_int64,
            # double * avgInteractionStrengthOut
            ct.POINTER(ct.c_double),
        ]
        self._unsafe.CalcInteractionStrength.restype = ct.c_int32

class Booster(AbstractContextManager):
    """Lightweight wrapper for EBM C boosting code.
    """

    def __init__(
        self,
        dataset,
        bag,
        init_scores,
        term_features,
        n_inner_bags,
        rng,
        experimental_params,
    ):

        """ Initializes internal wrapper for EBM C code.

        Args:
            dataset: binned data in a compressed native form
            bag: definition of what data is included. 1 = training, -1 = validation, 0 = not included
            init_scores: predictions from a prior predictor
                that this class will boost on top of.  For regression
                there is 1 prediction per sample.  For binary classification
                there is one logit.  For multiclass there are n_classes logits
            term_features: List of term feature indexes
            n_inner_bags: number of inner bags.
            rng: native random number generator
            experimental_params: unused data that can be passed into the native layer for debugging
        """

        self.dataset = dataset
        self.bag = bag
        self.init_scores = init_scores
        self.term_features = term_features
        self.n_inner_bags = n_inner_bags
        self.rng = rng
        self.experimental_params = experimental_params

        # start off with an invalid _term_idx
        self._term_idx = -1

    def __enter__(self):
        log.info("Booster allocation start")

        dimension_counts = np.empty(len(self.term_features), ct.c_int64)
        feature_indexes = []
        for term_idx, feature_idxs in enumerate(self.term_features):
            dimension_counts.itemset(term_idx, len(feature_idxs))
            feature_indexes.extend(feature_idxs)
        feature_indexes = np.array(feature_indexes, ct.c_int64)

        native = Native.get_native_singleton()

        n_samples, n_features, n_weights, n_targets = native.extract_dataset_header(self.dataset)

        if n_weights != 0 and n_weights != 1:  # pragma: no cover
            raise ValueError("n_weights must be 0 or 1")

        if n_targets != 1:  # pragma: no cover
            raise ValueError("n_targets must be 1")

        class_counts = native.extract_target_classes(self.dataset, n_targets)
        n_class_scores = sum((Native.get_count_scores_c(n_classes) for n_classes in class_counts))

        self._term_shapes = None
        if 0 < n_class_scores:
            bin_counts = native.extract_bin_counts(self.dataset, n_features)
            self._term_shapes = []
            for feature_idxs in self.term_features:
                dimensions = [bin_counts[feature_idx] for feature_idx in feature_idxs]
                dimensions.reverse()

                # Array returned for multiclass is one higher dimension
                if 1 < n_class_scores:
                    dimensions.append(n_class_scores)

                self._term_shapes.append(tuple(dimensions))

        n_bagged_samples = n_samples
        if self.bag is not None:
            if self.bag.shape[0] != n_samples:  # pragma: no cover
                raise ValueError("bag should be len(n_samples)")
            n_bagged_samples = np.count_nonzero(self.bag)

        if self.init_scores is not None:
            if self.init_scores.shape[0] != n_bagged_samples:  # pragma: no cover
                raise ValueError("init_scores should have the same length as the number of non-zero bag entries")

            if n_class_scores != 1 and self.init_scores.shape[1] != n_class_scores:  # pragma: no cover
                raise ValueError(f"init_scores should have {n_class_scores} scores")

        # Allocate external resources
        booster_handle = ct.c_void_p(0)
        return_code = native._unsafe.CreateBooster(
            Native._make_pointer(self.rng, np.ubyte, is_null_allowed=True),
            Native._make_pointer(self.dataset, np.ubyte),
            Native._make_pointer(self.bag, np.int8, 1, True),
            Native._make_pointer(self.init_scores, np.float64, 2 if 1 < n_class_scores else 1, True),
            len(dimension_counts),
            Native._make_pointer(dimension_counts, np.int64),
            Native._make_pointer(feature_indexes, np.int64),
            self.n_inner_bags,
            Native._make_pointer(self.experimental_params, np.float64, 1, True),
            ct.byref(booster_handle),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "CreateBooster")

        self._booster_handle = booster_handle.value

        log.info("Booster allocation end")
        return self

    def __exit__(self, *args):

        self.close()

    def close(self):

        """ Deallocates C objects used to boost EBM. """
        log.info("Deallocation boosting start")

        booster_handle = getattr(self, "_booster_handle", None)
        if booster_handle:
            native = Native.get_native_singleton()
            self._booster_handle = None
            native._unsafe.FreeBooster(booster_handle)

        log.info("Deallocation boosting end")

    def generate_term_update(
        self, 
        term_idx, 
        boost_flags, 
        learning_rate, 
        min_samples_leaf, 
        max_leaves, 
    ):

        """ Generates a boosting step update per feature
            by growing a shallow decision tree.

        Args:
            term_idx: The index for the term to generate the update for
            boost_flags: C interface options
            learning_rate: Learning rate as a float.
            min_samples_leaf: Min observations required to split.
            max_leaves: Max leaf nodes on feature step.

        Returns:
            gain for the generated boosting step.
        """

        # log.debug("Boosting step start")

        self._term_idx = -1

        native = Native.get_native_singleton()

        avg_gain = ct.c_double(0.0)
        n_features = len(self.term_features[term_idx])
        max_leaves_arr = np.full(n_features, max_leaves, dtype=ct.c_int64, order="C")

        return_code = native._unsafe.GenerateTermUpdate(
            Native._make_pointer(self.rng, np.ubyte, is_null_allowed=True),
            self._booster_handle, 
            term_idx,
            boost_flags,
            learning_rate,
            min_samples_leaf,
            Native._make_pointer(max_leaves_arr, np.int64),
            ct.byref(avg_gain),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "GenerateTermUpdate")
            
        self._term_idx = term_idx

        # log.debug("Boosting step end")
        return avg_gain.value

    def apply_term_update(self):

        """ Updates the interal C state with the last model update

        Args:

        Returns:
            Validation loss for the boosting step.
        """
        # log.debug("Boosting step start")

        self._term_idx = -1

        native = Native.get_native_singleton()

        avg_validation_metric = ct.c_double(np.inf)
        return_code = native._unsafe.ApplyTermUpdate(
            self._booster_handle, 
            ct.byref(avg_validation_metric),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "ApplyTermUpdate")

        # log.debug("Boosting step end")
        return avg_validation_metric.value

    def get_best_model(self):
        model = []
        for term_idx in range(len(self.term_features)):
            term_scores = self._get_best_term_scores(term_idx)
            model.append(term_scores)

        return model

    # TODO: Needs test.
    def get_current_model(self):
        model = []
        for term_idx in range(len(self.term_features)):
            term_scores = self._get_current_term_scores(term_idx)
            model.append(term_scores)

        return model

    def get_term_update_splits(self):
        if self._term_idx < 0:  # pragma: no cover
            raise RuntimeError("invalid internal self._term_idx")

        splits = []
        feature_idxs = self.term_features[self._term_idx]
        for dimension_idx in range(len(feature_idxs)):
            splits_dimension = self._get_term_update_splits_dimension(dimension_idx)
            splits.append(splits_dimension)

        return splits

    def _get_best_term_scores(self, term_idx):
        """ Returns best model/function according to validation set
            for a given feature group.

        Args:
            term_idx: The index for the feature group.

        Returns:
            An ndarray that represents the model.
        """

        if self._term_shapes is None:  # pragma: no cover
            # if there is only one legal state for a classification problem, then we know with 100%
            # certainty what the result will be, and our model has no information since we always predict
            # the only output
            return None

        native = Native.get_native_singleton()

        shape = self._term_shapes[term_idx]
        term_scores = np.empty(shape, dtype=np.float64, order="C")

        return_code = native._unsafe.GetBestTermScores(
            self._booster_handle, 
            term_idx, 
            Native._make_pointer(term_scores, np.float64, len(shape)),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "GetBestTermScores")

        n_dimensions = len(self.term_features[term_idx])
        temp_transpose = [*range(n_dimensions - 1, -1, -1)]
        if len(shape) != n_dimensions: # multiclass
            temp_transpose.append(len(temp_transpose))
        term_scores = np.ascontiguousarray(np.transpose(term_scores, tuple(temp_transpose)))
        return term_scores

    def _get_current_term_scores(self, term_idx):
        """ Returns current model/function according to validation set
            for a given feature group.

        Args:
            term_idx: The index for the feature group.

        Returns:
            An ndarray that represents the model.
        """

        if self._term_shapes is None:  # pragma: no cover
            # if there is only one legal state for a classification problem, then we know with 100%
            # certainty what the result will be, and our model has no information since we always predict
            # the only output
            return None

        native = Native.get_native_singleton()

        shape = self._term_shapes[term_idx]
        term_scores = np.empty(shape, dtype=np.float64, order="C")

        return_code = native._unsafe.GetCurrentTermScores(
            self._booster_handle, 
            term_idx, 
            Native._make_pointer(term_scores, np.float64, len(shape)),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "GetCurrentTermScores")

        n_dimensions = len(self.term_features[term_idx])
        temp_transpose = [*range(n_dimensions - 1, -1, -1)]
        if len(shape) != n_dimensions: # multiclass
            temp_transpose.append(len(temp_transpose))
        term_scores = np.ascontiguousarray(np.transpose(term_scores, tuple(temp_transpose)))
        return term_scores

    def _get_term_update_splits_dimension(self, dimension_index):
        if self._term_shapes is None:  # pragma: no cover
            # if there is only one legal state for a classification problem, then we know with 100%
            # certainty what the result will be, and our model has no information since we always predict
            # the only output
            return np.empty(0, np.int64)

        native = Native.get_native_singleton()

        n_bins = self._term_shapes[self._term_idx][dimension_index]

        count_splits = n_bins - 1
        splits = np.empty(count_splits, dtype=np.int64, order="C")
        count_splits = ct.c_int64(count_splits)

        return_code = native._unsafe.GetTermUpdateSplits(
            self._booster_handle, 
            dimension_index, 
            ct.byref(count_splits), 
            Native._make_pointer(splits, np.int64),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "GetTermUpdateSplits")

        splits = splits[:count_splits.value]
        return splits

    def get_term_update(self):
        if self._term_idx < 0:  # pragma: no cover
            raise RuntimeError("invalid internal self._term_idx")

        if self._term_shapes is None:  # pragma: no cover
            # if there is only one legal state for a classification problem, then we know with 100%
            # certainty what the result will be, and our model has no information since we always predict
            # the only output
            return None

        native = Native.get_native_singleton()

        shape = self._term_shapes[self._term_idx]
        update_scores = np.empty(shape, dtype=np.float64, order="C")

        return_code = native._unsafe.GetTermUpdate(
            self._booster_handle, 
            Native._make_pointer(update_scores, np.float64, len(shape)),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "GetTermUpdate")

        n_dimensions = len(self.term_features[self._term_idx])
        temp_transpose = [*range(n_dimensions - 1, -1, -1)]
        if len(shape) != n_dimensions: # multiclass
            temp_transpose.append(len(temp_transpose))
        update_scores = np.ascontiguousarray(np.transpose(update_scores, tuple(temp_transpose)))
        return update_scores

    def set_term_update(self, term_idx, update_scores):
        self._term_idx = -1

        if self._term_shapes is None:  # pragma: no cover
            if update_scores is None:  # pragma: no cover
                self._term_idx = term_idx
                return
            raise ValueError("a tensor with 1 class or less would be empty since the predictions would always be the same")

        shape = self._term_shapes[term_idx]

        n_dimensions = len(self.term_features[term_idx])
        temp_transpose = [*range(n_dimensions - 1, -1, -1)]
        if len(shape) != n_dimensions: # multiclass
            temp_transpose.append(len(temp_transpose))
        update_scores = np.ascontiguousarray(np.transpose(update_scores, tuple(temp_transpose)))

        if shape != update_scores.shape:  # pragma: no cover
            raise ValueError("incorrect tensor shape in call to set_term_update")

        native = Native.get_native_singleton()
        return_code = native._unsafe.SetTermUpdate(
            self._booster_handle, 
            term_idx, 
            Native._make_pointer(update_scores, np.float64, len(shape)),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "SetTermUpdate")

        self._term_idx = term_idx

        return


class InteractionDetector(AbstractContextManager):
    """Lightweight wrapper for EBM C interaction code.
    """

    def __init__(
        self, 
        dataset,
        bag,
        init_scores,
        experimental_params,
    ):

        """ Initializes internal wrapper for EBM C code.

        Args:
            dataset: binned data in a compressed native form
            bag: definition of what data is included. 1 = training, -1 = validation, 0 = not included
            init_scores: predictions from a prior predictor
                that this class will boost on top of.  For regression
                there is 1 prediction per sample.  For binary classification
                there is one logit.  For multiclass there are n_classes logits
            experimental_params: unused data that can be passed into the native layer for debugging

        """

        self.dataset = dataset
        self.bag = bag
        self.init_scores = init_scores
        self.experimental_params = experimental_params

    def __enter__(self):
        log.info("Allocation interaction start")

        native = Native.get_native_singleton()

        n_samples, n_features, n_weights, n_targets = native.extract_dataset_header(self.dataset)

        if n_weights != 0 and n_weights != 1:  # pragma: no cover
            raise ValueError("n_weights must be 0 or 1")

        if n_targets != 1:  # pragma: no cover
            raise ValueError("n_targets must be 1")

        class_counts = native.extract_target_classes(self.dataset, n_targets)
        n_class_scores = sum((Native.get_count_scores_c(n_classes) for n_classes in class_counts))

        n_bagged_samples = n_samples
        if self.bag is not None:
            if self.bag.shape[0] != n_samples:  # pragma: no cover
                raise ValueError("bag should be len(n_samples)")
            n_bagged_samples = np.count_nonzero(self.bag)

        if self.init_scores is not None:
            if self.init_scores.shape[0] != n_bagged_samples:  # pragma: no cover
                raise ValueError("init_scores should have the same length as the number of non-zero bag entries")

            if n_class_scores != 1 and self.init_scores.shape[1] != n_class_scores:  # pragma: no cover
                raise ValueError(f"init_scores should have {n_class_scores} scores")

        # Allocate external resources
        interaction_handle = ct.c_void_p(0)
        return_code = native._unsafe.CreateInteractionDetector(
            Native._make_pointer(self.dataset, np.ubyte),
            Native._make_pointer(self.bag, np.int8, 1, True),
            Native._make_pointer(self.init_scores, np.float64, 2 if 1 < n_class_scores else 1, True),
            Native._make_pointer(self.experimental_params, np.float64, 1, True),
            ct.byref(interaction_handle),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "CreateInteractionDetector")

        self._interaction_handle = interaction_handle.value

        log.info("Allocation interaction end")
        return self

    def __exit__(self, *args):

        self.close()

    def close(self):

        """ Deallocates C objects used to determine interactions in EBM. """
        log.info("Deallocation interaction start")

        interaction_handle = getattr(self, "_interaction_handle", None)
        if interaction_handle:
            native = Native.get_native_singleton()
            self._interaction_handle = None
            native._unsafe.FreeInteractionDetector(interaction_handle)
        
        log.info("Deallocation interaction end")

    def calc_interaction_strength(self, feature_idxs, interaction_flags, min_samples_leaf):
        """ Provides strength for an feature interaction. Higher is better."""
        log.info("Fast interaction strength start")

        native = Native.get_native_singleton()

        strength = ct.c_double(0.0)
        return_code = native._unsafe.CalcInteractionStrength(
            self._interaction_handle,
            len(feature_idxs),
            Native._make_pointer(np.array(feature_idxs, np.int64), np.int64),
            interaction_flags, 
            min_samples_leaf,
            ct.byref(strength),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "CalcInteractionStrength")

        log.info("Fast interaction strength end")
        return strength.value
