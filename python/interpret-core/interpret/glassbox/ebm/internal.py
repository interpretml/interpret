# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

# TODO: Add unit tests for internal EBM interfacing
from sys import platform
import ctypes as ct
import numpy as np
import os
import struct
import logging
from contextlib import AbstractContextManager

log = logging.getLogger(__name__)

class Native:

    # GenerateUpdateOptionsType
    GenerateUpdateOptions_Default               = 0x0000000000000000
    GenerateUpdateOptions_DisableNewtonGain     = 0x0000000000000001
    GenerateUpdateOptions_DisableNewtonUpdate   = 0x0000000000000002
    GenerateUpdateOptions_GradientSums          = 0x0000000000000004
    GenerateUpdateOptions_RandomSplits          = 0x0000000000000008

    # TraceLevel
    _TraceLevelOff = 0
    _TraceLevelError = 1
    _TraceLevelWarning = 2
    _TraceLevelInfo = 3
    _TraceLevelVerbose = 4

    _native = None
    # if we supported win32 32-bit functions then this would need to be WINFUNCTYPE
    _LogFuncType = ct.CFUNCTYPE(None, ct.c_int32, ct.c_char_p)

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

                if trace_level == self._TraceLevelError:
                    log.error(message)
                elif trace_level == self._TraceLevelWarning:
                    log.warning(message)
                elif trace_level == self._TraceLevelInfo:
                    log.info(message)
                elif trace_level == self._TraceLevelVerbose:
                    log.debug(message)
            except:  # pragma: no cover
                # we're being called from C, so we can't raise exceptions
                pass

        if level is None:
            root = logging.getLogger("interpret")
            level = root.getEffectiveLevel()

        level_dict = {
            logging.DEBUG: self._TraceLevelVerbose,
            logging.INFO: self._TraceLevelInfo,
            logging.WARNING: self._TraceLevelWarning,
            logging.ERROR: self._TraceLevelError,
            logging.CRITICAL: self._TraceLevelError,
            logging.NOTSET: self._TraceLevelOff,
            "DEBUG": self._TraceLevelVerbose,
            "INFO": self._TraceLevelInfo,
            "WARNING": self._TraceLevelWarning,
            "ERROR": self._TraceLevelError,
            "CRITICAL": self._TraceLevelError,
            "NOTSET": self._TraceLevelOff,
        }

        trace_level = level_dict[level]
        if self._typed_log_func is None and trace_level != self._TraceLevelOff:
            # it's critical that we put _LogFuncType(native_log) into 
            # self._typed_log_func, otherwise it will be garbage collected
            self._typed_log_func = self._LogFuncType(native_log)
            self._unsafe.SetLogMessageFunction(self._typed_log_func)

        self._unsafe.SetTraceLevel(trace_level)

    def generate_random_number(self, random_seed, stage_randomization_mix):
        return self._unsafe.GenerateRandomNumber(random_seed, stage_randomization_mix)

    def sample_without_replacement(
        self, 
        random_seed, 
        count_training_samples,
        count_validation_samples
    ):
        count_samples = count_training_samples + count_validation_samples
        random_seed = ct.c_int32(random_seed)
        count_training_samples = ct.c_int64(count_training_samples)
        count_validation_samples = ct.c_int64(count_validation_samples)

        sample_counts_out = np.empty(count_samples, dtype=np.int8, order="C")

        self._unsafe.SampleWithoutReplacement(
            random_seed,
            count_training_samples,
            count_validation_samples,
            Native._make_pointer(sample_counts_out, np.int8),
        )

        return sample_counts_out

    def stratified_sampling_without_replacement(
        self, 
        random_seed, 
        count_target_classes,
        count_training_samples,
        count_validation_samples,
        targets
    ):
        count_samples = count_training_samples + count_validation_samples

        if len(targets) != count_samples:
            raise ValueError("count_training_samples + count_validation_samples should be equal to len(targets)")

        random_seed = ct.c_int32(random_seed)
        count_target_classes = ct.c_int64(count_target_classes)
        count_training_samples = ct.c_int64(count_training_samples)
        count_validation_samples = ct.c_int64(count_validation_samples)

        sample_counts_out = np.empty(count_samples, dtype=np.int8, order="C")

        return_code = self._unsafe.StratifiedSamplingWithoutReplacement(
            random_seed,
            count_target_classes,
            count_training_samples,
            count_validation_samples,
            Native._make_pointer(targets, np.int64),
            Native._make_pointer(sample_counts_out, np.int8),
        )

        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "StratifiedSamplingWithoutReplacement")

        return sample_counts_out

    def get_histogram_cut_count(self, col_data, strategy=None):
        n_cuts = self._unsafe.GetHistogramCutCount(
            col_data.shape[0],
            Native._make_pointer(col_data, np.float64),
            0
        )
        return n_cuts

    def cut_quantile(self, col_data, min_samples_bin, is_humanized, max_cuts):
        if max_cuts < 0:
            raise Exception(f"max_cuts can't be negative: {max_cuts}.")

        cuts = np.empty(max_cuts, dtype=np.float64, order="C")
        count_cuts = ct.c_int64(max_cuts)
        return_code = self._unsafe.CutQuantile(
            col_data.shape[0],
            Native._make_pointer(col_data, np.float64),
            min_samples_bin,
            is_humanized,
            ct.byref(count_cuts),
            Native._make_pointer(cuts, np.float64),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "CutQuantile")

        return cuts[:count_cuts.value]

    def cut_uniform(self, col_data, max_cuts):
        if max_cuts < 0:
            raise Exception(f"max_cuts can't be negative: {max_cuts}.")

        cuts = np.empty(max_cuts, dtype=np.float64, order="C")
        count_cuts = ct.c_int64(max_cuts)
        self._unsafe.CutUniform(
            col_data.shape[0],
            Native._make_pointer(col_data, np.float64),
            ct.byref(count_cuts),
            Native._make_pointer(cuts, np.float64),
        )
        return cuts[:count_cuts.value]

    def cut_winsorized(self, col_data, max_cuts):
        if max_cuts < 0:
            raise Exception(f"max_cuts can't be negative: {max_cuts}.")

        cuts = np.empty(max_cuts, dtype=np.float64, order="C")
        count_cuts = ct.c_int64(max_cuts)
        return_code = self._unsafe.CutWinsorized(
            col_data.shape[0],
            Native._make_pointer(col_data, np.float64),
            ct.byref(count_cuts),
            Native._make_pointer(cuts, np.float64),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "CutWinsorized")

        return cuts[:count_cuts.value]

    def suggest_graph_bounds(self, cuts, min_val=np.nan, max_val=np.nan):
        # This function will never return NaN values for min_graph or max_graph
        # It can however return -inf values for min_graph and +inf for max_graph
        # if the min_val or max_val are +-inf or if there's an overflow when
        # extending the bounds.
        # A possible dangerous return value of -inf or +inf for both min_graph and max_graph
        # can occur if one of the min/max values is missing (NaN), or if the dataset consists
        # entirely of +inf or -inf values which makes both min_val and max_val the same
        # extreme value.  In this case the caller should probably check if min_graph == max_graph
        # to avoid subtracting them, which would lead to a NaN value for the difference.
        # Also max_graph - min_graph can be +inf even if both min_graph and max_graph are 
        # normal numbers.  An example of this occuring would be if min was the lowest float
        # and max was the highest float.

        min_graph = ct.c_double(np.nan)
        max_graph = ct.c_double(np.nan)
        return_code = self._unsafe.SuggestGraphBounds(
            len(cuts),
            cuts[0] if 0 < len(cuts) else np.nan,
            cuts[-1] if 0 < len(cuts) else np.nan,
            min_val,
            max_val,
            ct.byref(min_graph),
            ct.byref(max_graph),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "SuggestGraphBounds")

        return min_graph.value, max_graph.value

    def discretize(self, col_data, cuts):
        discretized = np.empty(col_data.shape[0], dtype=np.int64, order="C")
        return_code = self._unsafe.Discretize(
            col_data.shape[0],
            Native._make_pointer(col_data, np.float64),
            cuts.shape[0],
            Native._make_pointer(cuts, np.float64),
            Native._make_pointer(discretized, np.int64),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "Discretize")

        return discretized


    def size_dataset_header(self, n_features, n_weights, n_targets):
        n_bytes = self._unsafe.SizeDataSetHeader(n_features, n_weights, n_targets)
        if n_bytes < 0:  # pragma: no cover
            raise Native._get_native_exception(n_bytes, "SizeDataSetHeader")
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

    def size_feature(self, n_bins, missing, unknown, nominal, binned_data):
        n_bytes = self._unsafe.SizeFeature(
            n_bins, 
            missing, 
            unknown, 
            nominal, 
            len(binned_data), 
            Native._make_pointer(binned_data, np.int64),
        )
        if n_bytes < 0:  # pragma: no cover
            raise Native._get_native_exception(n_bytes, "SizeFeature")
        return n_bytes

    def fill_feature(self, n_bins, missing, unknown, nominal, binned_data, dataset):
        return_code = self._unsafe.FillFeature(
            n_bins, 
            missing, 
            unknown, 
            nominal, 
            len(binned_data), 
            Native._make_pointer(binned_data, np.int64),
            dataset.nbytes, 
            Native._make_pointer(dataset, np.ubyte),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "FillFeature")

    def size_weight(self, weights):
        n_bytes = self._unsafe.SizeWeight(
            len(weights), 
            Native._make_pointer(weights, np.float64),
        )
        if n_bytes < 0:  # pragma: no cover
            raise Native._get_native_exception(n_bytes, "SizeWeight")
        return n_bytes

    def fill_weight(self, weights, dataset):
        return_code = self._unsafe.FillWeight(
            len(weights), 
            Native._make_pointer(weights, np.float64),
            dataset.nbytes, 
            Native._make_pointer(dataset, np.ubyte),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "FillWeight")

    def size_classification_target(self, n_classes, targets):
        n_bytes = self._unsafe.SizeClassificationTarget(
            n_classes, 
            len(targets), 
            Native._make_pointer(targets, np.int64),
        )
        if n_bytes < 0:  # pragma: no cover
            raise Native._get_native_exception(n_bytes, "SizeClassificationTarget")
        return n_bytes

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

    def size_regression_target(self, targets):
        n_bytes = self._unsafe.SizeRegressionTarget(
            len(targets), 
            Native._make_pointer(targets, np.float64),
        )
        if n_bytes < 0:  # pragma: no cover
            raise Native._get_native_exception(n_bytes, "SizeRegressionTarget")
        return n_bytes

    def fill_regression_target(self, targets, dataset):
        return_code = self._unsafe.FillRegressionTarget(
            len(targets), 
            Native._make_pointer(targets, np.float64),
            dataset.nbytes, 
            Native._make_pointer(dataset, np.ubyte),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "FillRegressionTarget")

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


    @staticmethod
    def _get_ebm_lib_path(debug=False):
        """ Returns filepath of core EBM library.

        Returns:
            A string representing filepath.
        """
        bitsize = struct.calcsize("P") * 8
        is_64_bit = bitsize == 64

        script_path = os.path.dirname(os.path.abspath(__file__))
        package_path = os.path.join(script_path, "..", "..")

        debug_str = "_debug" if debug else ""
        log.info("Loading native on {0} | debug = {1}".format(platform, debug))
        if platform == "linux" or platform == "linux2" and is_64_bit:  # pragma: no cover
            return os.path.join(
                package_path, "lib", "lib_ebm_native_linux_x64{0}.so".format(debug_str)
            )
        elif platform == "win32" and is_64_bit:  # pragma: no cover
            return os.path.join(
                package_path, "lib", "lib_ebm_native_win_x64{0}.dll".format(debug_str)
            )
        elif platform == "darwin" and is_64_bit:  # pragma: no cover
            return os.path.join(
                package_path, "lib", "lib_ebm_native_mac_x64{0}.dylib".format(debug_str)
            )
        else:  # pragma: no cover
            msg = "Platform {0} at {1} bit not supported for EBM".format(
                platform, bitsize
            )
            log.error(msg)
            raise Exception(msg)

    def _initialize(self, is_debug):
        self.is_debug = is_debug

        self._typed_log_func = None
        self._unsafe = ct.cdll.LoadLibrary(Native._get_ebm_lib_path(debug=is_debug))

        self._unsafe.SetLogMessageFunction.argtypes = [
            # void (* fn)(int32 traceLevel, const char * message) logMessageFunction
            self._LogFuncType
        ]
        self._unsafe.SetLogMessageFunction.restype = None

        self._unsafe.SetTraceLevel.argtypes = [
            # int32 traceLevel
            ct.c_int32
        ]
        self._unsafe.SetTraceLevel.restype = None


        self._unsafe.GenerateRandomNumber.argtypes = [
            # int32_t randomSeed
            ct.c_int32,
            # int64_t stageRandomizationMix
            ct.c_int32,
        ]
        self._unsafe.GenerateRandomNumber.restype = ct.c_int32

        self._unsafe.SampleWithoutReplacement.argtypes = [
            # int32_t randomSeed
            ct.c_int32,
            # int64_t countTrainingSamples
            ct.c_int64,
            # int64_t countValidationSamples
            ct.c_int64,
            # int8_t * sampleCountsOut
            ct.c_void_p,
        ]
        self._unsafe.SampleWithoutReplacement.restype = None

        self._unsafe.StratifiedSamplingWithoutReplacement.argtypes = [
            # int32_t randomSeed
            ct.c_int32,
            # int64_t countTargetClasses
            ct.c_int64,
            # int64_t countTrainingSamples
            ct.c_int64,
            # int64_t countValidationSamples
            ct.c_int64,
            # int64_t * targets
            ct.c_void_p,
            # int8_t * sampleCountsOut
            ct.c_void_p,
        ]
        self._unsafe.StratifiedSamplingWithoutReplacement.restype = ct.c_int32

        self._unsafe.GetHistogramCutCount.argtypes = [
            # int64_t countSamples
            ct.c_int64,
            # double * featureValues
            ct.c_void_p,
            # int64_t strategy
            ct.c_int64,
        ]
        self._unsafe.GetHistogramCutCount.restype = ct.c_int64

        self._unsafe.CutQuantile.argtypes = [
            # int64_t countSamples
            ct.c_int64,
            # double * featureValues
            ct.c_void_p,
            # int64_t countSamplesPerBinMin
            ct.c_int64,
            # int64_t isHumanized
            ct.c_int64,
            # int64_t * countCutsInOut
            ct.POINTER(ct.c_int64),
            # double * cutsLowerBoundInclusiveOut
            ct.c_void_p,
        ]
        self._unsafe.CutQuantile.restype = ct.c_int32

        self._unsafe.CutUniform.argtypes = [
            # int64_t countSamples
            ct.c_int64,
            # double * featureValues
            ct.c_void_p,
            # int64_t * countCutsInOut
            ct.POINTER(ct.c_int64),
            # double * cutsLowerBoundInclusiveOut
            ct.c_void_p,
        ]
        self._unsafe.CutUniform.restype = None

        self._unsafe.CutWinsorized.argtypes = [
            # int64_t countSamples
            ct.c_int64,
            # double * featureValues
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
            # double minValue
            ct.c_double,
            # double maxValue
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
            # double * featureValues
            ct.c_void_p,
            # int64_t countCuts
            ct.c_int64,
            # double * cutsLowerBoundInclusive
            ct.c_void_p,
            # int64_t * discretizedOut
            ct.c_void_p,
        ]
        self._unsafe.Discretize.restype = ct.c_int32


        self._unsafe.SizeDataSetHeader.argtypes = [
            # int64_t countFeatures
            ct.c_int64,
            # int64_t countWeights
            ct.c_int64,
            # int64_t countTargets
            ct.c_int64,
        ]
        self._unsafe.SizeDataSetHeader.restype = ct.c_int64

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

        self._unsafe.SizeFeature.argtypes = [
            # int64_t countBins
            ct.c_int64,
            # int64_t missing
            ct.c_int64,
            # int64_t unknown
            ct.c_int64,
            # int64_t nominal
            ct.c_int64,
            # int64_t countSamples
            ct.c_int64,
            # int64_t * binnedData
            ct.c_void_p,
        ]
        self._unsafe.SizeFeature.restype = ct.c_int64

        self._unsafe.FillFeature.argtypes = [
            # int64_t countBins
            ct.c_int64,
            # int64_t missing
            ct.c_int64,
            # int64_t unknown
            ct.c_int64,
            # int64_t nominal
            ct.c_int64,
            # int64_t countSamples
            ct.c_int64,
            # int64_t * binnedData
            ct.c_void_p,
            # int64_t countBytesAllocated
            ct.c_int64,
            # void * fillMem
            ct.c_void_p,
        ]
        self._unsafe.FillFeature.restype = ct.c_int32

        self._unsafe.SizeWeight.argtypes = [
            # int64_t countSamples
            ct.c_int64,
            # FloatEbmType * weights
            ct.c_void_p,
        ]
        self._unsafe.SizeWeight.restype = ct.c_int64

        self._unsafe.FillWeight.argtypes = [
            # int64_t countSamples
            ct.c_int64,
            # FloatEbmType * weights
            ct.c_void_p,
            # int64_t countBytesAllocated
            ct.c_int64,
            # void * fillMem
            ct.c_void_p,
        ]
        self._unsafe.FillWeight.restype = ct.c_int32

        self._unsafe.SizeClassificationTarget.argtypes = [
            # int64_t countTargetClasses
            ct.c_int64,
            # int64_t countSamples
            ct.c_int64,
            # int64_t * targets
            ct.c_void_p,
        ]
        self._unsafe.SizeClassificationTarget.restype = ct.c_int64

        self._unsafe.FillClassificationTarget.argtypes = [
            # int64_t countTargetClasses
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

        self._unsafe.SizeRegressionTarget.argtypes = [
            # int64_t countSamples
            ct.c_int64,
            # FloatEbmType * targets
            ct.c_void_p,
        ]
        self._unsafe.SizeRegressionTarget.restype = ct.c_int64

        self._unsafe.FillRegressionTarget.argtypes = [
            # int64_t countSamples
            ct.c_int64,
            # FloatEbmType * targets
            ct.c_void_p,
            # int64_t countBytesAllocated
            ct.c_int64,
            # void * fillMem
            ct.c_void_p,
        ]
        self._unsafe.FillRegressionTarget.restype = ct.c_int32

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


        self._unsafe.Softmax.argtypes = [
            # int64_t countTargetClasses
            ct.c_int64,
            # int64_t countSamples
            ct.c_int64,
            # double * logits
            ct.c_void_p,
            # double * probabilitiesOut
            ct.c_void_p,
        ]
        self._unsafe.Softmax.restype = ct.c_int32


        self._unsafe.CreateBooster.argtypes = [
            # int32_t randomSeed
            ct.c_int32,
            # void * dataSet
            ct.c_void_p,
            # int8_t * bag
            ct.c_void_p,
            # double * predictorScores
            ct.c_void_p,
            # int64_t countFeatureGroups
            ct.c_int64,
            # int64_t * dimensionCounts
            ct.c_void_p,
            # int64_t * featureIndexes
            ct.c_void_p,
            # int64_t countInnerBags
            ct.c_int64,
            # double * optionalTempParams
            ct.c_void_p,
            # BoosterHandle * boosterHandleOut
            ct.POINTER(ct.c_void_p),
        ]
        self._unsafe.CreateBooster.restype = ct.c_int32

        self._unsafe.GenerateModelUpdate.argtypes = [
            # void * boosterHandle
            ct.c_void_p,
            # int64_t indexFeatureGroup
            ct.c_int64,
            # GenerateUpdateOptionsType options 
            ct.c_int64,
            # double learningRate
            ct.c_double,
            # int64_t countSamplesRequiredForChildSplitMin
            ct.c_int64,
            # int64_t * leavesMax
            ct.c_void_p,
            # double * gainOut
            ct.POINTER(ct.c_double),
        ]
        self._unsafe.GenerateModelUpdate.restype = ct.c_int32

        self._unsafe.GetModelUpdateSplits.argtypes = [
            # void * boosterHandle
            ct.c_void_p,
            # int64_t indexDimension
            ct.c_int64,
            # int64_t * countSplitsInOut
            ct.POINTER(ct.c_int64),
            # int64_t * splitIndexesOut
            ct.c_void_p,
        ]
        self._unsafe.GetModelUpdateSplits.restype = ct.c_int32

        self._unsafe.GetModelUpdateExpanded.argtypes = [
            # void * boosterHandle
            ct.c_void_p,
            # double * modelFeatureGroupUpdateTensorOut
            ct.c_void_p,
        ]
        self._unsafe.GetModelUpdateExpanded.restype = ct.c_int32

        self._unsafe.SetModelUpdateExpanded.argtypes = [
            # void * boosterHandle
            ct.c_void_p,
            # int64_t indexFeatureGroup
            ct.c_int64,
            # double * modelFeatureGroupUpdateTensor
            ct.c_void_p,
        ]
        self._unsafe.SetModelUpdateExpanded.restype = ct.c_int32

        self._unsafe.ApplyModelUpdate.argtypes = [
            # void * boosterHandle
            ct.c_void_p,
            # double * validationMetricOut
            ct.POINTER(ct.c_double),
        ]
        self._unsafe.ApplyModelUpdate.restype = ct.c_int32

        self._unsafe.GetBestModelFeatureGroup.argtypes = [
            # void * boosterHandle
            ct.c_void_p,
            # int64_t indexFeatureGroup
            ct.c_int64,
            # double * modelFeatureGroupTensorOut
            ct.c_void_p,
        ]
        self._unsafe.GetBestModelFeatureGroup.restype = ct.c_int32

        self._unsafe.GetCurrentModelFeatureGroup.argtypes = [
            # void * boosterHandle
            ct.c_void_p,
            # int64_t indexFeatureGroup
            ct.c_int64,
            # double * modelFeatureGroupTensorOut
            ct.c_void_p,
        ]
        self._unsafe.GetCurrentModelFeatureGroup.restype = ct.c_int32

        self._unsafe.FreeBooster.argtypes = [
            # void * boosterHandle
            ct.c_void_p
        ]
        self._unsafe.FreeBooster.restype = None


        self._unsafe.CreateInteractionDetector.argtypes = [
            # void * dataSet
            ct.c_void_p,
            # int8_t * bag
            ct.c_void_p,
            # double * predictorScores
            ct.c_void_p,
            # double * optionalTempParams
            ct.c_void_p,
            # InteractionHandle * interactionHandleOut
            ct.POINTER(ct.c_void_p),
        ]
        self._unsafe.CreateInteractionDetector.restype = ct.c_int32

        self._unsafe.CalculateInteractionScore.argtypes = [
            # void * interactionHandle
            ct.c_void_p,
            # int64_t countDimensions
            ct.c_int64,
            # int64_t * featureIndexes
            ct.c_void_p,
            # int64_t countSamplesRequiredForChildSplitMin
            ct.c_int64,
            # double * interactionScoreOut
            ct.POINTER(ct.c_double),
        ]
        self._unsafe.CalculateInteractionScore.restype = ct.c_int32

        self._unsafe.FreeInteractionDetector.argtypes = [
            # void * interactionHandle
            ct.c_void_p
        ]
        self._unsafe.FreeInteractionDetector.restype = None

class Booster(AbstractContextManager):
    """Lightweight wrapper for EBM C boosting code.
    """

    def __init__(
        self,
        dataset,
        bag,
        scores,
        term_features,
        n_inner_bags,
        random_state,
        optional_temp_params,
    ):

        """ Initializes internal wrapper for EBM C code.

        Args:
            dataset: binned data in a compressed native form
            bag: definition of what data is included. 1 = training, -1 = validation, 0 = not included
            scores: predictions from a prior predictor
                that this class will boost on top of.  For regression
                there is 1 prediction per sample.  For binary classification
                there is one logit.  For multiclass there are n_classes logits
            term_features: List of term feature indexes
            n_inner_bags: number of inner bags.
            random_state: Random seed as integer.
            optional_temp_params: unused data that can be passed into the native layer for debugging
        """

        self.dataset = dataset
        self.bag = bag
        self.scores = scores
        self.term_features = term_features
        self.n_inner_bags = n_inner_bags
        self.random_state = random_state
        self.optional_temp_params = optional_temp_params

        # start off with an invalid _term_idx
        self._term_idx = -1

    def __enter__(self):
        log.info("Booster allocation start")

        feature_counts = np.empty(len(self.term_features), ct.c_int64)
        feature_indexes = []
        for term_idx, feature_idxs in enumerate(self.term_features):
            feature_counts.itemset(term_idx, len(feature_idxs))
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
                if n_class_scores > 1:
                    dimensions.append(n_class_scores)

                self._term_shapes.append(tuple(dimensions))

        n_scores = n_samples
        if self.bag is not None:
            if self.bag.shape[0] != n_samples:  # pragma: no cover
                raise ValueError("bag should be len(n_samples)")
            n_scores = np.count_nonzero(self.bag)

        if self.scores is not None:
            if n_class_scores > 1:
                if self.scores.shape[1] != n_class_scores:  # pragma: no cover
                    raise ValueError(f"scores should have {n_class_scores} scores")

            if self.scores.shape[0] != n_scores:  # pragma: no cover
                raise ValueError("scores should have the same length as the number of non-zero bag entries")

        # Allocate external resources
        booster_handle = ct.c_void_p(0)
        return_code = native._unsafe.CreateBooster(
            self.random_state,
            Native._make_pointer(self.dataset, np.ubyte),
            Native._make_pointer(self.bag, np.int8, 1, True),
            Native._make_pointer(self.scores, np.float64, 2 if n_class_scores > 1 else 1, True),
            len(feature_counts),
            Native._make_pointer(feature_counts, np.int64),
            Native._make_pointer(feature_indexes, np.int64),
            self.n_inner_bags,
            Native._make_pointer(self.optional_temp_params, np.float64, 1, True),
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
        generate_update_options, 
        learning_rate, 
        min_samples_leaf, 
        max_leaves, 
    ):

        """ Generates a boosting step update per feature
            by growing a shallow decision tree.

        Args:
            term_idx: The index for the term to generate the update for
            generate_update_options: C interface options
            learning_rate: Learning rate as a float.
            min_samples_leaf: Min observations required to split.
            max_leaves: Max leaf nodes on feature step.

        Returns:
            gain for the generated boosting step.
        """

        # log.debug("Boosting step start")

        self._term_idx = -1

        native = Native.get_native_singleton()

        gain = ct.c_double(0.0)
        n_features = len(self.term_features[term_idx])
        max_leaves_arr = np.full(n_features, max_leaves, dtype=ct.c_int64, order="C")

        return_code = native._unsafe.GenerateModelUpdate(
            self._booster_handle, 
            term_idx,
            generate_update_options,
            learning_rate,
            min_samples_leaf,
            Native._make_pointer(max_leaves_arr, np.int64),
            ct.byref(gain),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "GenerateModelUpdate")
            
        self._term_idx = term_idx

        # log.debug("Boosting step end")
        return gain.value

    def apply_term_update(self):

        """ Updates the interal C state with the last model update

        Args:

        Returns:
            Validation loss for the boosting step.
        """
        # log.debug("Boosting step start")

        self._term_idx = -1

        native = Native.get_native_singleton()

        metric_output = ct.c_double(0.0)
        return_code = native._unsafe.ApplyModelUpdate(
            self._booster_handle, 
            ct.byref(metric_output),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "ApplyModelUpdate")

        # log.debug("Boosting step end")
        return metric_output.value

    def get_best_model(self):
        model = []
        for term_idx in range(len(self.term_features)):
            model_term = self._get_best_term(term_idx)
            model.append(model_term)

        return model

    # TODO: Needs test.
    def get_current_model(self):
        model = []
        for term_idx in range(len(self.term_features)):
            model_term = self._get_current_term(term_idx)
            model.append(model_term)

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

    def _get_best_term(self, term_idx):
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
        model_term = np.empty(shape, dtype=np.float64, order="C")

        return_code = native._unsafe.GetBestModelFeatureGroup(
            self._booster_handle, 
            term_idx, 
            Native._make_pointer(model_term, np.float64, len(shape)),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "GetBestModelFeatureGroup")

        n_dimensions = len(self.term_features[term_idx])
        temp_transpose = [*range(n_dimensions - 1, -1, -1)]
        if len(shape) != n_dimensions: # multiclass
            temp_transpose.append(len(temp_transpose))
        model_term = np.ascontiguousarray(np.transpose(model_term, tuple(temp_transpose)))
        return model_term

    def _get_current_term(self, term_idx):
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
        model_term = np.empty(shape, dtype=np.float64, order="C")

        return_code = native._unsafe.GetCurrentModelFeatureGroup(
            self._booster_handle, 
            term_idx, 
            Native._make_pointer(model_term, np.float64, len(shape)),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "GetCurrentModelFeatureGroup")

        n_dimensions = len(self.term_features[term_idx])
        temp_transpose = [*range(n_dimensions - 1, -1, -1)]
        if len(shape) != n_dimensions: # multiclass
            temp_transpose.append(len(temp_transpose))
        model_term = np.ascontiguousarray(np.transpose(model_term, tuple(temp_transpose)))
        return model_term

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

        return_code = native._unsafe.GetModelUpdateSplits(
            self._booster_handle, 
            dimension_index, 
            ct.byref(count_splits), 
            Native._make_pointer(splits, np.int64),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "GetModelUpdateSplits")

        splits = splits[:count_splits.value]
        return splits

    def get_term_update_expanded(self):
        if self._term_idx < 0:  # pragma: no cover
            raise RuntimeError("invalid internal self._term_idx")

        if self._term_shapes is None:  # pragma: no cover
            # if there is only one legal state for a classification problem, then we know with 100%
            # certainty what the result will be, and our model has no information since we always predict
            # the only output
            return None

        native = Native.get_native_singleton()

        shape = self._term_shapes[self._term_idx]
        term_update = np.empty(shape, dtype=np.float64, order="C")

        return_code = native._unsafe.GetModelUpdateExpanded(
            self._booster_handle, 
            Native._make_pointer(term_update, np.float64, len(shape)),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "GetModelUpdateExpanded")

        n_dimensions = len(self.term_features[self._term_idx])
        temp_transpose = [*range(n_dimensions - 1, -1, -1)]
        if len(shape) != n_dimensions: # multiclass
            temp_transpose.append(len(temp_transpose))
        term_update = np.ascontiguousarray(np.transpose(term_update, tuple(temp_transpose)))
        return term_update

    def set_term_update_expanded(self, term_idx, term_update):
        self._term_idx = -1

        if self._term_shapes is None:  # pragma: no cover
            if term_update is None:  # pragma: no cover
                self._term_idx = term_idx
                return
            raise ValueError("a tensor with 1 class or less would be empty since the predictions would always be the same")

        shape = self._term_shapes[term_idx]

        n_dimensions = len(self.term_features[term_idx])
        temp_transpose = [*range(n_dimensions - 1, -1, -1)]
        if len(shape) != n_dimensions: # multiclass
            temp_transpose.append(len(temp_transpose))
        term_update = np.ascontiguousarray(np.transpose(term_update, tuple(temp_transpose)))

        if shape != term_update.shape:  # pragma: no cover
            raise ValueError("incorrect tensor shape in call to set_term_update_expanded")

        native = Native.get_native_singleton()
        return_code = native._unsafe.SetModelUpdateExpanded(
            self._booster_handle, 
            term_idx, 
            Native._make_pointer(term_update, np.float64, len(shape)),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "SetModelUpdateExpanded")

        self._term_idx = term_idx

        return


class InteractionDetector(AbstractContextManager):
    """Lightweight wrapper for EBM C interaction code.
    """

    def __init__(
        self, 
        dataset,
        bag,
        scores,
        optional_temp_params,
    ):

        """ Initializes internal wrapper for EBM C code.

        Args:
            dataset: binned data in a compressed native form
            bag: definition of what data is included. 1 = training, -1 = validation, 0 = not included
            scores: predictions from a prior predictor
                that this class will boost on top of.  For regression
                there is 1 prediction per sample.  For binary classification
                there is one logit.  For multiclass there are n_classes logits
            optional_temp_params: unused data that can be passed into the native layer for debugging

        """

        self.dataset = dataset
        self.bag = bag
        self.scores = scores
        self.optional_temp_params = optional_temp_params

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

        n_scores = n_samples
        if self.bag is not None:
            if self.bag.shape[0] != n_samples:  # pragma: no cover
                raise ValueError("bag should be len(n_samples)")
            n_scores = np.count_nonzero(self.bag)

        if self.scores is not None:
            if n_class_scores > 1:
                if self.scores.shape[1] != n_class_scores:  # pragma: no cover
                    raise ValueError(f"scores should have {n_class_scores} scores")

            if self.scores.shape[0] != n_scores:  # pragma: no cover
                raise ValueError("scores should have the same length as the number of non-zero bag entries")

        # Allocate external resources
        interaction_handle = ct.c_void_p(0)
        return_code = native._unsafe.CreateInteractionDetector(
            Native._make_pointer(self.dataset, np.ubyte),
            Native._make_pointer(self.bag, np.int8, 1, True),
            Native._make_pointer(self.scores, np.float64, 2 if n_class_scores > 1 else 1, True),
            Native._make_pointer(self.optional_temp_params, np.float64, 1, True),
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

    def get_interaction_score(self, feature_idxs, min_samples_leaf):
        """ Provides score for an feature interaction. Higher is better."""
        log.info("Fast interaction score start")

        native = Native.get_native_singleton()

        score = ct.c_double(0.0)
        return_code = native._unsafe.CalculateInteractionScore(
            self._interaction_handle,
            len(feature_idxs),
            Native._make_pointer(np.array(feature_idxs, np.int64), np.int64),
            min_samples_leaf,
            ct.byref(score),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "CalculateInteractionScore")

        log.info("Fast interaction score end")
        return score.value
