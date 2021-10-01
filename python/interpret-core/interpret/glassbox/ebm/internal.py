# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

# TODO: Add unit tests for internal EBM interfacing
from sys import platform
import ctypes as ct
from numpy.ctypeslib import ndpointer
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
        return 1 if n_classes <= 2 else n_classes

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

        sample_counts_out = np.empty(count_samples, dtype=np.int64, order="C")

        self._unsafe.SampleWithoutReplacement(
            random_seed,
            count_training_samples,
            count_validation_samples,
            sample_counts_out
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

        sample_counts_out = np.empty(count_samples, dtype=np.int64, order="C")

        return_code = self._unsafe.StratifiedSamplingWithoutReplacement(
            random_seed,
            count_target_classes,
            count_training_samples,
            count_validation_samples,
            targets,
            sample_counts_out
        )

        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "StratifiedSamplingWithoutReplacement")

        return sample_counts_out

    def get_histogram_cut_count(self, col_data, strategy=None):
        n_cuts = self._unsafe.GetHistogramCutCount(
            col_data.shape[0],
            col_data, 
            0
        )
        return n_cuts

    def cut_quantile(self, col_data, min_samples_bin, is_humanized, max_cuts):
        cuts = np.empty(max_cuts, dtype=np.float64, order="C")
        count_cuts = ct.c_int64(max_cuts)
        return_code = self._unsafe.CutQuantile(
            col_data.shape[0],
            col_data, 
            min_samples_bin,
            is_humanized,
            ct.byref(count_cuts),
            cuts
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "CutQuantile")

        return cuts[:count_cuts.value]

    def cut_uniform(self, col_data, max_cuts):
        cuts = np.empty(max_cuts, dtype=np.float64, order="C")
        count_cuts = ct.c_int64(max_cuts)
        self._unsafe.CutUniform(
            col_data.shape[0],
            col_data, 
            ct.byref(count_cuts),
            cuts
        )
        return cuts[:count_cuts.value]

    def cut_winsorized(self, col_data, max_cuts):
        cuts = np.empty(max_cuts, dtype=np.float64, order="C")
        count_cuts = ct.c_int64(max_cuts)
        return_code = self._unsafe.CutWinsorized(
            col_data.shape[0],
            col_data, 
            ct.byref(count_cuts),
            cuts
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "CutWinsorized")

        return cuts[:count_cuts.value]

    def suggest_graph_bounds(self, cuts, min_val=np.nan, max_val=np.nan):
        low_graph_bound = ct.c_double(0)
        high_graph_bound = ct.c_double(0)
        return_code = self._unsafe.SuggestGraphBounds(
            len(cuts),
            cuts[0] if 0 < len(cuts) else np.nan,
            cuts[-1] if 0 < len(cuts) else np.nan,
            min_val,
            max_val,
            ct.byref(low_graph_bound),
            ct.byref(high_graph_bound),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "SuggestGraphBounds")

        return low_graph_bound.value, high_graph_bound.value

    def discretize(self, col_data, cuts):
        discretized = np.empty(col_data.shape[0], dtype=np.int64, order="C")
        return_code = self._unsafe.Discretize(
            col_data.shape[0],
            col_data,
            cuts.shape[0],
            cuts,
            discretized
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "Discretize")

        return discretized


    def size_data_set_header(self, n_features, n_weights, n_targets):
        n_bytes = self._unsafe.SizeDataSetHeader(n_features, n_weights, n_targets)
        if n_bytes < 0:  # pragma: no cover
            raise Native._get_native_exception(n_bytes, "SizeDataSetHeader")
        return n_bytes

    def fill_data_set_header(self, n_features, n_weights, n_targets, n_bytes, shared_data):
        return_code = self._unsafe.FillDataSetHeader(
            n_features, 
            n_weights, 
            n_targets, 
            n_bytes, 
            shared_data, 
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "FillDataSetHeader")

    def size_feature(self, categorical, n_bins, binned_data):
        n_bytes = self._unsafe.SizeFeature(categorical, n_bins, len(binned_data), binned_data)
        if n_bytes < 0:  # pragma: no cover
            raise Native._get_native_exception(n_bytes, "SizeFeature")
        return n_bytes

    def fill_feature(self, categorical, n_bins, binned_data, n_bytes, shared_data):
        return_code = self._unsafe.FillFeature(
            categorical, 
            n_bins, 
            len(binned_data), 
            binned_data, 
            n_bytes, 
            shared_data,
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "FillFeature")

    def size_weight(self, weights):
        n_bytes = self._unsafe.SizeWeight(len(weights), weights)
        if n_bytes < 0:  # pragma: no cover
            raise Native._get_native_exception(n_bytes, "SizeWeight")
        return n_bytes

    def fill_weight(self, weights, n_bytes, shared_data):
        return_code = self._unsafe.FillWeight(
            len(weights), 
            weights, 
            n_bytes, 
            shared_data, 
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "FillWeight")

    def size_classification_target(self, n_classes, targets):
        n_bytes = self._unsafe.SizeClassificationTarget(n_classes, len(targets), targets)
        if n_bytes < 0:  # pragma: no cover
            raise Native._get_native_exception(n_bytes, "SizeClassificationTarget")
        return n_bytes

    def fill_classification_target(self, n_classes, targets, n_bytes, shared_data):
        return_code = self._unsafe.FillClassificationTarget(
            n_classes, 
            len(targets), 
            targets, 
            n_bytes, 
            shared_data,
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "FillClassificationTarget")

    def size_regression_target(self, targets):
        n_bytes = self._unsafe.SizeRegressionTarget(len(targets), targets)
        if n_bytes < 0:  # pragma: no cover
            raise Native._get_native_exception(n_bytes, "SizeRegressionTarget")
        return n_bytes

    def fill_regression_target(self, targets, n_bytes, shared_data):
        return_code = self._unsafe.FillRegressionTarget(
            len(targets), 
            targets, 
            n_bytes, 
            shared_data, 
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "FillRegressionTarget")


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
            # int64_t * sampleCountsOut
            ndpointer(dtype=ct.c_int64, ndim=1, flags="C_CONTIGUOUS"),
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
            ndpointer(dtype=ct.c_int64, ndim=1),
            # int64_t * sampleCountsOut
            ndpointer(dtype=ct.c_int64, ndim=1, flags="C_CONTIGUOUS"),
        ]
        self._unsafe.StratifiedSamplingWithoutReplacement.restype = ct.c_int32

        self._unsafe.GetHistogramCutCount.argtypes = [
            # int64_t countSamples
            ct.c_int64,
            # double * featureValues
            ndpointer(dtype=ct.c_double, ndim=1, flags="C_CONTIGUOUS"),
            # int64_t strategy
            ct.c_int64,
        ]
        self._unsafe.GetHistogramCutCount.restype = ct.c_int64

        self._unsafe.CutQuantile.argtypes = [
            # int64_t countSamples
            ct.c_int64,
            # double * featureValues
            ndpointer(dtype=ct.c_double, ndim=1, flags="C_CONTIGUOUS"),
            # int64_t countSamplesPerBinMin
            ct.c_int64,
            # int64_t isHumanized
            ct.c_int64,
            # int64_t * countCutsInOut
            ct.POINTER(ct.c_int64),
            # double * cutsLowerBoundInclusiveOut
            ndpointer(dtype=ct.c_double, ndim=1, flags="C_CONTIGUOUS"),
        ]
        self._unsafe.CutQuantile.restype = ct.c_int32

        self._unsafe.CutUniform.argtypes = [
            # int64_t countSamples
            ct.c_int64,
            # double * featureValues
            ndpointer(dtype=ct.c_double, ndim=1, flags="C_CONTIGUOUS"),
            # int64_t * countCutsInOut
            ct.POINTER(ct.c_int64),
            # double * cutsLowerBoundInclusiveOut
            ndpointer(dtype=ct.c_double, ndim=1, flags="C_CONTIGUOUS"),
        ]
        self._unsafe.CutUniform.restype = None

        self._unsafe.CutWinsorized.argtypes = [
            # int64_t countSamples
            ct.c_int64,
            # double * featureValues
            ndpointer(dtype=ct.c_double, ndim=1, flags="C_CONTIGUOUS"),
            # int64_t * countCutsInOut
            ct.POINTER(ct.c_int64),
            # double * cutsLowerBoundInclusiveOut
            ndpointer(dtype=ct.c_double, ndim=1, flags="C_CONTIGUOUS"),
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
            ndpointer(dtype=ct.c_double, ndim=1, flags="C_CONTIGUOUS"),
            # int64_t countCuts
            ct.c_int64,
            # double * cutsLowerBoundInclusive
            ndpointer(dtype=ct.c_double, ndim=1, flags="C_CONTIGUOUS"),
            # int64_t * discretizedOut
            ndpointer(dtype=ct.c_int64, ndim=1, flags="C_CONTIGUOUS"),
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
            # int64_t categorical
            ct.c_int64,
            # int64_t countBins
            ct.c_int64,
            # int64_t countSamples
            ct.c_int64,
            # int64_t * binnedData
            ndpointer(dtype=ct.c_int64, ndim=1),
        ]
        self._unsafe.SizeFeature.restype = ct.c_int64

        self._unsafe.FillFeature.argtypes = [
            # int64_t categorical
            ct.c_int64,
            # int64_t countBins
            ct.c_int64,
            # int64_t countSamples
            ct.c_int64,
            # int64_t * binnedData
            ndpointer(dtype=ct.c_int64, ndim=1),
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
            ndpointer(dtype=ct.c_double, ndim=1),
        ]
        self._unsafe.SizeWeight.restype = ct.c_int64

        self._unsafe.FillWeight.argtypes = [
            # int64_t countSamples
            ct.c_int64,
            # FloatEbmType * weights
            ndpointer(dtype=ct.c_double, ndim=1),
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
            ndpointer(dtype=ct.c_int64, ndim=1),
        ]
        self._unsafe.SizeClassificationTarget.restype = ct.c_int64

        self._unsafe.FillClassificationTarget.argtypes = [
            # int64_t countTargetClasses
            ct.c_int64,
            # int64_t countSamples
            ct.c_int64,
            # int64_t * targets
            ndpointer(dtype=ct.c_int64, ndim=1),
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
            ndpointer(dtype=ct.c_double, ndim=1),
        ]
        self._unsafe.SizeRegressionTarget.restype = ct.c_int64

        self._unsafe.FillRegressionTarget.argtypes = [
            # int64_t countSamples
            ct.c_int64,
            # FloatEbmType * targets
            ndpointer(dtype=ct.c_double, ndim=1),
            # int64_t countBytesAllocated
            ct.c_int64,
            # void * fillMem
            ct.c_void_p,
        ]
        self._unsafe.FillRegressionTarget.restype = ct.c_int32


        self._unsafe.Softmax.argtypes = [
            # int64_t countTargetClasses
            ct.c_int64,
            # int64_t countSamples
            ct.c_int64,
            # double * logits
            ndpointer(dtype=ct.c_double, ndim=1, flags="C_CONTIGUOUS"),
            # double * probabilitiesOut
            ndpointer(dtype=ct.c_double, ndim=1, flags="C_CONTIGUOUS"),
        ]
        self._unsafe.Softmax.restype = ct.c_int32


        self._unsafe.CreateClassificationBooster.argtypes = [
            # int32_t randomSeed
            ct.c_int32,
            # int64_t countTargetClasses
            ct.c_int64,
            # int64_t countFeatures
            ct.c_int64,
            # int64_t * featuresCategorical
            ndpointer(dtype=ct.c_int64, ndim=1),
            # int64_t * featuresBinCount
            ndpointer(dtype=ct.c_int64, ndim=1),
            # int64_t countFeatureGroups
            ct.c_int64,
            # int64_t * featureGroupsDimensionCount
            ndpointer(dtype=ct.c_int64, ndim=1),
            # int64_t * featureGroupsFeatureIndexes
            ndpointer(dtype=ct.c_int64, ndim=1),
            # int64_t countTrainingSamples
            ct.c_int64,
            # int64_t * trainingBinnedData
            ndpointer(dtype=ct.c_int64, ndim=2, flags="C_CONTIGUOUS"),
            # int64_t * trainingTargets
            ndpointer(dtype=ct.c_int64, ndim=1),
            # double * trainingWeights
            ndpointer(dtype=ct.c_double, ndim=1),
            # ct.c_void_p,
            # double * trainingPredictorScores
            # scores can either be 1 or 2 dimensional
            ndpointer(dtype=ct.c_double, flags="C_CONTIGUOUS"),
            # int64_t countValidationSamples
            ct.c_int64,
            # int64_t * validationBinnedData
            ndpointer(dtype=ct.c_int64, ndim=2, flags="C_CONTIGUOUS"),
            # int64_t * validationTargets
            ndpointer(dtype=ct.c_int64, ndim=1),
            # double * validationWeights
            ndpointer(dtype=ct.c_double, ndim=1),
            # ct.c_void_p,
            # double * validationPredictorScores
            # scores can either be 1 or 2 dimensional
            ndpointer(dtype=ct.c_double, flags="C_CONTIGUOUS"),
            # int64_t countInnerBags
            ct.c_int64,
            # double * optionalTempParams
            ct.POINTER(ct.c_double),
            # BoosterHandle * boosterHandleOut
            ct.POINTER(ct.c_void_p),
        ]
        self._unsafe.CreateClassificationBooster.restype = ct.c_int32

        self._unsafe.CreateRegressionBooster.argtypes = [
            # int32_t randomSeed
            ct.c_int32,
            # int64_t countFeatures
            ct.c_int64,
            # int64_t * featuresCategorical
            ndpointer(dtype=ct.c_int64, ndim=1),
            # int64_t * featuresBinCount
            ndpointer(dtype=ct.c_int64, ndim=1),
            # int64_t countFeatureGroups
            ct.c_int64,
            # int64_t * featureGroupsDimensionCount
            ndpointer(dtype=ct.c_int64, ndim=1),
            # int64_t * featureGroupsFeatureIndexes
            ndpointer(dtype=ct.c_int64, ndim=1),
            # int64_t countTrainingSamples
            ct.c_int64,
            # int64_t * trainingBinnedData
            ndpointer(dtype=ct.c_int64, ndim=2, flags="C_CONTIGUOUS"),
            # double * trainingTargets
            ndpointer(dtype=ct.c_double, ndim=1),
            # double * trainingWeights
            ndpointer(dtype=ct.c_double, ndim=1),
            # ct.c_void_p,
            # double * trainingPredictorScores
            ndpointer(dtype=ct.c_double, ndim=1),
            # int64_t countValidationSamples
            ct.c_int64,
            # int64_t * validationBinnedData
            ndpointer(dtype=ct.c_int64, ndim=2, flags="C_CONTIGUOUS"),
            # double * validationTargets
            ndpointer(dtype=ct.c_double, ndim=1),
            # double * validationWeights
            ndpointer(dtype=ct.c_double, ndim=1),
            # ct.c_void_p,
            # double * validationPredictorScores
            ndpointer(dtype=ct.c_double, ndim=1),
            # int64_t countInnerBags
            ct.c_int64,
            # double * optionalTempParams
            ct.POINTER(ct.c_double),
            # BoosterHandle * boosterHandleOut
            ct.POINTER(ct.c_void_p),
        ]
        self._unsafe.CreateRegressionBooster.restype = ct.c_int32

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
            ndpointer(dtype=ct.c_int64, ndim=1),
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
            ndpointer(dtype=ct.c_int64, ndim=1),
        ]
        self._unsafe.GetModelUpdateSplits.restype = ct.c_int32

        self._unsafe.GetModelUpdateExpanded.argtypes = [
            # void * boosterHandle
            ct.c_void_p,
            # double * modelFeatureGroupUpdateTensorOut
            ndpointer(dtype=ct.c_double, flags="C_CONTIGUOUS"),
        ]
        self._unsafe.GetModelUpdateExpanded.restype = ct.c_int32

        self._unsafe.SetModelUpdateExpanded.argtypes = [
            # void * boosterHandle
            ct.c_void_p,
            # int64_t indexFeatureGroup
            ct.c_int64,
            # double * modelFeatureGroupUpdateTensor
            ndpointer(dtype=ct.c_double, flags="C_CONTIGUOUS"),
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
            ndpointer(dtype=ct.c_double, flags="C_CONTIGUOUS"),
        ]
        self._unsafe.GetBestModelFeatureGroup.restype = ct.c_int32

        self._unsafe.GetCurrentModelFeatureGroup.argtypes = [
            # void * boosterHandle
            ct.c_void_p,
            # int64_t indexFeatureGroup
            ct.c_int64,
            # double * modelFeatureGroupTensorOut
            ndpointer(dtype=ct.c_double, flags="C_CONTIGUOUS"),
        ]
        self._unsafe.GetCurrentModelFeatureGroup.restype = ct.c_int32

        self._unsafe.FreeBooster.argtypes = [
            # void * boosterHandle
            ct.c_void_p
        ]
        self._unsafe.FreeBooster.restype = None


        self._unsafe.CreateClassificationInteractionDetector.argtypes = [
            # int64_t countTargetClasses
            ct.c_int64,
            # int64_t countFeatures
            ct.c_int64,
            # int64_t * featuresCategorical
            ndpointer(dtype=ct.c_int64, ndim=1),
            # int64_t * featuresBinCount
            ndpointer(dtype=ct.c_int64, ndim=1),
            # int64_t countSamples
            ct.c_int64,
            # int64_t * binnedData
            ndpointer(dtype=ct.c_int64, ndim=2, flags="C_CONTIGUOUS"),
            # int64_t * targets
            ndpointer(dtype=ct.c_int64, ndim=1),
            # double * weights
            ndpointer(dtype=ct.c_double, ndim=1),
            # ct.c_void_p,
            # double * predictorScores
            # scores can either be 1 or 2 dimensional
            ndpointer(dtype=ct.c_double, flags="C_CONTIGUOUS"),
            # double * optionalTempParams
            ct.POINTER(ct.c_double),
            # InteractionHandle * interactionHandleOut
            ct.POINTER(ct.c_void_p),
        ]
        self._unsafe.CreateClassificationInteractionDetector.restype = ct.c_int32

        self._unsafe.CreateRegressionInteractionDetector.argtypes = [
            # int64_t countFeatures
            ct.c_int64,
            # int64_t * featuresCategorical
            ndpointer(dtype=ct.c_int64, ndim=1),
            # int64_t * featuresBinCount
            ndpointer(dtype=ct.c_int64, ndim=1),
            # int64_t countSamples
            ct.c_int64,
            # int64_t * binnedData
            ndpointer(dtype=ct.c_int64, ndim=2, flags="C_CONTIGUOUS"),
            # double * targets
            ndpointer(dtype=ct.c_double, ndim=1),
            # double * weights
            ndpointer(dtype=ct.c_double, ndim=1),
            # ct.c_void_p,
            # double * predictorScores
            ndpointer(dtype=ct.c_double, ndim=1),
            # double * optionalTempParams
            ct.POINTER(ct.c_double),
            # InteractionHandle * interactionHandleOut
            ct.POINTER(ct.c_void_p),
        ]
        self._unsafe.CreateRegressionInteractionDetector.restype = ct.c_int32

        self._unsafe.CalculateInteractionScore.argtypes = [
            # void * interactionHandle
            ct.c_void_p,
            # int64_t countDimensions
            ct.c_int64,
            # int64_t * featureIndexes
            ndpointer(dtype=ct.c_int64, ndim=1),
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

    @staticmethod
    def _convert_feature_groups_to_c(feature_groups):
        # Create C form of feature_groups

        feature_groups_feature_count = np.empty(len(feature_groups), dtype=ct.c_int64, order='C')
        feature_groups_feature_indexes = []
        for idx, features_in_group in enumerate(feature_groups):
            feature_groups_feature_count[idx] = len(features_in_group)
            for feature_idx in features_in_group:
                feature_groups_feature_indexes.append(feature_idx)

        feature_groups_feature_indexes = np.array(feature_groups_feature_indexes, dtype=ct.c_int64)

        return feature_groups_feature_count, feature_groups_feature_indexes


class Booster(AbstractContextManager):
    """Lightweight wrapper for EBM C boosting code.
    """

    def __init__(
        self,
        model_type,
        n_classes,
        features_categorical, 
        features_bin_count,
        feature_groups,
        X_train,
        y_train,
        w_train,
        scores_train,
        X_val,
        y_val,
        w_val,
        scores_val,
        n_inner_bags,
        random_state,
        optional_temp_params,
    ):

        """ Initializes internal wrapper for EBM C code.

        Args:
            model_type: 'regression'/'classification'.
            n_classes: Specific to classification,
                number of unique classes.
            features_categorical: list of categorical features represented by bools 
            features_bin_count: count of the number of bins for each feature
            feature_groups: List of feature groups represented as
                a dictionary of keys ("features")
            X_train: Training design matrix as 2-D ndarray.
            y_train: Training response as 1-D ndarray.
            scores_train: training predictions from a prior predictor
                that this class will boost on top of.  For regression
                there is 1 prediction per sample.  For binary classification
                there is one logit.  For multiclass there are n_classes logits
            X_val: Validation design matrix as 2-D ndarray.
            y_val: Validation response as 1-D ndarray.
            scores_val: Validation predictions from a prior predictor
                that this class will boost on top of.  For regression
                there is 1 prediction per sample.  For binary classification
                there is one logit.  For multiclass there are n_classes logits
            n_inner_bags: number of inner bags.
            random_state: Random seed as integer.
        """

        self.model_type = model_type
        self.n_classes = n_classes
        self.features_categorical = features_categorical
        self.features_bin_count = features_bin_count
        self.feature_groups = feature_groups
        self.X_train = X_train
        self.y_train = y_train
        self.w_train = w_train
        self.scores_train = scores_train
        self.X_val = X_val
        self.y_val = y_val
        self.w_val = w_val
        self.scores_val = scores_val
        self.n_inner_bags = n_inner_bags
        self.random_state = random_state
        self.optional_temp_params = optional_temp_params

        # start off with an invalid _feature_group_index
        self._feature_group_index = -1

    def __enter__(self):

        # check inputs for important inputs or things that would segfault in C
        if not isinstance(self.features_categorical, np.ndarray):  # pragma: no cover
            raise ValueError("features_categorical should be an np.ndarray")

        if not isinstance(self.features_bin_count, np.ndarray):  # pragma: no cover
            raise ValueError("features_bin_count should be an np.ndarray")

        if not isinstance(self.feature_groups, list):  # pragma: no cover
            raise ValueError("feature_groups should be a list")

        if self.X_train.ndim != 2:  # pragma: no cover
            raise ValueError("X_train should have exactly 2 dimensions")

        if self.y_train.ndim != 1:  # pragma: no cover
            raise ValueError("y_train should have exactly 1 dimension")

        if self.X_train.shape[0] != len(self.features_categorical):  # pragma: no cover
            raise ValueError(
                "X_train does not have the same number of items as the features_categorical array"
            )

        if self.X_train.shape[0] != len(self.features_bin_count):  # pragma: no cover
            raise ValueError(
                "X_train does not have the same number of items as the features_bin_count array"
            )

        if self.X_train.shape[1] != len(self.y_train):  # pragma: no cover
            raise ValueError(
                "X_train does not have the same number of samples as y_train"
            )

        if self.X_val.ndim != 2:  # pragma: no cover
            raise ValueError("X_val should have exactly 2 dimensions")

        if self.y_val.ndim != 1:  # pragma: no cover
            raise ValueError("y_val should have exactly 1 dimension")

        if self.X_val.shape[0] != self.X_train.shape[0]:  # pragma: no cover
            raise ValueError(
                "X_val does not have the same number of features as the X_train array"
            )

        if self.X_val.shape[1] != len(self.y_val):  # pragma: no cover
            raise ValueError(
                "X_val does not have the same number of samples as y_val"
            )

        if self.w_train.shape != self.y_train.shape or self.w_val.shape != self.y_val.shape:
            raise ValueError("Sample weight shape must be equal to training label shape.")

        native = Native.get_native_singleton()

        log.info("Allocation training start")

        (
            feature_groups_feature_count,
            feature_groups_feature_indexes,
        ) = Native._convert_feature_groups_to_c(self.feature_groups)

        n_scores = Native.get_count_scores_c(self.n_classes)
        scores_train = self.scores_train
        if scores_train is None:
            scores_train = np.zeros(len(self.y_train) * n_scores, dtype=ct.c_double, order="C")
        else:
            if scores_train.shape[0] != len(self.y_train):  # pragma: no cover
                raise ValueError(
                    "scores_train does not have the same number of samples as y_train"
                )
            if n_scores == 1:
                if scores_train.ndim != 1:  # pragma: no cover
                    raise ValueError(
                        "scores_train should have exactly 1 dimensions for regression or binary classification"
                    )
            else:
                if scores_train.ndim != 2:  # pragma: no cover
                    raise ValueError(
                        "scores_train should have exactly 2 dimensions for multiclass"
                    )
                if scores_train.shape[1] != n_scores:  # pragma: no cover
                    raise ValueError(
                        "scores_train does not have the same number of logit scores as n_scores"
                    )

        scores_val = self.scores_val
        if scores_val is None:
            scores_val = np.zeros(len(self.y_val) * n_scores, dtype=ct.c_double, order="C")
        else:
            if scores_val.shape[0] != len(self.y_val):  # pragma: no cover
                raise ValueError(
                    "scores_val does not have the same number of samples as y_val"
                )
            if n_scores == 1:
                if scores_val.ndim != 1:  # pragma: no cover
                    raise ValueError(
                        "scores_val should have exactly 1 dimensions for regression or binary classification"
                    )
            else:
                if scores_val.ndim != 2:  # pragma: no cover
                    raise ValueError(
                        "scores_val should have exactly 2 dimensions for multiclass"
                    )
                if scores_val.shape[1] != n_scores:  # pragma: no cover
                    raise ValueError(
                        "scores_val does not have the same number of logit scores as n_scores"
                    )

        optional_temp_params = self.optional_temp_params
        if optional_temp_params is not None:  # pragma: no cover
            optional_temp_params = (ct.c_double * len(optional_temp_params))(
                *optional_temp_params
            )

        # Allocate external resources
        booster_handle = ct.c_void_p(0)
        if self.model_type == "classification":
            return_code = native._unsafe.CreateClassificationBooster(
                self.random_state,
                self.n_classes,
                len(self.features_bin_count),
                self.features_categorical, 
                self.features_bin_count,
                len(feature_groups_feature_count),
                feature_groups_feature_count,
                feature_groups_feature_indexes,
                len(self.y_train),
                self.X_train,
                self.y_train,
                self.w_train,
                scores_train,
                len(self.y_val),
                self.X_val,
                self.y_val,
                self.w_val,
                scores_val,
                self.n_inner_bags,
                optional_temp_params,
                ct.byref(booster_handle),
            )
            if return_code:  # pragma: no cover
                raise Native._get_native_exception(return_code, "CreateClassificationBooster")
        elif self.model_type == "regression":
            return_code = native._unsafe.CreateRegressionBooster(
                self.random_state,
                len(self.features_bin_count),
                self.features_categorical, 
                self.features_bin_count,
                len(feature_groups_feature_count),
                feature_groups_feature_count,
                feature_groups_feature_indexes,
                len(self.y_train),
                self.X_train,
                self.y_train,
                self.w_train,
                scores_train,
                len(self.y_val),
                self.X_val,
                self.y_val,
                self.w_val,
                scores_val,
                self.n_inner_bags,
                optional_temp_params,
                ct.byref(booster_handle),
            )
            if return_code:  # pragma: no cover
                raise Native._get_native_exception(return_code, "CreateRegressionBooster")
        else:  # pragma: no cover
            raise AttributeError("Unrecognized model_type")

        self._booster_handle = booster_handle.value

        log.info("Allocation boosting end")
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

    def generate_model_update(
        self, 
        feature_group_index, 
        generate_update_options, 
        learning_rate, 
        min_samples_leaf, 
        max_leaves, 
    ):

        """ Generates a boosting step update per feature
            by growing a shallow decision tree.

        Args:
            feature_group_index: The index for the feature group to generate the update for
            generate_update_options: C interface options
            learning_rate: Learning rate as a float.
            min_samples_leaf: Min observations required to split.
            max_leaves: Max leaf nodes on feature step.

        Returns:
            gain for the generated boosting step.
        """

        # log.debug("Boosting step start")

        self._feature_group_index = -1

        native = Native.get_native_singleton()

        gain = ct.c_double(0.0)
        n_features = len(self.feature_groups[feature_group_index])
        max_leaves_arr = np.full(n_features, max_leaves, dtype=ct.c_int64, order="C")

        return_code = native._unsafe.GenerateModelUpdate(
            self._booster_handle, 
            feature_group_index,
            generate_update_options,
            learning_rate,
            min_samples_leaf,
            max_leaves_arr,
            ct.byref(gain),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "GenerateModelUpdate")
            
        self._feature_group_index = feature_group_index

        # log.debug("Boosting step end")
        return gain.value

    def apply_model_update(self):

        """ Updates the interal C state with the last model update

        Args:

        Returns:
            Validation loss for the boosting step.
        """
        # log.debug("Boosting step start")

        self._feature_group_index = -1

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
        for index in range(len(self.feature_groups)):
            model_feature_group = self._get_best_model_feature_group(index)
            model.append(model_feature_group)

        return model

    # TODO: Needs test.
    def get_current_model(self):
        model = []
        for index in range(len(self.feature_groups)):
            model_feature_group = self._get_current_model_feature_group(
                index
            )
            model.append(model_feature_group)

        return model

    def get_model_update_splits(self):
        if self._feature_group_index < 0:  # pragma: no cover
            raise RuntimeError("invalid internal self._feature_group_index")

        splits = []
        feature_indexes = self.feature_groups[self._feature_group_index]
        for dimension_idx, _ in enumerate(feature_indexes):
            splits_dimension = self._get_model_update_splits_dimension(dimension_idx)
            splits.append(splits_dimension)

        return splits

    def _get_feature_group_shape(self, feature_group_index):
        # TODO PK do this once during construction so that we don't have to do it again
        #         and so that we don't have to store self._features & self.feature_groups

        # Retrieve dimensions of log odds tensor
        dimensions = []
        feature_indexes = self.feature_groups[feature_group_index]
        for _, feature_idx in enumerate(feature_indexes):
            n_bins = self.features_bin_count[feature_idx]
            dimensions.append(n_bins)

        dimensions = list(reversed(dimensions))

        # Array returned for multiclass is one higher dimension
        n_scores = Native.get_count_scores_c(self.n_classes)
        if n_scores > 1:
            dimensions.append(n_scores)

        shape = tuple(dimensions)
        return shape

    def _get_best_model_feature_group(self, feature_group_index):
        """ Returns best model/function according to validation set
            for a given feature group.

        Args:
            feature_group_index: The index for the feature group.

        Returns:
            An ndarray that represents the model.
        """

        if self.model_type == "classification" and self.n_classes <= 1:  # pragma: no cover
            # if there is only one legal state for a classification problem, then we know with 100%
            # certainty what the result will be, and our model has no information since we always predict
            # the only output
            return None

        native = Native.get_native_singleton()

        shape = self._get_feature_group_shape(feature_group_index)
        model_feature_group = np.empty(shape, dtype=np.float64, order="C")

        return_code = native._unsafe.GetBestModelFeatureGroup(
            self._booster_handle, feature_group_index, model_feature_group
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "GetBestModelFeatureGroup")

        if len(self.feature_groups[feature_group_index]) == 2:
            if 2 < self.n_classes:
                model_feature_group = np.ascontiguousarray(np.transpose(model_feature_group, (1, 0, 2)))
            else:
                model_feature_group = np.ascontiguousarray(np.transpose(model_feature_group, (1, 0)))

        return model_feature_group

    def _get_current_model_feature_group(self, feature_group_index):
        """ Returns current model/function according to validation set
            for a given feature group.

        Args:
            feature_group_index: The index for the feature group.

        Returns:
            An ndarray that represents the model.
        """

        if self.model_type == "classification" and self.n_classes <= 1:  # pragma: no cover
            # if there is only one legal state for a classification problem, then we know with 100%
            # certainty what the result will be, and our model has no information since we always predict
            # the only output
            return None

        native = Native.get_native_singleton()

        shape = self._get_feature_group_shape(feature_group_index)
        model_feature_group = np.empty(shape, dtype=np.float64, order="C")

        return_code = native._unsafe.GetCurrentModelFeatureGroup(
            self._booster_handle, feature_group_index, model_feature_group
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "GetCurrentModelFeatureGroup")

        if len(self.feature_groups[feature_group_index]) == 2:
            if 2 < self.n_classes:
                model_feature_group = np.ascontiguousarray(np.transpose(model_feature_group, (1, 0, 2)))
            else:
                model_feature_group = np.ascontiguousarray(np.transpose(model_feature_group, (1, 0)))

        return model_feature_group

    def _get_model_update_splits_dimension(self, dimension_index):
        native = Native.get_native_singleton()

        feature_index = self.feature_groups[self._feature_group_index][dimension_index]
        n_bins = self.features_bin_count[feature_index]

        count_splits = n_bins - 1
        splits = np.empty(count_splits, dtype=np.int64, order="C")
        count_splits = ct.c_int64(count_splits)

        return_code = native._unsafe.GetModelUpdateSplits(
            self._booster_handle, 
            dimension_index, 
            ct.byref(count_splits), 
            splits
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "GetModelUpdateSplits")

        splits = splits[:count_splits.value]
        return splits

    def get_model_update_expanded(self):
        if self._feature_group_index < 0:  # pragma: no cover
            raise RuntimeError("invalid internal self._feature_group_index")

        if self.model_type == "classification" and self.n_classes <= 1:  # pragma: no cover
            # if there is only one legal state for a classification problem, then we know with 100%
            # certainty what the result will be, and our model has no information since we always predict
            # the only output
            return None

        native = Native.get_native_singleton()

        shape = self._get_feature_group_shape(self._feature_group_index)
        model_update = np.empty(shape, dtype=np.float64, order="C")

        return_code = native._unsafe.GetModelUpdateExpanded(self._booster_handle, model_update)
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "GetModelUpdateExpanded")

        if len(self.feature_groups[self._feature_group_index]) == 2:
            if 2 < self.n_classes:
                model_update = np.ascontiguousarray(np.transpose(model_update, (1, 0, 2)))
            else:
                model_update = np.ascontiguousarray(np.transpose(model_update, (1, 0)))

        return model_update

    def set_model_update_expanded(self, feature_group_index, model_update):
        self._feature_group_index = -1

        if self.model_type == "classification" and self.n_classes <= 1:  # pragma: no cover
            if model_update is None:  # pragma: no cover
                self._feature_group_index = feature_group_index
                return
            raise ValueError("a tensor with 1 class or less would be empty since the predictions would always be the same")

        if len(self.feature_groups[feature_group_index]) == 2:
            if 2 < self.n_classes:
                model_update = np.ascontiguousarray(np.transpose(model_update, (1, 0, 2)))
            else:
                model_update = np.ascontiguousarray(np.transpose(model_update, (1, 0)))

        shape = self._get_feature_group_shape(feature_group_index)

        if shape != model_update.shape:  # pragma: no cover
            raise ValueError("incorrect tensor shape in call to set_model_update_expanded")

        native = Native.get_native_singleton()
        return_code = native._unsafe.SetModelUpdateExpanded(
            self._booster_handle, feature_group_index, model_update
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "SetModelUpdateExpanded")

        self._feature_group_index = feature_group_index

        return


class InteractionDetector(AbstractContextManager):
    """Lightweight wrapper for EBM C interaction code.
    """

    def __init__(
        self, 
        model_type, 
        n_classes, 
        features_categorical, 
        features_bin_count, 
        X, 
        y, 
        w, 
        scores, 
        optional_temp_params
    ):

        """ Initializes internal wrapper for EBM C code.

        Args:
            model_type: 'regression'/'classification'.
            n_classes: Specific to classification,
                number of unique classes.
            features_categorical: list of categorical features represented by bools 
            features_bin_count: count of the number of bins for each feature
            X: Training design matrix as 2-D ndarray.
            y: Training response as 1-D ndarray.
            w: Sample weights as 1-D ndarray (must be same shape as y).
            scores: predictions from a prior predictor.  For regression
                there is 1 prediction per sample.  For binary classification
                there is one logit.  For multiclass there are n_classes logits

        """

        self.model_type = model_type
        self.n_classes = n_classes
        self.features_categorical = features_categorical
        self.features_bin_count = features_bin_count
        self.X = X
        self.y = y
        self.w = w
        self.scores = scores
        self.optional_temp_params = optional_temp_params

    def __enter__(self):
        # check inputs for important inputs or things that would segfault in C
        if not isinstance(self.features_categorical, np.ndarray):  # pragma: no cover
            raise ValueError("features_categorical should be an np.ndarray")

        if not isinstance(self.features_bin_count, np.ndarray):  # pragma: no cover
            raise ValueError("features_bin_count should be an np.ndarray")

        if self.X.ndim != 2:  # pragma: no cover
            raise ValueError("X should have exactly 2 dimensions")

        if self.y.ndim != 1:  # pragma: no cover
            raise ValueError("y should have exactly 1 dimension")


        if self.X.shape[0] != len(self.features_categorical):  # pragma: no cover
            raise ValueError(
                "X does not have the same number of items as the features_categorical array"
            )

        if self.X.shape[0] != len(self.features_bin_count):  # pragma: no cover
            raise ValueError(
                "X does not have the same number of items as the features_bin_count array"
            )

        if self.X.shape[1] != len(self.y):  # pragma: no cover
            raise ValueError("X does not have the same number of samples as y")

        native = Native.get_native_singleton()

        log.info("Allocation interaction start")

        n_scores = Native.get_count_scores_c(self.n_classes)
        scores = self.scores
        if scores is None:  # pragma: no cover
            scores = np.zeros(len(self.y) * n_scores, dtype=ct.c_double, order="C")
        else:
            if scores.shape[0] != len(self.y):  # pragma: no cover
                raise ValueError(
                    "scores does not have the same number of samples as y"
                )
            if n_scores == 1:
                if scores.ndim != 1:  # pragma: no cover
                    raise ValueError(
                        "scores should have exactly 1 dimensions for regression or binary classification"
                    )
            else:
                if scores.ndim != 2:  # pragma: no cover
                    raise ValueError(
                        "scores should have exactly 2 dimensions for multiclass"
                    )
                if scores.shape[1] != n_scores:  # pragma: no cover
                    raise ValueError(
                        "scores does not have the same number of logit scores as n_scores"
                    )

        optional_temp_params = self.optional_temp_params
        if optional_temp_params is not None:  # pragma: no cover
            optional_temp_params = (ct.c_double * len(optional_temp_params))(
                *optional_temp_params
            )

        # Allocate external resources
        interaction_handle = ct.c_void_p(0)
        if self.model_type == "classification":
            return_code = native._unsafe.CreateClassificationInteractionDetector(
                self.n_classes,
                len(self.features_bin_count),
                self.features_categorical, 
                self.features_bin_count,
                len(self.y),
                self.X,
                self.y,
                self.w,
                scores,
                optional_temp_params,
                ct.byref(interaction_handle),
            )
            if return_code:  # pragma: no cover
                raise Native._get_native_exception(return_code, "CreateClassificationInteractionDetector")
        elif self.model_type == "regression":
            return_code = native._unsafe.CreateRegressionInteractionDetector(
                len(self.features_bin_count),
                self.features_categorical, 
                self.features_bin_count,
                len(self.y),
                self.X,
                self.y,
                self.w,
                scores,
                optional_temp_params,
                ct.byref(interaction_handle),
            )
            if return_code:  # pragma: no cover
                raise Native._get_native_exception(return_code, "CreateRegressionInteractionDetector")
        else:  # pragma: no cover
            raise AttributeError("Unrecognized model_type")

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

    def get_interaction_score(self, feature_index_tuple, min_samples_leaf):
        """ Provides score for an feature interaction. Higher is better."""
        log.info("Fast interaction score start")

        native = Native.get_native_singleton()

        score = ct.c_double(0.0)
        return_code = native._unsafe.CalculateInteractionScore(
            self._interaction_handle,
            len(feature_index_tuple),
            np.array(feature_index_tuple, dtype=ct.c_int64),
            min_samples_leaf,
            ct.byref(score),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "CalculateInteractionScore")

        log.info("Fast interaction score end")
        return score.value
