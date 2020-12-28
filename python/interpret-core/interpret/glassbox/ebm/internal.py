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
from contextlib import closing

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
    _LogFuncType = ct.CFUNCTYPE(None, ct.c_int32, ct.c_char_p)

    def __init__(self):
        # Do not call "Native()".  Call "Native.get_native_singleton()" instead
        pass

    @staticmethod
    def get_native_singleton(is_debug=False):
        log.debug("Check if EBM lib is loaded")
        if Native._native is None:
            log.info("EBM lib loading.")
            native = Native()
            native._initialize(is_debug=is_debug)
            Native._native = native
        else:
            log.debug("EBM lib already loaded")
        return Native._native

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

    def generate_quantile_bin_cuts(
        self, 
        col_data, 
        min_samples_bin, 
        is_humanized, 
        max_cuts, 
    ):
        bin_cuts = np.empty(max_cuts, dtype=np.float64, order="C")
        count_cuts = ct.c_int64(max_cuts)
        count_missing = ct.c_int64(0)
        min_val = ct.c_double(0)
        count_neg_inf = ct.c_int64(0)
        max_val = ct.c_double(0)
        count_inf = ct.c_int64(0)

        return_code = self._unsafe.GenerateQuantileBinCuts(
            col_data.shape[0],
            col_data, 
            min_samples_bin,
            is_humanized,
            ct.byref(count_cuts),
            bin_cuts,
            ct.byref(count_missing),
            ct.byref(min_val),
            ct.byref(count_neg_inf),
            ct.byref(max_val),
            ct.byref(count_inf)
        )

        if return_code != 0:  # pragma: no cover
            raise Exception("Out of memory in GenerateQuantileBinCuts")

        bin_cuts = bin_cuts[:count_cuts.value]
        count_missing = count_missing.value
        min_val = min_val.value
        max_val = max_val.value

        return bin_cuts, count_missing, min_val, max_val

    def generate_uniform_bin_cuts(
        self, 
        col_data, 
        max_cuts, 
    ):
        bin_cuts = np.empty(max_cuts, dtype=np.float64, order="C")
        count_cuts = ct.c_int64(max_cuts)
        count_missing = ct.c_int64(0)
        min_val = ct.c_double(0)
        count_neg_inf = ct.c_int64(0)
        max_val = ct.c_double(0)
        count_inf = ct.c_int64(0)

        self._unsafe.GenerateUniformBinCuts(
            col_data.shape[0],
            col_data, 
            ct.byref(count_cuts),
            bin_cuts,
            ct.byref(count_missing),
            ct.byref(min_val),
            ct.byref(count_neg_inf),
            ct.byref(max_val),
            ct.byref(count_inf)
        )

        bin_cuts = bin_cuts[:count_cuts.value]
        count_missing = count_missing.value
        min_val = min_val.value
        max_val = max_val.value

        return bin_cuts, count_missing, min_val, max_val

    def discretize(
        self, 
        col_data, 
        bin_cuts, 
    ):
        discretized = np.empty(col_data.shape[0], dtype=np.int64, order="C")
        return_code = self._unsafe.Discretize(
            col_data.shape[0],
            col_data,
            bin_cuts.shape[0],
            bin_cuts,
            discretized
        )

        if return_code != 0:  # pragma: no cover
            raise Exception("Out of memory in Discretize")

        return discretized


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
        if platform == "linux" or platform == "linux2" and is_64_bit:
            return os.path.join(
                package_path, "lib", "lib_ebm_native_linux_x64{0}.so".format(debug_str)
            )
        elif platform == "win32" and is_64_bit:
            return os.path.join(
                package_path, "lib", "lib_ebm_native_win_x64{0}.dll".format(debug_str)
            )
        elif platform == "darwin" and is_64_bit:
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


        self._unsafe.GenerateQuantileBinCuts.argtypes = [
            # int64_t countSamples
            ct.c_int64,
            # double * featureValues
            ndpointer(dtype=ct.c_double, ndim=1, flags="C_CONTIGUOUS"),
            # int64_t countSamplesPerBinMin
            ct.c_int64,
            # int64_t isHumanized
            ct.c_int64,
            # int64_t * countBinCutsInOut
            ct.POINTER(ct.c_int64),
            # double * binCutsLowerBoundInclusiveOut
            ndpointer(dtype=ct.c_double, ndim=1, flags="C_CONTIGUOUS"),
            # int64_t * countMissingValuesOut
            ct.POINTER(ct.c_int64),
            # double * minNonInfinityValueOut
            ct.POINTER(ct.c_double),
            # int64_t * countNegativeInfinityOut
            ct.POINTER(ct.c_int64),
            # double * maxNonInfinityValueOut
            ct.POINTER(ct.c_double),
            # int64_t * countPositiveInfinityOut
            ct.POINTER(ct.c_int64),
        ]
        self._unsafe.GenerateQuantileBinCuts.restype = ct.c_int64

        self._unsafe.GenerateUniformBinCuts.argtypes = [
            # int64_t countSamples
            ct.c_int64,
            # double * featureValues
            ndpointer(dtype=ct.c_double, ndim=1, flags="C_CONTIGUOUS"),
            # int64_t * countBinCutsInOut
            ct.POINTER(ct.c_int64),
            # double * binCutsLowerBoundInclusiveOut
            ndpointer(dtype=ct.c_double, ndim=1, flags="C_CONTIGUOUS"),
            # int64_t * countMissingValuesOut
            ct.POINTER(ct.c_int64),
            # double * minNonInfinityValueOut
            ct.POINTER(ct.c_double),
            # int64_t * countNegativeInfinityOut
            ct.POINTER(ct.c_int64),
            # double * maxNonInfinityValueOut
            ct.POINTER(ct.c_double),
            # int64_t * countPositiveInfinityOut
            ct.POINTER(ct.c_int64),
        ]
        self._unsafe.GenerateUniformBinCuts.restype = None

        self._unsafe.GenerateWinsorizedBinCuts.argtypes = [
            # int64_t countSamples
            ct.c_int64,
            # double * featureValues
            ndpointer(dtype=ct.c_double, ndim=1, flags="C_CONTIGUOUS"),
            # int64_t * countBinCutsInOut
            ct.POINTER(ct.c_int64),
            # double * binCutsLowerBoundInclusiveOut
            ndpointer(dtype=ct.c_double, ndim=1, flags="C_CONTIGUOUS"),
            # int64_t * countMissingValuesOut
            ct.POINTER(ct.c_int64),
            # double * minNonInfinityValueOut
            ct.POINTER(ct.c_double),
            # int64_t * countNegativeInfinityOut
            ct.POINTER(ct.c_int64),
            # double * maxNonInfinityValueOut
            ct.POINTER(ct.c_double),
            # int64_t * countPositiveInfinityOut
            ct.POINTER(ct.c_int64),
        ]
        self._unsafe.GenerateWinsorizedBinCuts.restype = ct.c_int64


        self._unsafe.SuggestGraphBounds.argtypes = [
            # int64_t countBinCuts
            ct.c_int64,
            # double lowestBinCut
            ct.c_double,
            # double highestBinCut
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
        self._unsafe.SuggestGraphBounds.restype = None


        self._unsafe.Discretize.argtypes = [
            # int64_t countSamples
            ct.c_int64,
            # double * featureValues
            ndpointer(dtype=ct.c_double, ndim=1, flags="C_CONTIGUOUS"),
            # int64_t countBinCuts
            ct.c_int64,
            # double * binCutsLowerBoundInclusive
            ndpointer(dtype=ct.c_double, ndim=1, flags="C_CONTIGUOUS"),
            # int64_t * discretizedOut
            ndpointer(dtype=ct.c_int64, ndim=1, flags="C_CONTIGUOUS"),
        ]
        self._unsafe.Discretize.restype = ct.c_int64


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
        self._unsafe.Softmax.restype = ct.c_int64


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
            # int64_t * featureGroupsFeatureCount
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
            # ndpointer(dtype=ct.c_double, ndim=1),
            ct.c_void_p,
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
            # ndpointer(dtype=ct.c_double, ndim=1),
            ct.c_void_p,
            # double * validationPredictorScores
            # scores can either be 1 or 2 dimensional
            ndpointer(dtype=ct.c_double, flags="C_CONTIGUOUS"),
            # int64_t countInnerBags
            ct.c_int64,
            # double * optionalTempParams
            ct.POINTER(ct.c_double),
        ]
        self._unsafe.CreateClassificationBooster.restype = ct.c_void_p

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
            # int64_t * featureGroupsFeatureCount
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
            # ndpointer(dtype=ct.c_double, ndim=1),
            ct.c_void_p,
            # double * trainingPredictorScores
            ndpointer(dtype=ct.c_double, ndim=1),
            # int64_t countValidationSamples
            ct.c_int64,
            # int64_t * validationBinnedData
            ndpointer(dtype=ct.c_int64, ndim=2, flags="C_CONTIGUOUS"),
            # double * validationTargets
            ndpointer(dtype=ct.c_double, ndim=1),
            # double * validationWeights
            # ndpointer(dtype=ct.c_double, ndim=1),
            ct.c_void_p,
            # double * validationPredictorScores
            ndpointer(dtype=ct.c_double, ndim=1),
            # int64_t countInnerBags
            ct.c_int64,
            # double * optionalTempParams
            ct.POINTER(ct.c_double),
        ]
        self._unsafe.CreateRegressionBooster.restype = ct.c_void_p

        self._unsafe.GenerateModelFeatureGroupUpdate.argtypes = [
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
        self._unsafe.GenerateModelFeatureGroupUpdate.restype = ct.POINTER(ct.c_double)

        self._unsafe.ApplyModelFeatureGroupUpdate.argtypes = [
            # void * boosterHandle
            ct.c_void_p,
            # int64_t indexFeatureGroup
            ct.c_int64,
            # double * modelFeatureGroupUpdateTensor
            ndpointer(dtype=ct.c_double, flags="C_CONTIGUOUS"),
            # double * validationMetricOut
            ct.POINTER(ct.c_double),
        ]
        self._unsafe.ApplyModelFeatureGroupUpdate.restype = ct.c_int64

        self._unsafe.GetBestModelFeatureGroup.argtypes = [
            # void * boosterHandle
            ct.c_void_p,
            # int64_t indexFeatureGroup
            ct.c_int64,
        ]
        self._unsafe.GetBestModelFeatureGroup.restype = ct.POINTER(ct.c_double)

        self._unsafe.GetCurrentModelFeatureGroup.argtypes = [
            # void * boosterHandle
            ct.c_void_p,
            # int64_t indexFeatureGroup
            ct.c_int64,
        ]
        self._unsafe.GetCurrentModelFeatureGroup.restype = ct.POINTER(ct.c_double)

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
            # ndpointer(dtype=ct.c_double, ndim=1),
            ct.c_void_p,
            # double * predictorScores
            # scores can either be 1 or 2 dimensional
            ndpointer(dtype=ct.c_double, flags="C_CONTIGUOUS"),
            # double * optionalTempParams
            ct.POINTER(ct.c_double),
        ]
        self._unsafe.CreateClassificationInteractionDetector.restype = ct.c_void_p

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
            # ndpointer(dtype=ct.c_double, ndim=1),
            ct.c_void_p,
            # double * predictorScores
            ndpointer(dtype=ct.c_double, ndim=1),
            # double * optionalTempParams
            ct.POINTER(ct.c_double),
        ]
        self._unsafe.CreateRegressionInteractionDetector.restype = ct.c_void_p

        self._unsafe.CalculateInteractionScore.argtypes = [
            # void * interactionDetectorHandle
            ct.c_void_p,
            # int64_t countFeaturesInGroup
            ct.c_int64,
            # int64_t * featureIndexes
            ndpointer(dtype=ct.c_int64, ndim=1),
            # int64_t countSamplesRequiredForChildSplitMin
            ct.c_int64,
            # double * interactionScoreOut
            ct.POINTER(ct.c_double),
        ]
        self._unsafe.CalculateInteractionScore.restype = ct.c_int64

        self._unsafe.FreeInteractionDetector.argtypes = [
            # void * interactionDetectorHandle
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


class NativeEBMBooster:
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
        scores_train,
        X_val,
        y_val,
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

        # first set the one thing that we will close on
        self._booster_handle = None

        # check inputs for important inputs or things that would segfault in C
        if not isinstance(features_categorical, np.ndarray):  # pragma: no cover
            raise ValueError("features_categorical should be an np.ndarray")

        if not isinstance(features_bin_count, np.ndarray):  # pragma: no cover
            raise ValueError("features_bin_count should be an np.ndarray")

        if not isinstance(feature_groups, list):  # pragma: no cover
            raise ValueError("feature_groups should be a list")

        if X_train.ndim != 2:  # pragma: no cover
            raise ValueError("X_train should have exactly 2 dimensions")

        if y_train.ndim != 1:  # pragma: no cover
            raise ValueError("y_train should have exactly 1 dimension")

        if X_train.shape[0] != len(features_categorical):  # pragma: no cover
            raise ValueError(
                "X_train does not have the same number of items as the features_categorical array"
            )

        if X_train.shape[0] != len(features_bin_count):  # pragma: no cover
            raise ValueError(
                "X_train does not have the same number of items as the features_bin_count array"
            )

        if X_train.shape[1] != len(y_train):  # pragma: no cover
            raise ValueError(
                "X_train does not have the same number of samples as y_train"
            )

        if X_val.ndim != 2:  # pragma: no cover
            raise ValueError("X_val should have exactly 2 dimensions")

        if y_val.ndim != 1:  # pragma: no cover
            raise ValueError("y_val should have exactly 1 dimension")

        if X_val.shape[0] != X_train.shape[0]:  # pragma: no cover
            raise ValueError(
                "X_val does not have the same number of features as the X_train array"
            )

        if X_val.shape[1] != len(y_val):  # pragma: no cover
            raise ValueError(
                "X_val does not have the same number of samples as y_val"
            )

        self._native = Native.get_native_singleton()

        log.info("Allocation training start")

        # Store args
        self._model_type = model_type
        self._n_classes = n_classes

        self._features_bin_count = features_bin_count

        self._feature_groups = feature_groups
        (
            feature_groups_feature_count,
            feature_groups_feature_indexes,
        ) = Native._convert_feature_groups_to_c(feature_groups)

        n_scores = Native.get_count_scores_c(n_classes)
        if scores_train is None:
            scores_train = np.zeros(len(y_train) * n_scores, dtype=ct.c_double, order="C")
        else:
            if scores_train.shape[0] != len(y_train):  # pragma: no cover
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

        if scores_val is None:
            scores_val = np.zeros(len(y_val) * n_scores, dtype=ct.c_double, order="C")
        else:
            if scores_val.shape[0] != len(y_val):  # pragma: no cover
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

        if optional_temp_params is not None:
            optional_temp_params = (ct.c_double * len(optional_temp_params))(
                *optional_temp_params
            )

        # Allocate external resources
        if model_type == "classification":
            self._booster_handle = self._native._unsafe.CreateClassificationBooster(
                random_state,
                n_classes,
                len(features_bin_count),
                features_categorical, 
                features_bin_count,
                len(feature_groups_feature_count),
                feature_groups_feature_count,
                feature_groups_feature_indexes,
                len(y_train),
                X_train,
                y_train,
                0,
                scores_train,
                len(y_val),
                X_val,
                y_val,
                0,
                scores_val,
                n_inner_bags,
                optional_temp_params,
            )
            if not self._booster_handle:  # pragma: no cover
                raise MemoryError("Out of memory in CreateClassificationBooster")
        elif model_type == "regression":
            self._booster_handle = self._native._unsafe.CreateRegressionBooster(
                random_state,
                len(features_bin_count),
                features_categorical, 
                features_bin_count,
                len(feature_groups_feature_count),
                feature_groups_feature_count,
                feature_groups_feature_indexes,
                len(y_train),
                X_train,
                y_train,
                0,
                scores_train,
                len(y_val),
                X_val,
                y_val,
                0,
                scores_val,
                n_inner_bags,
                optional_temp_params,
            )
            if not self._booster_handle:  # pragma: no cover
                raise MemoryError("Out of memory in CreateRegressionBooster")
        else:  # pragma: no cover
            raise AttributeError("Unrecognized model_type")

        log.info("Allocation boosting end")

    def close(self):
        """ Deallocates C objects used to boost EBM. """
        log.info("Deallocation boosting start")
        self._native._unsafe.FreeBooster(self._booster_handle)
        log.info("Deallocation boosting end")

    def boosting_step(
        self, 
        feature_group_index, 
        generate_update_options, 
        learning_rate, 
        min_samples_leaf, 
        max_leaves, 
    ):

        """ Conducts a boosting step per feature
            by growing a shallow decision tree.

        Args:
            feature_group_index: The index for the feature group
                to boost on.
            learning_rate: Learning rate as a float.
            max_leaves: Max leaf nodes on feature step.
            min_samples_leaf: Min observations required to split.

        Returns:
            Validation loss for the boosting step.
        """
        # log.debug("Boosting step start")

        metric_output = ct.c_double(0.0)
        # for a classification problem with only 1 target value, we will always predict the answer perfectly
        if self._model_type != "classification" or 2 <= self._n_classes:
            gain = ct.c_double(0.0)

            # TODO : !WARNING! currently we can only accept a single number for max_leaves because in C++ we eliminate
            #        dimensions that have only 1 state.  If a dimension in C++ is eliminated this way then it won't
            #        match the dimensionality of the max_leaves_arr that we pass in here.  If all the numbers are the
            #        same though as they currently are below, we're safe since we just access one less item in the
            #        array, and still get the same numbers in the C++ code
            #        Look at GenerateModelFeatureGroupUpdate in the C++ for more details on resolving this issue
            n_features = len(self._feature_groups[feature_group_index])
            max_leaves_arr = np.full(n_features, max_leaves, dtype=ct.c_int64, order="C")

            model_update_tensor_pointer = self._native._unsafe.GenerateModelFeatureGroupUpdate(
                self._booster_handle,
                feature_group_index,
                generate_update_options,
                learning_rate,
                min_samples_leaf,
                max_leaves_arr,
                ct.byref(gain),
            )
            if not model_update_tensor_pointer:  # pragma: no cover
                raise MemoryError(
                    "Out of memory in GenerateModelFeatureGroupUpdate"
                )

            shape = self._get_feature_group_shape(feature_group_index)
            # TODO PK verify that we aren't copying data while making the view and/or passing to ApplyModelFeatureGroupUpdate
            model_update_tensor = NativeEBMBooster._make_ndarray(
                model_update_tensor_pointer, shape, dtype=ct.c_double, copy_data=False
            )

            return_code = self._native._unsafe.ApplyModelFeatureGroupUpdate(
                self._booster_handle,
                feature_group_index,
                model_update_tensor,
                ct.byref(metric_output),
            )
            if return_code != 0:  # pragma: no cover
                raise Exception("Out of memory in ApplyModelFeatureGroupUpdate")

        # log.debug("Boosting step end")
        return metric_output.value

    def get_best_model(self):
        model = []
        for index in range(len(self._feature_groups)):
            model_feature_group = self._get_best_model_feature_group(index)
            model.append(model_feature_group)

        return model

    # TODO: Needs test.
    def get_current_model(self):
        model = []
        for index in range(len(self._feature_groups)):
            model_feature_group = self._get_current_model_feature_group(
                index
            )
            model.append(model_feature_group)

        return model


    @staticmethod
    def _make_ndarray(c_pointer, shape, dtype, writable=False, copy_data=True):
        """ Returns an ndarray based from a C array.

        Code largely borrowed from:
        https://stackoverflow.com/questions/4355524/getting-data-from-ctypes-array-into-numpy

        Args:
            c_pointer: Pointer to C array.
            shape: Shape of ndarray to form.
            dtype: Numpy data type.

        Returns:
            An ndarray.
        """

        arr_size = np.prod(shape[:]) * np.dtype(dtype).itemsize
        buf_from_mem = ct.pythonapi.PyMemoryView_FromMemory
        buf_from_mem.restype = ct.py_object
        buf_from_mem.argtypes = (ct.c_void_p, ct.c_ssize_t, ct.c_int)
        PyBUF_READ = 0x100
        PyBUF_WRITE = 0x200
        access = (PyBUF_READ | PyBUF_WRITE) if writable else PyBUF_READ
        buffer = buf_from_mem(c_pointer, arr_size, access)
        # from https://github.com/python/cpython/blob/master/Objects/memoryobject.c , PyMemoryView_FromMemory can return null
        if not buffer:
            raise MemoryError("Out of memory in PyMemoryView_FromMemory")
        arr = np.ndarray(tuple(shape[:]), dtype, buffer, order="C")
        if copy_data:
            return arr.copy()
        else:
            return arr

    def _get_feature_group_shape(self, feature_group_index):
        # TODO PK do this once during construction so that we don't have to do it again
        #         and so that we don't have to store self._features & self._feature_groups

        # Retrieve dimensions of log odds tensor
        dimensions = []
        feature_indexes = self._feature_groups[feature_group_index]
        for _, feature_idx in enumerate(feature_indexes):
            n_bins = self._features_bin_count[feature_idx]
            dimensions.append(n_bins)

        dimensions = list(reversed(dimensions))

        # Array returned for multiclass is one higher dimension
        n_scores = Native.get_count_scores_c(self._n_classes)
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

        if self._model_type == "classification" and self._n_classes <= 1:
            # if there is only one legal state for a classification problem, then we know with 100%
            # certainty what the result will be, and our logits for that result should be infinity
            # since we reduce the number of logits by 1, we would get back an empty array from the C code
            # after we expand the model for our caller, the tensor's dimensions should match
            # the features for the feature_group, but the last class_index should have a dimension
            # of 1 for the infinities.  This all needs to be special cased anyways, so we can just return
            # a None value here for now and handle in the upper levels
            #
            # If we were to allow datasets with zero samples, then it would also be legal for there
            # to be 0 states.  We can probably handle this the same as having 1 state though since
            # any samples in any evaluations need to have a state

            # TODO PK make sure the None value here is handled by our caller
            return None

        array_p = self._native._unsafe.GetBestModelFeatureGroup(
            self._booster_handle, feature_group_index
        )

        if not array_p:  # pragma: no cover
            raise MemoryError("Out of memory in GetBestModelFeatureGroup")

        shape = self._get_feature_group_shape(feature_group_index)

        array = NativeEBMBooster._make_ndarray(array_p, shape, dtype=ct.c_double)
        if len(self._feature_groups[feature_group_index]) == 2:
            if 2 < self._n_classes:
                array = np.ascontiguousarray(np.transpose(array, (1, 0, 2)))
            else:
                array = np.ascontiguousarray(np.transpose(array, (1, 0)))

        return array

    def _get_current_model_feature_group(self, feature_group_index):
        """ Returns current model/function according to validation set
            for a given feature group.

        Args:
            feature_group_index: The index for the feature group.

        Returns:
            An ndarray that represents the model.
        """

        if self._model_type == "classification" and self._n_classes <= 1:
            # if there is only one legal state for a classification problem, then we know with 100%
            # certainty what the result will be, and our logits for that result should be infinity
            # since we reduce the number of logits by 1, we would get back an empty array from the C code
            # after we expand the model for our caller, the tensor's dimensions should match
            # the features for the feature_group, but the last class_index should have a dimension
            # of 1 for the infinities.  This all needs to be special cased anyways, so we can just return
            # a None value here for now and handle in the upper levels
            #
            # If we were to allow datasets with zero samples, then it would also be legal for there
            # to be 0 states.  We can probably handle this the same as having 1 state though since
            # any samples in any evaluations need to have a state

            # TODO PK make sure the None value here is handled by our caller
            return None

        array_p = self._native._unsafe.GetCurrentModelFeatureGroup(
            self._booster_handle, feature_group_index
        )

        if not array_p:  # pragma: no cover
            raise MemoryError("Out of memory in GetCurrentModelFeatureGroup")

        shape = self._get_feature_group_shape(feature_group_index)

        array = NativeEBMBooster._make_ndarray(array_p, shape, dtype=ct.c_double)
        if len(self._feature_groups[feature_group_index]) == 2:
            if 2 < self._n_classes:
                array = np.ascontiguousarray(np.transpose(array, (1, 0, 2)))
            else:
                array = np.ascontiguousarray(np.transpose(array, (1, 0)))

        return array


class NativeEBMInteraction:
    """Lightweight wrapper for EBM C interaction code.
    """

    def __init__(
        self, model_type, n_classes, features_categorical, features_bin_count, X, y, scores, optional_temp_params
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
            scores: predictions from a prior predictor.  For regression
                there is 1 prediction per sample.  For binary classification
                there is one logit.  For multiclass there are n_classes logits

        """

        # first set the one thing that we will close on
        self._interaction_handle = None

        # check inputs for important inputs or things that would segfault in C
        if not isinstance(features_categorical, np.ndarray):  # pragma: no cover
            raise ValueError("features_categorical should be an np.ndarray")

        if not isinstance(features_bin_count, np.ndarray):  # pragma: no cover
            raise ValueError("features_bin_count should be an np.ndarray")

        if X.ndim != 2:  # pragma: no cover
            raise ValueError("X should have exactly 2 dimensions")

        if y.ndim != 1:  # pragma: no cover
            raise ValueError("y should have exactly 1 dimension")


        if X.shape[0] != len(features_categorical):  # pragma: no cover
            raise ValueError(
                "X does not have the same number of items as the features_categorical array"
            )

        if X.shape[0] != len(features_bin_count):  # pragma: no cover
            raise ValueError(
                "X does not have the same number of items as the features_bin_count array"
            )

        if X.shape[1] != len(y):  # pragma: no cover
            raise ValueError("X does not have the same number of samples as y")

        self._native = Native.get_native_singleton()

        log.info("Allocation interaction start")

        n_scores = Native.get_count_scores_c(n_classes)
        if scores is None:  # pragma: no cover
            scores = np.zeros(len(y) * n_scores, dtype=ct.c_double, order="C")
        else:
            if scores.shape[0] != len(y):  # pragma: no cover
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

        if optional_temp_params is not None:
            optional_temp_params = (ct.c_double * len(optional_temp_params))(
                *optional_temp_params
            )

        # Allocate external resources
        if model_type == "classification":
            self._interaction_handle = self._native._unsafe.CreateClassificationInteractionDetector(
                n_classes,
                len(features_bin_count),
                features_categorical, 
                features_bin_count,
                len(y),
                X,
                y,
                0,
                scores,
                optional_temp_params,
            )
            if not self._interaction_handle:  # pragma: no cover
                raise MemoryError(
                    "Out of memory in CreateClassificationInteractionDetector"
                )
        elif model_type == "regression":
            self._interaction_handle = self._native._unsafe.CreateRegressionInteractionDetector(
                len(features_bin_count),
                features_categorical, 
                features_bin_count,
                len(y),
                X,
                y,
                0,
                scores,
                optional_temp_params,
            )
            if not self._interaction_handle:  # pragma: no cover
                raise MemoryError("Out of memory in CreateRegressionInteractionDetector")
        else:  # pragma: no cover
            raise AttributeError("Unrecognized model_type")

        log.info("Allocation interaction end")

    def close(self):
        """ Deallocates C objects used to determine interactions in EBM. """
        log.info("Deallocation interaction start")
        self._native._unsafe.FreeInteractionDetector(self._interaction_handle)
        log.info("Deallocation interaction end")

    def get_interaction_score(self, feature_index_tuple, min_samples_leaf):
        """ Provides score for an feature interaction. Higher is better."""
        log.info("Fast interaction score start")
        score = ct.c_double(0.0)
        return_code = self._native._unsafe.CalculateInteractionScore(
            self._interaction_handle,
            len(feature_index_tuple),
            np.array(feature_index_tuple, dtype=ct.c_int64),
            min_samples_leaf,
            ct.byref(score),
        )
        if return_code != 0:  # pragma: no cover
            raise Exception("Out of memory in CalculateInteractionScore")

        log.info("Fast interaction score end")
        return score.value


class NativeHelper:
    @staticmethod
    def cyclic_gradient_boost(
        model_type,
        n_classes,
        features_categorical, 
        features_bin_count,
        feature_groups,
        X_train,
        y_train,
        scores_train,
        X_val,
        y_val,
        scores_val,
        n_inner_bags,
        generate_update_options,
        learning_rate,
        min_samples_leaf,
        max_leaves,
        early_stopping_rounds,
        early_stopping_tolerance,
        max_rounds,
        random_state,
        name,
        optional_temp_params=None,
    ):
        min_metric = np.inf
        episode_index = 0
        with closing(
            NativeEBMBooster(
                model_type,
                n_classes,
                features_categorical, 
                features_bin_count,
                feature_groups,
                X_train,
                y_train,
                scores_train,
                X_val,
                y_val,
                scores_val,
                n_inner_bags,
                random_state,
                optional_temp_params,
            )
        ) as native_ebm_booster:
            no_change_run_length = 0
            bp_metric = np.inf
            log.info("Start boosting {0}".format(name))
            for episode_index in range(max_rounds):
                if episode_index % 10 == 0:
                    log.debug("Sweep Index for {0}: {1}".format(name, episode_index))
                    log.debug("Metric: {0}".format(min_metric))

                for feature_group_index in range(len(feature_groups)):
                    curr_metric = native_ebm_booster.boosting_step(
                        feature_group_index=feature_group_index,
                        generate_update_options=generate_update_options,
                        learning_rate=learning_rate,
                        min_samples_leaf=min_samples_leaf,
                        max_leaves=max_leaves,
                    )

                    min_metric = min(curr_metric, min_metric)

                # TODO PK this early_stopping_tolerance is a little inconsistent
                #      since it triggers intermittently and only re-triggers if the
                #      threshold is re-passed, but not based on a smooth windowed set
                #      of checks.  We can do better by keeping a list of the last
                #      number of measurements to have a consistent window of values.
                #      If we only cared about the metric at the start and end of the epoch
                #      window a circular buffer would be best choice with O(1).
                if no_change_run_length == 0:
                    bp_metric = min_metric
                if min_metric + early_stopping_tolerance < bp_metric:
                    no_change_run_length = 0
                else:
                    no_change_run_length += 1

                if (
                    early_stopping_rounds >= 0
                    and no_change_run_length >= early_stopping_rounds
                ):
                    break

            log.info(
                "End boosting {0}, Best Metric: {1}, Num Rounds: {2}".format(
                    name, min_metric, episode_index
                )
            )

            # TODO: Add alternative | get_current_model
            model_update = native_ebm_booster.get_best_model()

        return model_update, min_metric, episode_index

    @staticmethod
    def get_interactions(
        n_interactions,
        iter_feature_groups,
        model_type,
        n_classes,
        features_categorical, 
        features_bin_count,
        X,
        y,
        scores,
        min_samples_leaf,
        optional_temp_params=None,
    ):
        interaction_scores = []
        with closing(
            NativeEBMInteraction(
                model_type, n_classes, features_categorical, features_bin_count, X, y, scores, optional_temp_params
            )
        ) as native_ebm_interactions:
            for feature_group in iter_feature_groups:
                score = native_ebm_interactions.get_interaction_score(
                    feature_group, min_samples_leaf,
                )
                interaction_scores.append((feature_group, score))

        ranked_scores = list(
            sorted(interaction_scores, key=lambda x: x[1], reverse=True)
        )
        final_ranked_scores = ranked_scores

        final_indices = [x[0] for x in final_ranked_scores]
        final_scores = [x[1] for x in final_ranked_scores]

        return final_indices, final_scores
