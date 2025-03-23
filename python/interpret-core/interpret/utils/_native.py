# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

# TODO: Add unit tests for internal EBM interfacing
import ctypes as ct
import logging
import os
import platform
import struct
import sys
import math
from contextlib import AbstractContextManager

import numpy as np

_log = logging.getLogger(__name__)


class Native:
    # see notes in libebm.h on the maximum representable int64 in float64 format
    FLOAT64_TO_INT64_MAX = 9223372036854774784

    # LinkFlags
    LinkFlags_Default = 0x00000000
    LinkFlags_DifferentialPrivacy = 0x00000001

    # CreateBoosterFlags
    CreateBoosterFlags_Default = 0x00000000
    CreateBoosterFlags_DifferentialPrivacy = 0x00000001
    CreateBoosterFlags_UseApprox = 0x00000002

    # TermBoostFlags
    TermBoostFlags_Default = 0x00000000
    TermBoostFlags_PurifyGain = 0x00000001
    TermBoostFlags_DisableNewtonGain = 0x00000002
    TermBoostFlags_DisableCategorical = 0x00000004
    TermBoostFlags_PurifyUpdate = 0x00000008
    TermBoostFlags_DisableNewtonUpdate = 0x00000010
    TermBoostFlags_GradientSums = 0x00000020
    TermBoostFlags_RandomSplits = 0x00000040
    TermBoostFlags_MissingLow = 0x00000080
    TermBoostFlags_MissingHigh = 0x00000100
    TermBoostFlags_MissingSeparate = 0x00000200
    TermBoostFlags_Corners = 0x00000400

    # CreateInteractionFlags
    CreateInteractionFlags_Default = 0x00000000
    CreateInteractionFlags_DifferentialPrivacy = 0x00000001
    CreateInteractionFlags_UseApprox = 0x00000002

    # CalcInteractionFlags
    CalcInteractionFlags_Default = 0x00000000
    CalcInteractionFlags_Purify = 0x00000001
    CalcInteractionFlags_DisableNewton = 0x00000002
    CalcInteractionFlags_Full = 0x00000004

    # AccelerationFlags
    AccelerationFlags_NONE = 0x00000000
    AccelerationFlags_Nvidia = 0x00000001
    AccelerationFlags_AVX2 = 0x00000002
    AccelerationFlags_AVX512F = 0x00000004
    AccelerationFlags_IntelSIMD = AccelerationFlags_AVX2 | AccelerationFlags_AVX512F
    AccelerationFlags_SIMD = AccelerationFlags_IntelSIMD
    AccelerationFlags_GPU = AccelerationFlags_Nvidia
    AccelerationFlags_ALL = 0xFFFFFFFF

    # Tasks
    Task_Ranking = -3
    Task_Regression = -2
    Task_Unknown = -1
    Task_GeneralClassification = 0
    Task_MonoClassification = 1
    Task_BinaryClassification = 2
    Task_MulticlassPlus = 3

    # Objectives
    Objective_Other = 0
    Objective_MonoClassification = 1
    Objective_LogLossBinary = 2
    Objective_LogLossMulticlass = 3
    Objective_Rmse = 4

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
            _log.info("EBM lib loading.")
            native = Native()
            native._initialize(is_debug=is_debug)
            Native._native = native
        return Native._native

    @staticmethod
    def _make_pointer(array, dtype, ndim=1, is_null_allowed=False):
        # using ndpointer creates cyclic garbage references which could clog up
        # our memory and require extra garbage collections.  This function avoids that

        # array MUST be an object with a reference that will remain valid for the duration
        # of the pointer's use. We do not create a new reference to array

        if array is None:
            if not is_null_allowed:  # pragma: no cover
                msg = "array cannot be None"
                raise ValueError(msg)
            return None

        if not isinstance(array, np.ndarray):  # pragma: no cover
            msg = "array should be an ndarray"
            raise ValueError(msg)

        if array.dtype != dtype:  # pragma: no cover
            msg = f"array should be an ndarray of type {dtype}, but is type {array.dtype.type}"
            raise ValueError(msg)

        # if the data is transposed, Fortran ordered, or has strides, we can't use it
        if not array.flags.c_contiguous:  # pragma: no cover
            msg = "array should be a contiguous ndarray"
            raise ValueError(msg)

        if ndim is not None and array.ndim != ndim:  # pragma: no cover
            msg = f"array should have {ndim} dimensions"
            raise ValueError(msg)

        # ctypes.data will return an interor pointer when given an array that is sliced,
        # so arr[1:-1] will return a pointer to the 1st element within arr.
        return array.ctypes.data

    @staticmethod
    def _get_native_exception(error_code, native_function):  # pragma: no cover
        if error_code == -1:
            return Exception(f"Out of memory in {native_function}")
        if error_code == -2:
            return Exception(f"Unexpected internal error in {native_function}")
        if error_code == -3:
            return Exception(f"Illegal native parameter value in {native_function}")
        if error_code == -4:
            return Exception(f"User native parameter value error in {native_function}")
        if error_code == -5:
            return Exception(f"Thread start failed in {native_function}")
        if error_code == -10:
            return Exception(f"Objective constructor exception in {native_function}")
        if error_code == -11:
            return Exception("Objective parameter unknown")
        if error_code == -12:
            return Exception("Objective parameter value malformed")
        if error_code == -13:
            return Exception("Objective parameter value out of range")
        if error_code == -14:
            return Exception("Objective parameter mismatch")
        if error_code == -15:
            return Exception("Unrecognized objective type")
        if error_code == -16:
            return Exception("Illegal objective registration name")
        if error_code == -17:
            return Exception("Illegal objective parameter name")
        if error_code == -18:
            return Exception("Duplicate objective parameter name")
        if error_code == -19:
            return Exception("Differential Privacy not supported by the objective")
        if error_code == -20:
            return Exception(
                "Differential Privacy not supported by an objective parameter value"
            )
        if error_code == -21:
            return Exception("Illegal value in y for the objective")
        return Exception(
            f"Unrecognized native return code {error_code} in {native_function}"
        )

    @staticmethod
    def get_count_scores_c(n_classes):
        return n_classes if n_classes >= Native.Task_MulticlassPlus else 1

    def set_logging(self, level=None):
        # NOTE: Not part of code coverage. It runs in tests, but isn't registered for some reason.
        def native_log(trace_level, message):  # pragma: no cover
            try:
                message = message.decode("ascii")

                if trace_level == self._Trace_Error:
                    _log.error(message)
                elif trace_level == self._Trace_Warning:
                    _log.warning(message)
                elif trace_level == self._Trace_Info:
                    _log.info(message)
                elif trace_level == self._Trace_Verbose:
                    _log.debug(message)
            except:  # noqa: E722
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
        self._unsafe.CleanFloats(
            len(val_array), Native._make_pointer(val_array, np.float64)
        )
        return val_array[0]

    def flat_mean(self, vals, weights=None):
        if weights is not None:
            if vals.shape != weights.shape:
                msg = "vals and weights must have the same shape to call flat_mean."
                raise Exception(msg)

        n_tensor_bins = math.prod(vals.shape)

        mean_result = ct.c_double(np.nan)

        return_code = self._unsafe.SafeMean(
            n_tensor_bins,
            1,
            Native._make_pointer(vals, np.float64, None),
            Native._make_pointer(weights, np.float64, None, True),
            ct.byref(mean_result),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "SafeMean")

        return mean_result

    def safe_mean(self, tensor, weights=None):
        n_bags = tensor.shape[0]
        if weights is not None:
            if weights.ndim != 1:
                msg = (
                    f"weights must be 1 dimensional, but has {weights.ndim} dimensions."
                )
                raise Exception(msg)
            if len(weights) != n_bags:
                msg = "weights must contain the same number of items as there are bags in tensor."
                raise Exception(msg)

        n_tensor_bins = 1
        for n_bins in tensor.shape[1:]:
            n_tensor_bins *= n_bins

        mean_result = np.empty(tensor.shape[1:] if tensor.ndim > 1 else 1, np.float64)

        return_code = self._unsafe.SafeMean(
            n_bags,
            n_tensor_bins,
            Native._make_pointer(tensor, np.float64, None),
            Native._make_pointer(weights, np.float64, is_null_allowed=True),
            Native._make_pointer(mean_result, np.float64, None),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "SafeMean")

        return mean_result

    def safe_stddev(self, tensor, weights=None):
        n_bags = tensor.shape[0]
        if weights is not None:
            if weights.ndim != 1:
                msg = (
                    f"weights must be 1 dimensional, but has {weights.ndim} dimensions."
                )
                raise Exception(msg)
            if len(weights) != n_bags:
                msg = "weights must contain the same number of items as there are bags in tensor."
                raise Exception(msg)

        n_tensor_bins = 1
        for n_bins in tensor.shape[1:]:
            n_tensor_bins *= n_bins

        stddev_result = np.empty(tensor.shape[1:] if tensor.ndim > 1 else 1, np.float64)

        return_code = self._unsafe.SafeStandardDeviation(
            n_bags,
            n_tensor_bins,
            Native._make_pointer(tensor, np.float64, None),
            Native._make_pointer(weights, np.float64, is_null_allowed=True),
            Native._make_pointer(stddev_result, np.float64, None),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "SafeStandardDeviation")

        return stddev_result

    def create_rng(self, random_state):
        if random_state is None:
            return None  # non-deterministic

        if random_state < -2147483648 or random_state > 2147483647:
            msg = f'random_state of "{random_state}" must be cleaned to be a 32-bit signed integer before calling create_rng'
            _log.error(msg)
            raise Exception(msg)

        n_bytes = self._unsafe.MeasureRNG()
        rng = np.empty(n_bytes, np.ubyte)

        self._unsafe.InitRNG(random_state, Native._make_pointer(rng, np.ubyte))
        return rng

    def copy_rng(self, rng):
        if rng is None:
            return None  # non-deterministic

        n_bytes = self._unsafe.MeasureRNG()
        rngCopy = np.empty(n_bytes, np.ubyte)

        self._unsafe.CopyRNG(
            Native._make_pointer(rng, np.ubyte), Native._make_pointer(rngCopy, np.ubyte)
        )
        return rngCopy

    def branch_rng(self, rng):
        if rng is None:
            return None  # non-deterministic

        n_bytes = self._unsafe.MeasureRNG()
        rngBranch = np.empty(n_bytes, np.ubyte)

        self._unsafe.BranchRNG(
            Native._make_pointer(rng, np.ubyte),
            Native._make_pointer(rngBranch, np.ubyte),
        )
        return rngBranch

    def generate_seed(self, rng):
        # Unlike our other functions, this will generate a 32-bit seed even if rng is None
        seed = ct.c_int32(0)
        return_code = self._unsafe.GenerateSeed(
            Native._make_pointer(rng, np.ubyte, is_null_allowed=True), ct.byref(seed)
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
            Native._make_pointer(random_numbers, np.float64),
        )

        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "GenerateGaussianRandom")

        return random_numbers

    def shuffle(self, rng, vals):
        return_code = self._unsafe.Shuffle(
            Native._make_pointer(rng, np.ubyte, is_null_allowed=True),
            len(vals),
            Native._make_pointer(vals, np.int64),
        )

        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "Shuffle")

    def measure_impurity(self, scores, weights):
        shape_all = scores.shape
        shape_classless = scores.shape
        n_multi_scores = 1
        if len(shape_all) == len(weights.shape) + 1:
            # multiclass
            n_multi_scores = shape_all[-1]
            shape_classless = shape_all[:-1]

        if shape_classless != weights.shape:
            msg = f"scores with shape {scores.shape} needs to match the weights with shape {weights.shape}."
            raise Exception(msg)

        if not scores.flags.c_contiguous:
            scores = scores.copy()

        if not weights.flags.c_contiguous:
            weights = weights.copy()

        shape_array = np.array(tuple(reversed(shape_classless)), np.int64)
        impurities = np.empty(n_multi_scores, np.float64)

        for i in range(n_multi_scores):
            impurity = self._unsafe.MeasureImpurity(
                n_multi_scores,
                i,
                len(shape_classless),
                Native._make_pointer(shape_array, np.int64),
                Native._make_pointer(weights, np.float64, None),
                Native._make_pointer(scores, np.float64, None),
            )

            if impurity < 0.0:  # pragma: no cover
                raise Native._get_native_exception(int(impurity), "MeasureImpurity")

            impurities[i] = impurity

        return impurities

    def purify(self, scores, weights, tolerance, is_randomized):
        if np.isnan(tolerance) or tolerance < 0.0 or tolerance >= 1.0:
            msg = (
                f"tolerance must be between 0.0 and less than 1.0, but is {tolerance}."
            )
            raise Exception(msg)

        if is_randomized is not True and is_randomized is not False:
            msg = "is_randomized must be True or False."
            raise Exception(msg)

        shape_all = scores.shape
        shape_classless = scores.shape
        n_multi_scores = 1
        if len(shape_all) == len(weights.shape) + 1:
            # multiclass
            n_multi_scores = shape_all[-1]
            shape_classless = shape_all[:-1]

        if shape_classless != weights.shape:
            msg = f"scores with shape {scores.shape} needs to match the weights with shape {weights.shape}."
            raise Exception(msg)

        intercept = np.zeros(n_multi_scores, np.float64)

        impurity = None
        if len(shape_classless) >= 2:
            n_tensor = 1
            for n_bins in shape_classless:
                n_tensor *= n_bins

            if n_tensor == 0:
                return [
                    np.zeros(shape_all[:i] + shape_all[i + 1 :], np.float64)
                    for i in range(len(shape_classless) - 1, -1, -1)
                ], intercept

            n_impurity_scores = 0
            for n_bins in shape_classless:
                n_impurity_scores += n_tensor // n_bins
            n_impurity_scores *= n_multi_scores
            impurity = np.empty(n_impurity_scores, np.float64)
        shape_array = np.array(tuple(reversed(shape_classless)), np.int64)

        is_multiclass_normalization = True

        return_code = self._unsafe.Purify(
            tolerance,
            is_randomized,
            is_multiclass_normalization,
            n_multi_scores,
            len(shape_classless),
            Native._make_pointer(shape_array, np.int64),
            Native._make_pointer(weights, np.float64, None),
            Native._make_pointer(scores, np.float64, None),
            Native._make_pointer(impurity, np.float64, is_null_allowed=True),
            Native._make_pointer(intercept, np.float64),
        )

        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "Purify")

        impurities = []
        if len(shape_classless) >= 2:
            base_idx = 0
            for exclude_idx in range(len(shape_classless) - 1, -1, -1):
                count = n_tensor // shape_classless[exclude_idx]
                count *= n_multi_scores
                impure_shape = list(shape_all)
                del impure_shape[exclude_idx]
                impurities.append(
                    impurity[base_idx : base_idx + count].reshape(tuple(impure_shape))
                )
                base_idx += count

        return impurities, intercept

    def get_histogram_cut_count(self, X_col):
        return self._unsafe.GetHistogramCutCount(
            X_col.shape[0], Native._make_pointer(X_col, np.float64)
        )

    def cut_uniform(self, X_col, max_cuts):
        if max_cuts < 0:
            msg = f"max_cuts can't be negative: {max_cuts}."
            raise Exception(msg)

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
            msg = f"max_cuts can't be negative: {max_cuts}."
            raise Exception(msg)

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

        return cuts[: count_cuts.value]

    def cut_winsorized(self, X_col, max_cuts):
        if max_cuts < 0:
            msg = f"max_cuts can't be negative: {max_cuts}."
            raise Exception(msg)

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

        return cuts[: count_cuts.value]

    def suggest_graph_bounds(
        self, cuts, min_feature_val=np.nan, max_feature_val=np.nan
    ):
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
            cuts[0] if len(cuts) > 0 else np.nan,
            cuts[-1] if len(cuts) > 0 else np.nan,
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

    def measure_feature(self, n_bins, is_missing, is_unseen, is_nominal, bin_indexes):
        n_bytes = self._unsafe.MeasureFeature(
            n_bins,
            is_missing,
            is_unseen,
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

    def fill_feature(
        self, n_bins, is_missing, is_unseen, is_nominal, bin_indexes, dataset
    ):
        return_code = self._unsafe.FillFeature(
            n_bins,
            is_missing,
            is_unseen,
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

    def extract_nominals(self, dataset):
        _, n_features, _, _ = self.extract_dataset_header(dataset)

        nominals = np.empty(n_features, np.int32, order="C")

        return_code = self._unsafe.ExtractNominals(
            Native._make_pointer(dataset, np.ubyte),
            n_features,
            Native._make_pointer(nominals, np.int32),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "ExtractNominals")

        return nominals.astype(np.bool_)

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
        self, rng, count_training_samples, count_validation_samples
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
        self, rng, n_classes, count_training_samples, count_validation_samples, targets
    ):
        count_samples = count_training_samples + count_validation_samples

        if len(targets) != count_samples:
            msg = "count_training_samples + count_validation_samples should be equal to len(targets)"
            raise ValueError(msg)

        bag = np.empty(count_samples, dtype=np.int8, order="C")

        if not targets.flags.c_contiguous:
            # targets could be a slice with a stride. We need contiguous for C
            targets = targets.copy()

        return_code = self._unsafe.SampleWithoutReplacementStratified(
            Native._make_pointer(rng, np.ubyte, is_null_allowed=True),
            n_classes,
            count_training_samples,
            count_validation_samples,
            Native._make_pointer(targets, np.int64),
            Native._make_pointer(bag, np.int8),
        )

        if return_code:  # pragma: no cover
            raise Native._get_native_exception(
                return_code, "SampleWithoutReplacementStratified"
            )

        return bag

    def determine_task(self, objective):
        task = ct.c_int64(Native.Task_Unknown)

        return_code = self._unsafe.DetermineTask(
            objective.encode("ascii"),
            ct.byref(task),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "DetermineTask")

        task = self._unsafe.GetTaskStr(task.value)
        if not task:  # pragma: no cover
            msg = "internal error in call to GetTaskStr"
            _log.error(msg)
            raise Exception(msg)

        return task.decode("ascii")

    def determine_link(self, flags, objective, n_classes):
        objective_code = ct.c_int32(Native.Objective_Other)
        link = ct.c_int32(0)
        link_param = ct.c_double(np.nan)

        return_code = self._unsafe.DetermineLinkFunction(
            flags,
            objective.encode("ascii"),
            n_classes,
            ct.byref(objective_code),
            ct.byref(link),
            ct.byref(link_param),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "DetermineLinkFunction")

        link = self._unsafe.GetLinkFunctionStr(link.value)
        if not link:  # pragma: no cover
            msg = "internal error in call to GetLinkFunctionStr"
            _log.error(msg)
            raise Exception(msg)

        return (objective_code.value, link.decode("ascii"), link_param.value)

    @staticmethod
    def _get_ebm_lib_path(debug=False):
        """Returns filepath of core EBM library.

        Returns:
            A string representing filepath.
        """

        plat = platform.system()
        if plat == "Windows":  # pragma: no cover
            extension = ".dll"
        elif plat == "Linux":  # pragma: no cover
            extension = ".so"
        elif plat == "Darwin":  # pragma: no cover
            extension = ".dylib"
        else:
            msg = f"Unsupported platform {plat}"
            _log.error(msg)
            raise Exception(msg)

        if debug:
            extension = "_debug" + extension

        machine = platform.machine()

        bitsize = struct.calcsize("P") * 8
        is_64_bit = bitsize == 64

        _log.info(
            f"Finding library for {plat}, {machine}, bitsize={bitsize}, debug={debug}"
        )

        interpret_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        root_path = os.path.join(interpret_path, "root")
        if os.path.isdir(root_path):
            bld_path = os.path.join(root_path, "bld")
            if os.path.isdir(bld_path):
                lib_path = os.path.join(bld_path, "lib")
                if os.path.isdir(lib_path):
                    # first check for the general name
                    lib_file = os.path.join(lib_path, "libebm" + extension)
                    if os.path.isfile(lib_file):
                        _log.info(f"Loading EBM library {lib_file!s}")
                        return lib_file

                    # next check for a specific platform
                    lib_file = None
                    if plat == "Linux" and machine == "x86_64" and is_64_bit:
                        lib_file = "libebm_linux_x64"
                    elif plat == "Windows" and machine == "AMD64" and is_64_bit:
                        lib_file = "libebm_win_x64"
                    elif plat == "Darwin" and machine == "x86_64" and is_64_bit:
                        lib_file = "libebm_mac_x64"
                    elif plat == "Darwin" and machine == "arm64":
                        lib_file = "libebm_mac_arm"

                    if lib_file is not None:
                        lib_file = os.path.join(lib_path, lib_file + extension)
                        if os.path.isfile(lib_file):
                            _log.info(f"Loading EBM library {lib_file!s}")
                            return lib_file
        else:
            # root should at least contain the visualization javascript. If root does
            # not exist, then we're running from a cloned git repo

            root_path = os.path.join(interpret_path, "..", "..", "..")
            if os.path.isdir(root_path):
                bld_path = os.path.join(root_path, "bld")
                if os.path.isdir(bld_path):
                    lib_path = os.path.join(bld_path, "lib")
                    if os.path.isdir(lib_path):
                        # first check for the general name
                        lib_file = os.path.join(lib_path, "libebm" + extension)
                        if os.path.isfile(lib_file):
                            _log.info(f"Loading EBM library {lib_file!s}")
                            return lib_file

                    # next check for a specific platform
                    lib_file = None
                    if plat == "Linux" and machine == "x86_64" and is_64_bit:
                        lib_file = "libebm_linux_x64"
                    elif plat == "Windows" and machine == "AMD64" and is_64_bit:
                        lib_file = "libebm_win_x64"
                    elif plat == "Darwin" and machine == "x86_64" and is_64_bit:
                        lib_file = "libebm_mac_x64"
                    elif plat == "Darwin" and machine == "arm64":
                        lib_file = "libebm_mac_arm"

                    if lib_file is not None:
                        lib_file = os.path.join(lib_path, lib_file + extension)
                        if os.path.isfile(lib_file):
                            _log.info(f"Loading EBM library {lib_file!s}")
                            return lib_file

            # TODO: if the library does not exists, build it using build.sh or build.bat

        env_path = sys.base_prefix
        lib_path = None
        if os.path.isdir(env_path):
            if plat == "Windows":
                env_path = os.path.join(env_path, "Library")
                if os.path.isdir(env_path):
                    env_path = os.path.join(env_path, "bin")
                    if os.path.isdir(env_path):
                        lib_path = env_path
            else:
                env_path = os.path.join(env_path, "lib")
                if os.path.isdir(env_path):
                    lib_path = env_path

        if lib_path is not None:
            lib_file = os.path.join(lib_path, "libebm" + extension)
            if os.path.isfile(lib_file):
                _log.info(f"Loading EBM library {lib_file!s}")
                return lib_file

        if debug:
            msg = "Could not find DEBUG libebm shared library. Consider setting debug=False"
        else:
            msg = "Could not find libebm shared library."
        _log.error(msg)
        raise Exception(msg)

    def _initialize(self, is_debug):
        self.is_debug = is_debug
        self.approximates = False

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

        self._unsafe.SafeMean.argtypes = [
            # int64_t countBags
            ct.c_int64,
            # int64_t countTensorBins
            ct.c_int64,
            # double * vals
            ct.c_void_p,
            # double * weights
            ct.c_void_p,
            # double * tensorOut
            ct.c_void_p,
        ]
        self._unsafe.SafeMean.restype = ct.c_int32

        self._unsafe.SafeStandardDeviation.argtypes = [
            # int64_t countBags
            ct.c_int64,
            # int64_t countTensorBins
            ct.c_int64,
            # double * vals
            ct.c_void_p,
            # double * weights
            ct.c_void_p,
            # double * tensorOut
            ct.c_void_p,
        ]
        self._unsafe.SafeStandardDeviation.restype = ct.c_int32

        self._unsafe.MeasureRNG.argtypes = []
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

        self._unsafe.Shuffle.argtypes = [
            # void * rng
            ct.c_void_p,
            # int64_t count
            ct.c_int64,
            # int64_t * randomOut
            ct.c_void_p,
        ]
        self._unsafe.Shuffle.restype = ct.c_int32

        self._unsafe.MeasureImpurity.argtypes = [
            # int64_t countMultiScores
            ct.c_int64,
            # int64_t indexMultiScore
            ct.c_int64,
            # int64_t countDimensions
            ct.c_int64,
            # int64_t * dimensionLengths
            ct.c_void_p,
            # double * weights
            ct.c_void_p,
            # double * scores
            ct.c_void_p,
        ]
        self._unsafe.MeasureImpurity.restype = ct.c_double

        self._unsafe.Purify.argtypes = [
            # double tolerance
            ct.c_double,
            # int32_t isRandomized
            ct.c_int32,
            # int32_t isMulticlassNormalization
            ct.c_int32,
            # int64_t countMultiScores
            ct.c_int64,
            # int64_t countDimensions
            ct.c_int64,
            # int64_t * dimensionLengths
            ct.c_void_p,
            # double * weights
            ct.c_void_p,
            # double * scoresInOut
            ct.c_void_p,
            # double * impuritiesOut
            ct.c_void_p,
            # double * interceptOut
            ct.c_void_p,
        ]
        self._unsafe.Purify.restype = ct.c_int32

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
            # int32_t isUnseen
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
            # int32_t isUnseen
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

        self._unsafe.ExtractNominals.argtypes = [
            # void * dataSet
            ct.c_void_p,
            # int64_t countFeaturesVerify
            ct.c_int64,
            # int32_t * nominalsOut
            ct.c_void_p,
        ]
        self._unsafe.ExtractNominals.restype = ct.c_int32

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

        self._unsafe.DetermineTask.argtypes = [
            # char * objective
            ct.c_char_p,
            # int64_t * taskOut
            ct.c_void_p,
        ]
        self._unsafe.DetermineTask.restype = ct.c_int32

        self._unsafe.GetTaskStr.argtypes = [
            # int64_t task
            ct.c_int64,
        ]
        self._unsafe.GetTaskStr.restype = ct.c_char_p

        self._unsafe.DetermineLinkFunction.argtypes = [
            # LinkFlags flags
            ct.c_int32,
            # char * objective
            ct.c_char_p,
            # int64_t countClasses
            ct.c_int64,
            # int32_t * objectiveOut
            ct.c_void_p,
            # int32_t * linkOut
            ct.c_void_p,
            # double * linkParamOut
            ct.c_void_p,
        ]
        self._unsafe.DetermineLinkFunction.restype = ct.c_int32

        self._unsafe.GetLinkFunctionStr.argtypes = [
            # int32_t link
            ct.c_int32,
        ]
        self._unsafe.GetLinkFunctionStr.restype = ct.c_char_p

        self._unsafe.CreateBooster.argtypes = [
            # void * rng
            ct.c_void_p,
            # void * dataSet
            ct.c_void_p,
            # double * intercept
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
            # CreateBoosterFlags flags
            ct.c_int32,
            # AccelerationFlags acceleration
            ct.c_int32,
            # char * objective
            ct.c_char_p,
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
            # TermBoostFlags flags
            ct.c_int32,
            # double learningRate
            ct.c_double,
            # int64_t minSamplesLeaf
            ct.c_int64,
            # double minHessian
            ct.c_double,
            # double regAlpha
            ct.c_double,
            # double regLambda
            ct.c_double,
            # double maxDeltaStep
            ct.c_double,
            # int64_t minCategorySamples
            ct.c_int64,
            # double categoricalSmoothing
            ct.c_double,
            # int64_t maxCategoricalThreshold
            ct.c_int64,
            # double categoricalInclusionPercent
            ct.c_double,
            # int64_t * leavesMax
            ct.c_void_p,
            # MonotoneDirection * direction
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
            # int64_t * splitsOut
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
            # double * intercept
            ct.c_void_p,
            # int8_t * bag
            ct.c_void_p,
            # double * initScores
            ct.c_void_p,
            # CreateInteractionFlags flags
            ct.c_int32,
            # AccelerationFlags acceleration
            ct.c_int32,
            # char * objective
            ct.c_char_p,
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
            # CalcInteractionFlags flags
            ct.c_int32,
            # int64_t maxCardinality
            ct.c_int64,
            # int64_t minSamplesLeaf
            ct.c_int64,
            # double minHessian
            ct.c_double,
            # double regAlpha
            ct.c_double,
            # double regLambda
            ct.c_double,
            # double maxDeltaStep
            ct.c_double,
            # double * avgInteractionStrengthOut
            ct.POINTER(ct.c_double),
        ]
        self._unsafe.CalcInteractionStrength.restype = ct.c_int32


class Booster(AbstractContextManager):
    """Lightweight wrapper for EBM C boosting code."""

    def __init__(
        self,
        dataset,
        intercept,
        bag,
        init_scores,
        term_features,
        n_inner_bags,
        rng,
        create_booster_flags,
        objective,
        acceleration,
        experimental_params,
    ):
        """Initializes internal wrapper for EBM C code.

        Args:
            dataset: binned data in a compressed native form
            intercept: initial intercept
            bag: definition of what data is included. 1 = training, -1 = validation, 0 = not included
            init_scores: predictions from a prior predictor
                that this class will boost on top of.  For regression
                there is 1 score per sample.  For binary classification
                there is one score.  For multiclass there are n_classes scores
            term_features: List of term feature indexes
            n_inner_bags: number of inner bags.
            rng: native random number generator
            experimental_params: unused data that can be passed into the native layer for debugging
        """

        self.dataset = dataset
        self.intercept = intercept
        self.bag = bag
        self.init_scores = init_scores
        self.term_features = term_features
        self.n_inner_bags = n_inner_bags
        self.rng = rng
        self.create_booster_flags = create_booster_flags
        self.objective = objective
        self.acceleration = acceleration
        self.experimental_params = experimental_params

        # start off with an invalid _term_idx
        self._term_idx = -2

    def __enter__(self):
        _log.info("Booster allocation start")

        if self.objective is None or len(self.objective.strip()) == 0:
            msg = "objective must be specified"
            _log.error(msg)
            raise Exception(msg)

        dimension_counts = np.empty(len(self.term_features), ct.c_int64)
        feature_indexes = []
        for term_idx, feature_idxs in enumerate(self.term_features):
            dimension_counts[term_idx] = len(feature_idxs)
            feature_indexes.extend(feature_idxs)
        feature_indexes = np.array(feature_indexes, ct.c_int64)

        native = Native.get_native_singleton()

        n_samples, n_features, n_weights, n_targets = native.extract_dataset_header(
            self.dataset
        )

        if n_weights not in (0, 1):  # pragma: no cover
            msg = "n_weights must be 0 or 1"
            raise ValueError(msg)

        if n_targets != 1:  # pragma: no cover
            msg = "n_targets must be 1"
            raise ValueError(msg)

        class_counts = native.extract_target_classes(self.dataset, n_targets)
        n_class_scores = sum(
            Native.get_count_scores_c(n_classes) for n_classes in class_counts
        )
        self._n_class_scores = n_class_scores

        bin_counts = native.extract_bin_counts(self.dataset, n_features)
        self._term_shapes = []
        for feature_idxs in self.term_features:
            dimensions = [bin_counts[feature_idx] for feature_idx in feature_idxs]

            # multiclass needs a second dimension
            if n_class_scores != 1:
                dimensions.append(n_class_scores)

            self._term_shapes.append(tuple(dimensions))

        n_bagged_samples = n_samples
        if self.bag is not None:
            if self.bag.shape[0] != n_samples:  # pragma: no cover
                msg = "bag should be len(n_samples)"
                raise ValueError(msg)
            n_bagged_samples = np.count_nonzero(self.bag)

        init_scores = self.init_scores
        if init_scores is not None:
            if not init_scores.flags.c_contiguous:
                # init_scores could be a slice that has a stride.  We need contiguous for caling into C
                init_scores = init_scores.copy()

            if init_scores.shape[0] != n_bagged_samples:  # pragma: no cover
                msg = "init_scores should have the same length as the number of non-zero bag entries"
                raise ValueError(msg)

            if n_class_scores == 1:
                if init_scores.ndim != 1:  # pragma: no cover
                    msg = "init_scores should have ndim == 1 for regression or binary classification"
                    raise ValueError(msg)
            else:
                if init_scores.ndim != 2:  # pragma: no cover
                    msg = "init_scores should have ndim == 2 for multiclass"
                    raise ValueError(msg)
                if init_scores.shape[1] != n_class_scores:  # pragma: no cover
                    msg = f"init_scores should have {n_class_scores} scores"
                    raise ValueError(msg)

        intercept = self.intercept
        if intercept is not None:
            if len(intercept) != n_class_scores:  # pragma: no cover
                msg = f"intercept should have {n_class_scores} scores"
                raise ValueError(msg)

            if not intercept.flags.c_contiguous:
                # intercept could be a slice that has a stride.  We need contiguous for caling into C
                intercept = intercept.copy()

        flags = self.create_booster_flags
        if native.approximates:
            flags |= Native.CreateBoosterFlags_UseApprox

        # Allocate external resources
        booster_handle = ct.c_void_p(0)
        return_code = native._unsafe.CreateBooster(
            Native._make_pointer(self.rng, np.ubyte, is_null_allowed=True),
            Native._make_pointer(self.dataset, np.ubyte),
            Native._make_pointer(intercept, np.float64, 1, True),
            Native._make_pointer(self.bag, np.int8, is_null_allowed=True),
            Native._make_pointer(
                init_scores, np.float64, 1 if n_class_scores == 1 else 2, True
            ),
            len(dimension_counts),
            Native._make_pointer(dimension_counts, np.int64),
            Native._make_pointer(feature_indexes, np.int64),
            self.n_inner_bags,
            flags,
            self.acceleration,
            self.objective.encode("ascii"),
            Native._make_pointer(
                self.experimental_params, np.float64, is_null_allowed=True
            ),
            ct.byref(booster_handle),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "CreateBooster")

        self._booster_handle = booster_handle.value

        _log.info("Booster allocation end")
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        """Deallocates C objects used to boost EBM."""
        _log.info("Deallocation boosting start")

        booster_handle = getattr(self, "_booster_handle", None)
        if booster_handle:
            native = Native.get_native_singleton()
            self._booster_handle = None
            native._unsafe.FreeBooster(booster_handle)

        _log.info("Deallocation boosting end")

    def generate_term_update(
        self,
        rng,
        term_idx,
        term_boost_flags,
        learning_rate,
        min_samples_leaf,
        min_hessian,
        reg_alpha,
        reg_lambda,
        max_delta_step,
        min_cat_samples,
        cat_smooth,
        max_cat_threshold,
        cat_include,
        max_leaves,
        monotone_constraints,
    ):
        """Generates a boosting step update per feature
            by growing a shallow decision tree.

        Args:
            term_idx: The index for the term to generate the update for
            term_boost_flags: C interface options
            learning_rate: Learning rate as a float.
            min_samples_leaf: Min observations required to split.
            min_hessian: Min hessian required to split.
            reg_alpha: L1 regularization.
            reg_lambda: L2 regularization.
            max_delta_step: Used to limit the max output of tree leaves. <=0.0 means no constraint.
            min_cat_samples: Min samples to consider category independently
            cat_smooth: Parameter used to determine which categories are included each boosting round and ordering.
            max_cat_threshold: max number of categories to include each boosting round
            cat_include: percentage of categories to include in each boosting round
            max_leaves: Max leaf nodes on feature step.
            monotone_constraints: monotone constraints (1=increasing, 0=none, -1=decreasing)

        Returns:
            gain for the generated boosting step.
        """

        # _log.debug("Boosting step start")

        self._term_idx = -2

        native = Native.get_native_singleton()

        avg_gain = ct.c_double(0.0)

        if term_idx <= -2:
            msg = f"term_idx cannot be -2 or less. -1 would mean intercept boosting."
            raise ValueError(msg)
        elif term_idx == -1:
            if monotone_constraints is not None:
                msg = f"monotone_constraints should be None for intercept boosting."
                raise ValueError(msg)
            max_leaves_arr = None
        else:
            n_features = len(self.term_features[term_idx])
            if monotone_constraints is not None:
                if len(monotone_constraints) != n_features:
                    msg = f"monotone_constraints should have the same length {len(monotone_constraints)} as the number of features {n_features}."
                    raise ValueError(msg)
            max_leaves_arr = np.full(
                n_features, max_leaves, dtype=ct.c_int64, order="C"
            )

        return_code = native._unsafe.GenerateTermUpdate(
            Native._make_pointer(rng, np.ubyte, is_null_allowed=True),
            self._booster_handle,
            term_idx,
            term_boost_flags,
            learning_rate,
            min_samples_leaf,
            min_hessian,
            reg_alpha,
            reg_lambda,
            max_delta_step,
            min_cat_samples,
            cat_smooth,
            max_cat_threshold,
            cat_include,
            Native._make_pointer(max_leaves_arr, np.int64, is_null_allowed=True),
            Native._make_pointer(monotone_constraints, np.int32, is_null_allowed=True),
            ct.byref(avg_gain),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "GenerateTermUpdate")

        self._term_idx = term_idx

        # _log.debug("Boosting step end")
        return avg_gain.value

    def apply_term_update(self):
        """Updates the interal C state with the last model update

        Args:

        Returns:
            Validation loss for the boosting step.
        """
        # _log.debug("Boosting step start")

        if self._term_idx <= -2:
            msg = f"The update needs to be set prior to calling apply_term_update."
            raise ValueError(msg)

        self._term_idx = -2

        native = Native.get_native_singleton()

        avg_validation_metric = ct.c_double(np.inf)
        return_code = native._unsafe.ApplyTermUpdate(
            self._booster_handle,
            ct.byref(avg_validation_metric),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "ApplyTermUpdate")

        # _log.debug("Boosting step end")
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
        splits = []
        if self._term_idx != -1:
            if self._term_idx <= -2:  # pragma: no cover
                msg = "invalid internal self._term_idx"
                raise RuntimeError(msg)

            feature_idxs = self.term_features[self._term_idx]
            for dimension_idx in range(len(feature_idxs)):
                splits_dimension = self._get_term_update_splits_dimension(dimension_idx)
                splits.append(splits_dimension)

        return splits

    def _get_best_term_scores(self, term_idx):
        """Returns best model/function according to validation set
            for a given term.

        Args:
            term_idx: The index for the term.

        Returns:
            An ndarray that represents the model.
        """

        native = Native.get_native_singleton()

        shape = self._term_shapes[term_idx]
        term_scores = np.full(shape, -np.inf, np.float64, "C")

        return_code = native._unsafe.GetBestTermScores(
            self._booster_handle,
            term_idx,
            Native._make_pointer(term_scores, np.float64, None),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "GetBestTermScores")

        return term_scores

    def _get_current_term_scores(self, term_idx):
        """Returns current model/function according to validation set
            for a given term.

        Args:
            term_idx: The index for the term.

        Returns:
            An ndarray that represents the model.
        """

        native = Native.get_native_singleton()

        shape = self._term_shapes[term_idx]
        term_scores = np.full(shape, -np.inf, np.float64, "C")

        return_code = native._unsafe.GetCurrentTermScores(
            self._booster_handle,
            term_idx,
            Native._make_pointer(term_scores, np.float64, None),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "GetCurrentTermScores")

        return term_scores

    def _get_term_update_splits_dimension(self, dimension_index):
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

        return splits[: count_splits.value]

    def get_term_update(self):
        if self._term_idx <= -2:  # pragma: no cover
            msg = "invalid internal self._term_idx"
            raise RuntimeError(msg)

        native = Native.get_native_singleton()

        shape = self._n_class_scores
        if self._term_idx != -1:
            shape = self._term_shapes[self._term_idx]
        update_scores = np.full(shape, -np.inf, np.float64, "C")

        return_code = native._unsafe.GetTermUpdate(
            self._booster_handle,
            Native._make_pointer(update_scores, np.float64, None),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "GetTermUpdate")

        return update_scores

    def set_term_update(self, term_idx, update_scores):
        self._term_idx = -2

        shape = self._n_class_scores
        if term_idx != -1:
            shape = self._term_shapes[term_idx]

        if shape != update_scores.shape:  # pragma: no cover
            msg = "incorrect tensor shape in call to set_term_update"
            raise ValueError(msg)

        native = Native.get_native_singleton()
        return_code = native._unsafe.SetTermUpdate(
            self._booster_handle,
            term_idx,
            Native._make_pointer(update_scores, np.float64, len(shape)),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "SetTermUpdate")

        self._term_idx = term_idx


class InteractionDetector(AbstractContextManager):
    """Lightweight wrapper for EBM C interaction code."""

    def __init__(
        self,
        dataset,
        intercept,
        bag,
        init_scores,
        create_interaction_flags,
        objective,
        acceleration,
        experimental_params,
    ):
        """Initializes internal wrapper for EBM C code.

        Args:
            dataset: binned data in a compressed native form
            intercept: prediction shift
            bag: definition of what data is included. 1 = training, -1 = validation, 0 = not included
            init_scores: predictions from a prior predictor
                that this class will boost on top of.  For regression
                there is 1 score per sample.  For binary classification
                there is one score.  For multiclass there are n_classes scores
            experimental_params: unused data that can be passed into the native layer for debugging

        """

        self.dataset = dataset
        self.intercept = intercept
        self.bag = bag
        self.init_scores = init_scores
        self.create_interaction_flags = create_interaction_flags
        self.objective = objective
        self.acceleration = acceleration
        self.experimental_params = experimental_params

    def __enter__(self):
        _log.info("Allocation interaction start")

        if self.objective is None or len(self.objective.strip()) == 0:
            msg = "objective must be specified"
            _log.error(msg)
            raise Exception(msg)

        native = Native.get_native_singleton()

        n_samples, n_features, n_weights, n_targets = native.extract_dataset_header(
            self.dataset
        )

        if n_weights not in (0, 1):  # pragma: no cover
            msg = "n_weights must be 0 or 1"
            raise ValueError(msg)

        if n_targets != 1:  # pragma: no cover
            msg = "n_targets must be 1"
            raise ValueError(msg)

        class_counts = native.extract_target_classes(self.dataset, n_targets)
        n_class_scores = sum(
            Native.get_count_scores_c(n_classes) for n_classes in class_counts
        )

        n_bagged_samples = n_samples
        if self.bag is not None:
            if self.bag.shape[0] != n_samples:  # pragma: no cover
                msg = "bag should be len(n_samples)"
                raise ValueError(msg)
            n_bagged_samples = np.count_nonzero(self.bag)

        init_scores = self.init_scores
        if init_scores is not None:
            if not init_scores.flags.c_contiguous:
                # init_scores could be a slice that has a stride.  We need contiguous for caling into C
                init_scores = init_scores.copy()

            if init_scores.shape[0] != n_bagged_samples:  # pragma: no cover
                msg = "init_scores should have the same length as the number of non-zero bag entries"
                raise ValueError(msg)

            if n_class_scores == 1:
                if init_scores.ndim != 1:  # pragma: no cover
                    msg = "init_scores should have ndim == 1 for regression or binary classification"
                    raise ValueError(msg)
            else:
                if init_scores.ndim != 2:  # pragma: no cover
                    msg = "init_scores should have ndim == 2 for multiclass"
                    raise ValueError(msg)
                if init_scores.shape[1] != n_class_scores:  # pragma: no cover
                    msg = f"init_scores should have {n_class_scores} scores"
                    raise ValueError(msg)

        intercept = self.intercept
        if intercept is not None:
            if len(intercept) != n_class_scores:  # pragma: no cover
                msg = f"intercept should have {n_class_scores} scores"
                raise ValueError(msg)

            if not intercept.flags.c_contiguous:
                # intercept could be a slice that has a stride.  We need contiguous for caling into C
                intercept = intercept.copy()

        flags = self.create_interaction_flags
        if native.approximates:
            flags |= Native.CreateInteractionFlags_UseApprox

        # Allocate external resources
        interaction_handle = ct.c_void_p(0)

        return_code = native._unsafe.CreateInteractionDetector(
            Native._make_pointer(self.dataset, np.ubyte),
            Native._make_pointer(intercept, np.float64, 1, True),
            Native._make_pointer(self.bag, np.int8, 1, True),
            Native._make_pointer(
                init_scores, np.float64, 1 if n_class_scores == 1 else 2, True
            ),
            flags,
            self.acceleration,
            self.objective.encode("ascii"),
            Native._make_pointer(self.experimental_params, np.float64, 1, True),
            ct.byref(interaction_handle),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "CreateInteractionDetector")

        self._interaction_handle = interaction_handle.value

        _log.info("Allocation interaction end")
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        """Deallocates C objects used to determine interactions in EBM."""
        _log.info("Deallocation interaction start")

        interaction_handle = getattr(self, "_interaction_handle", None)
        if interaction_handle:
            native = Native.get_native_singleton()
            self._interaction_handle = None
            native._unsafe.FreeInteractionDetector(interaction_handle)

        _log.info("Deallocation interaction end")

    def calc_interaction_strength(
        self,
        feature_idxs,
        calc_interaction_flags,
        max_cardinality,
        min_samples_leaf,
        min_hessian,
        reg_alpha,
        reg_lambda,
        max_delta_step,
    ):
        """Provides a strength measurement of a feature interaction. Higher is better."""
        _log.info("Fast interaction strength start")

        native = Native.get_native_singleton()

        feature_idxs = np.array(feature_idxs, np.int64)

        strength = ct.c_double(0.0)
        return_code = native._unsafe.CalcInteractionStrength(
            self._interaction_handle,
            len(feature_idxs),
            Native._make_pointer(feature_idxs, np.int64),
            calc_interaction_flags,
            max_cardinality,
            min_samples_leaf,
            min_hessian,
            reg_alpha,
            reg_lambda,
            max_delta_step,
            ct.byref(strength),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "CalcInteractionStrength")

        _log.info("Fast interaction strength end")
        return strength.value
