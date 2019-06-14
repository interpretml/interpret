# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

# TODO: Add unit tests for internal EBM interfacing
import sys
from sys import platform
import ctypes as ct
from numpy.ctypeslib import ndpointer
import numpy as np
import os
import struct
import logging

log = logging.getLogger(__name__)

this = sys.modules[__name__]
this.native = None


class Native:
    """Layer/Class responsible for native function calls."""

    # enum AttributeType : int64_t
    # Ordinal = 0
    AttributeTypeOrdinal = 0
    # Nominal = 1
    AttributeTypeNominal = 1

    class Attribute(ct.Structure):
        _fields_ = [
            # AttributeType attributeType;
            ("attributeType", ct.c_longlong),
            # int64_t hasMissing;
            ("hasMissing", ct.c_longlong),
            # int64_t countStates;
            ("countStates", ct.c_longlong),
        ]

    class AttributeSet(ct.Structure):
        _fields_ = [
            # int64_t countAttributes;
            ("countAttributes", ct.c_longlong)
        ]

    LogFuncType = ct.CFUNCTYPE(None, ct.c_char, ct.c_char_p)

    # const signed char TraceLevelOff = 0;
    TraceLevelOff = 0
    # const signed char TraceLevelError = 1;
    TraceLevelError = 1
    # const signed char TraceLevelWarning = 2;
    TraceLevelWarning = 2
    # const signed char TraceLevelInfo = 3;
    TraceLevelInfo = 3
    # const signed char TraceLevelVerbose = 4;
    TraceLevelVerbose = 4

    def __init__(self, is_debug=False, log_level=None):
        self.is_debug = is_debug
        self.log_level = log_level

        self.lib = ct.cdll.LoadLibrary(self.get_ebm_lib_path(debug=is_debug))
        self.harden_function_signatures()
        self.set_logging(level=log_level)

    def harden_function_signatures(self):
        """ Adds types to function signatures. """
        self.lib.SetLogMessageFunction.argtypes = [
            # void (* fn)(signed char traceLevel, const char * message)
            self.LogFuncType
        ]
        self.lib.SetTraceLevel.argtypes = [
            # signed char traceLevel
            ct.c_char
        ]
        self.lib.InitializeTrainingRegression.argtypes = [
            # int64_t randomSeed
            ct.c_longlong,
            # int64_t countAttributes
            ct.c_longlong,
            # Attribute * attributes
            ct.POINTER(self.Attribute),
            # int64_t countAttributeSets
            ct.c_longlong,
            # AttributeSet * attributeSets
            ct.POINTER(self.AttributeSet),
            # int64_t * attributeSetIndexes
            ndpointer(dtype=ct.c_longlong, flags="F_CONTIGUOUS", ndim=1),
            # int64_t countTrainingCases
            ct.c_longlong,
            # double * trainingTargets
            ndpointer(dtype=ct.c_double, flags="F_CONTIGUOUS", ndim=1),
            # int64_t * trainingData
            ndpointer(dtype=ct.c_longlong, flags="F_CONTIGUOUS", ndim=2),
            # double * trainingPredictionScores
            ndpointer(dtype=ct.c_double, flags="F_CONTIGUOUS", ndim=1),
            # int64_t countValidationCases
            ct.c_longlong,
            # double * validationTargets
            ndpointer(dtype=ct.c_double, flags="F_CONTIGUOUS", ndim=1),
            # int64_t * validationData
            ndpointer(dtype=ct.c_longlong, flags="F_CONTIGUOUS", ndim=2),
            # double * validationPredictionScores
            ndpointer(dtype=ct.c_double, flags="F_CONTIGUOUS", ndim=1),
            # int64_t countInnerBags
            ct.c_longlong,
        ]
        self.lib.InitializeTrainingRegression.restype = ct.c_void_p

        self.lib.InitializeTrainingClassification.argtypes = [
            # int64_t randomSeed
            ct.c_longlong,
            # int64_t countAttributes
            ct.c_longlong,
            # Attribute * attributes
            ct.POINTER(self.Attribute),
            # int64_t countAttributeSets
            ct.c_longlong,
            # AttributeSet2 * attributeSets
            ct.POINTER(self.AttributeSet),
            # int64_t * attributeSetIndexes
            ndpointer(dtype=ct.c_longlong, flags="F_CONTIGUOUS", ndim=1),
            # int64_t countTargetStates
            ct.c_longlong,
            # int64_t countTrainingCases
            ct.c_longlong,
            # int64_t * trainingTargets
            ndpointer(dtype=ct.c_longlong, flags="F_CONTIGUOUS", ndim=1),
            # int64_t * trainingData
            ndpointer(dtype=ct.c_longlong, flags="F_CONTIGUOUS", ndim=2),
            # double * trainingPredictionScores
            ndpointer(dtype=ct.c_double, flags="F_CONTIGUOUS", ndim=1),
            # int64_t countValidationCases
            ct.c_longlong,
            # int64_t * validationTargets
            ndpointer(dtype=ct.c_longlong, flags="F_CONTIGUOUS", ndim=1),
            # int64_t * validationData
            ndpointer(dtype=ct.c_longlong, flags="F_CONTIGUOUS", ndim=2),
            # double * validationPredictionScores
            ndpointer(dtype=ct.c_double, flags="F_CONTIGUOUS", ndim=1),
            # int64_t countInnerBags
            ct.c_longlong,
        ]
        self.lib.InitializeTrainingClassification.restype = ct.c_void_p

        self.lib.TrainingStep.argtypes = [
            # void * tml
            ct.c_void_p,
            # int64_t indexAttributeSet
            ct.c_longlong,
            # double learningRate
            ct.c_double,
            # int64_t countTreeSplitsMax
            ct.c_longlong,
            # int64_t countCasesRequiredForSplitParentMin
            ct.c_longlong,
            # double * trainingWeights
            # ndpointer(dtype=ct.c_double, flags="F_CONTIGUOUS", ndim=1),
            ct.c_void_p,
            # double * validationWeights
            # ndpointer(dtype=ct.c_double, flags="F_CONTIGUOUS", ndim=1),
            ct.c_void_p,
            # double * validationMetricReturn
            ct.POINTER(ct.c_double),
        ]
        self.lib.TrainingStep.restype = ct.c_longlong

        self.lib.GetCurrentModel.argtypes = [
            # void * tml
            ct.c_void_p,
            # int64_t indexAttributeSet
            ct.c_longlong,
        ]
        self.lib.GetCurrentModel.restype = ct.POINTER(ct.c_double)

        self.lib.GetBestModel.argtypes = [
            # void * tml
            ct.c_void_p,
            # int64_t indexAttributeSet
            ct.c_longlong,
        ]
        self.lib.GetBestModel.restype = ct.POINTER(ct.c_double)

        self.lib.FreeTraining.argtypes = [
            # void * tml
            ct.c_void_p
        ]

        self.lib.CancelTraining.argtypes = [
            # void * tml
            ct.c_void_p
        ]

        self.lib.InitializeInteractionClassification.argtypes = [
            # int64_t countAttributes
            ct.c_longlong,
            # Attribute * attributes
            ct.POINTER(self.Attribute),
            # int64_t countTargetStates
            ct.c_longlong,
            # int64_t countCases
            ct.c_longlong,
            # int64_t * targets
            ndpointer(dtype=ct.c_longlong, flags="F_CONTIGUOUS", ndim=1),
            # int64_t * data
            ndpointer(dtype=ct.c_longlong, flags="F_CONTIGUOUS", ndim=2),
            # double * predictionScores
            ndpointer(dtype=ct.c_double, flags="F_CONTIGUOUS", ndim=1),
        ]
        self.lib.InitializeInteractionClassification.restype = ct.c_void_p

        self.lib.InitializeInteractionRegression.argtypes = [
            # int64_t countAttributes
            ct.c_longlong,
            # Attribute * attributes
            ct.POINTER(self.Attribute),
            # int64_t countCases
            ct.c_longlong,
            # double * targets
            ndpointer(dtype=ct.c_double, flags="F_CONTIGUOUS", ndim=1),
            # int64_t * data
            ndpointer(dtype=ct.c_longlong, flags="F_CONTIGUOUS", ndim=2),
            # double * predictionScores
            ndpointer(dtype=ct.c_double, flags="F_CONTIGUOUS", ndim=1),
        ]
        self.lib.InitializeInteractionRegression.restype = ct.c_void_p

        self.lib.GetInteractionScore.argtypes = [
            # void * tmlInteraction
            ct.c_void_p,
            # int64_t countAttributesInCombination
            ct.c_longlong,
            # int64_t * attributeIndexes
            ndpointer(dtype=ct.c_longlong, flags="F_CONTIGUOUS", ndim=1),
            # double * interactionScoreReturn
            ct.POINTER(ct.c_double),
        ]
        self.lib.GetInteractionScore.restype = ct.c_longlong

        self.lib.FreeInteraction.argtypes = [
            # void * tmlInteraction
            ct.c_void_p
        ]

        self.lib.CancelInteraction.argtypes = [
            # void * tmlInteraction
            ct.c_void_p
        ]

    def set_logging(self, level=None):
        def native_log(trace_level, message):
            trace_level = int(trace_level[0])
            message = message.decode("utf-8")

            if trace_level == self.TraceLevelOff:
                pass
            elif trace_level == self.TraceLevelError:
                log.error(message)
            elif trace_level == self.TraceLevelWarning:
                log.warning(message)
            elif trace_level == self.TraceLevelInfo:
                log.info(message)
            elif trace_level == self.TraceLevelVerbose:
                log.debug(message)

        if level is None:
            root = logging.getLogger("interpret")
            level = root.getEffectiveLevel()

        level_dict = {
            logging.DEBUG: self.TraceLevelVerbose,
            logging.INFO: self.TraceLevelInfo,
            logging.WARNING: self.TraceLevelWarning,
            logging.ERROR: self.TraceLevelError,
            logging.NOTSET: self.TraceLevelOff,
            "DEBUG": self.TraceLevelVerbose,
            "INFO": self.TraceLevelInfo,
            "WARNING": self.TraceLevelWarning,
            "ERROR": self.TraceLevelError,
            "NOTSET": self.TraceLevelOff,
        }

        self.typed_log_func = self.LogFuncType(native_log)
        self.lib.SetLogMessageFunction(self.typed_log_func)
        self.lib.SetTraceLevel(ct.c_char(level_dict[level]))

    def get_ebm_lib_path(self, debug=False):
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
                package_path, "lib", "ebmcore_linux_x64{0}.so".format(debug_str)
            )
        elif platform == "win32" and is_64_bit:
            return os.path.join(
                package_path, "lib", "ebmcore_win_x64{0}.dll".format(debug_str)
            )
        elif platform == "darwin" and is_64_bit:
            return os.path.join(
                package_path, "lib", "ebmcore_mac_x64{0}.dylib".format(debug_str)
            )
        else:
            msg = "Platform {0} at {1} bit not supported for EBM".format(
                platform, bitsize
            )
            log.error(msg)
            raise Exception(msg)


class NativeEBM:
    """Lightweight wrapper for EBM C code.
    """

    def __init__(
        self,
        attributes,
        attribute_sets,
        X_train,
        y_train,
        X_val,
        y_val,
        model_type="regression",
        num_inner_bags=0,
        num_classification_states=2,
        training_scores=None,
        validation_scores=None,
        random_state=1337,
    ):

        # TODO: Update documentation for training/val scores args.
        """ Initializes internal wrapper for EBM C code.

        Args:
            attributes: List of attributes represented individually as
                dictionary of keys ('type', 'has_missing', 'n_bins').
            attribute_sets: List of attribute sets represented as
                a dictionary of keys ('n_attributes', 'attributes')
            X_train: Training design matrix as 2-D ndarray.
            y_train: Training response as 1-D ndarray.
            X_val: Validation design matrix as 2-D ndarray.
            y_val: Validation response as 1-D ndarray.
            model_type: 'regression'/'classification'.
            num_inner_bags: Per feature training step, number of inner bags.
            num_classification_states: Specific to classification,
                number of unique classes.
            training_scores: Undocumented.
            validation_scores: Undocumented.
            random_state: Random seed as integer.
        """
        log.debug("Check if EBM lib is loaded")
        if this.native is None:
            log.info("EBM lib loading.")
            this.native = Native()
        else:
            log.debug("EBM lib already loaded")

        log.info("Allocation start")

        # Store args
        self.attributes = attributes
        self.attribute_sets = attribute_sets
        self.attribute_array, self.attribute_sets_array, self.attribute_set_indexes = self._convert_attribute_info_to_c(
            attributes, attribute_sets
        )

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.model_type = model_type
        self.num_inner_bags = num_inner_bags
        self.num_classification_states = num_classification_states

        # Set train/val scores to zeros if not passed.
        self.training_scores = (
            training_scores
            if training_scores is not None
            else np.zeros(X_train.shape[0])
        )
        self.validation_scores = (
            validation_scores
            if validation_scores is not None
            else np.zeros(X_val.shape[0])
        )
        self.random_state = random_state

        # Convert n-dim arrays ready for C.
        self.X_train_f = np.asfortranarray(self.X_train)
        self.X_val_f = np.asfortranarray(self.X_val)

        # Define extra properties
        self.model_pointer = None
        self.interaction_pointer = None

        # Allocate external resources
        if self.model_type == "regression":
            self.y_train = self.y_train.astype("float64")
            self.y_val = self.y_val.astype("float64")
            self._initialize_training_regression()
            self._initialize_interaction_regression()
        elif self.model_type == "classification":
            self.y_train = self.y_train.astype("int64")
            self.y_val = self.y_val.astype("int64")

            self._initialize_training_classification()
            self._initialize_interaction_classification()

        log.info("Allocation end")

    def _convert_attribute_info_to_c(self, attributes, attribute_sets):
        # Create C form of attributes
        attribute_ar = (this.native.Attribute * len(attributes))()
        for idx, attribute in enumerate(attributes):
            if attribute["type"] == "categorical":
                attribute_ar[idx].attributeType = this.native.AttributeTypeNominal
            else:
                attribute_ar[idx].attributeType = this.native.AttributeTypeOrdinal
            attribute_ar[idx].hasMissing = 1 * attribute["has_missing"]
            attribute_ar[idx].countStates = attribute["n_bins"]

        attribute_set_indexes = []
        attribute_sets_ar = (this.native.AttributeSet * len(attribute_sets))()
        for idx, attribute_set in enumerate(attribute_sets):
            attribute_sets_ar[idx].countAttributes = attribute_set["n_attributes"]

            for attr_idx in attribute_set["attributes"]:
                attribute_set_indexes.append(attr_idx)

        attribute_set_indexes = np.array(attribute_set_indexes, dtype="int64")

        return attribute_ar, attribute_sets_ar, attribute_set_indexes

    def _initialize_interaction_regression(self):
        self.interaction_pointer = this.native.lib.InitializeInteractionRegression(
            len(self.attribute_array),
            self.attribute_array,
            self.X_train.shape[0],
            self.y_train,
            self.X_train_f,
            self.training_scores,
        )

    def _initialize_interaction_classification(self):
        self.interaction_pointer = this.native.lib.InitializeInteractionClassification(
            len(self.attribute_array),
            self.attribute_array,
            self.num_classification_states,
            self.X_train.shape[0],
            self.y_train,
            self.X_train_f,
            self.training_scores,
        )

    def _initialize_training_regression(self):
        self.model_pointer = this.native.lib.InitializeTrainingRegression(
            self.random_state,
            len(self.attribute_array),
            self.attribute_array,
            len(self.attribute_sets_array),
            self.attribute_sets_array,
            self.attribute_set_indexes,
            self.X_train.shape[0],
            self.y_train,
            self.X_train_f,
            self.training_scores,
            self.X_val.shape[0],
            self.y_val,
            self.X_val_f,
            self.validation_scores,
            self.num_inner_bags,
        )

    def _initialize_training_classification(self):
        self.model_pointer = this.native.lib.InitializeTrainingClassification(
            self.random_state,
            len(self.attribute_array),
            self.attribute_array,
            len(self.attribute_sets_array),
            self.attribute_sets_array,
            self.attribute_set_indexes,
            self.num_classification_states,
            self.X_train.shape[0],
            self.y_train,
            self.X_train_f,
            self.training_scores,
            self.X_val.shape[0],
            self.y_val,
            self.X_val_f,
            self.validation_scores,
            self.num_inner_bags,
        )

    def close(self):
        """ Deallocates C objects used to train EBM. """
        log.info("Deallocation start")
        this.native.lib.FreeTraining(self.model_pointer)
        this.native.lib.FreeInteraction(self.interaction_pointer)
        log.info("Deallocation end")

    def fast_interaction_score(self, attribute_index_tuple):
        """ Provides score for an attribute interaction. Higher is better."""
        log.info("Fast interaction score start")
        score = ct.c_double(0.0)
        this.native.lib.GetInteractionScore(
            self.interaction_pointer,
            len(attribute_index_tuple),
            np.array(attribute_index_tuple, dtype=np.int64),
            ct.byref(score),
        )
        log.info("Fast interaction score end")
        return score.value

    def training_step(
        self,
        attribute_set_index,
        training_step_episodes=1,
        learning_rate=0.01,
        max_tree_splits=2,
        min_cases_for_split=2,
        training_weights=0,
        validation_weights=0,
    ):

        """ Conducts a training step per feature
            by growing a shallow decision tree.

        Args:
            attribute_set_index: The index for the attribute set
                to train on.
            training_step_episodes: Number of episodes to train feature step.
            learning_rate: Learning rate as a float.
            max_tree_splits: Max tree splits on feature step.
            min_cases_for_split: Min observations required to split.
            training_weights: Training weights as float vector.
            validation_weights: Validation weights as float vector.

        Returns:
            Validation loss for the training step.
        """
        # log.debug("Training step start")

        metric_output = ct.c_double(0.0)
        for i in range(training_step_episodes):
            return_code = this.native.lib.TrainingStep(
                self.model_pointer,
                attribute_set_index,
                learning_rate,
                max_tree_splits,
                min_cases_for_split,
                training_weights,
                validation_weights,
                ct.byref(metric_output),
            )
            if return_code != 0:
                raise Exception("TrainingStep Exception")

        # log.debug("Training step end")
        return metric_output.value

    def _get_attribute_set_shape(self, attribute_set_index):
        # Retrieve dimensions of log odds tensor
        dimensions = []
        attr_idxs = []
        attribute_set = self.attribute_sets[attribute_set_index]
        for _, attr_idx in enumerate(attribute_set["attributes"]):
            n_bins = self.attributes[attr_idx]["n_bins"]
            attr_idxs.append(attr_idx)
            dimensions.append(n_bins)
        shape = tuple(dimensions)
        return shape

    def get_best_model(self, attribute_set_index):
        """ Returns best model/function according to validation set
            for a given attribute set.

        Args:
            attribute_set_index: The index for the attribute set.

        Returns:
            An ndarray that represents the model.
        """
        array_p = this.native.lib.GetBestModel(self.model_pointer, attribute_set_index)
        shape = self._get_attribute_set_shape(attribute_set_index)

        array = make_nd_array(
            array_p, shape, dtype=np.double, order="F", own_data=False
        )
        return array.copy()

    def get_current_model(self, attribute_set_index):
        """ Returns current model/function according to validation set
            for a given attribute set.

        Args:
            attribute_set_index: The index for the attribute set.

        Returns:
            An ndarray that represents the model.
        """
        array_p = this.native.lib.GetCurrentModel(
            self.model_pointer, attribute_set_index
        )
        shape = self._get_attribute_set_shape(attribute_set_index)

        array = make_nd_array(
            array_p, shape, dtype=np.double, order="F", own_data=False
        )
        return array.copy()


def make_nd_array(c_pointer, shape, dtype=np.float64, order="C", own_data=True):
    """ Returns an ndarray based from a C array.

    Code largely borrowed from:
    https://stackoverflow.com/questions/4355524/getting-data-from-ctypes-array-into-numpy

    Args:
        c_pointer: Pointer to C array.
        shape: Shape of ndarray to form.
        dtype: Numpy data type.
        order: C/Fortran contiguous.
        own_data: Whether data is copied into Python space.

    Returns:
        An ndarray.
    """

    arr_size = np.prod(shape[:]) * np.dtype(dtype).itemsize
    if sys.version_info.major >= 3:
        buf_from_mem = ct.pythonapi.PyMemoryView_FromMemory
        buf_from_mem.restype = ct.py_object
        buf_from_mem.argtypes = (ct.c_void_p, ct.c_int, ct.c_int)
        buffer = buf_from_mem(c_pointer, arr_size, 0x100)
    else:
        buf_from_mem = ct.pythonapi.PyBuffer_FromMemory
        buf_from_mem.restype = ct.py_object
        buffer = buf_from_mem(c_pointer, arr_size)
    arr = np.ndarray(tuple(shape[:]), dtype, buffer, order=order)
    if own_data and not arr.flags.owndata:
        return arr.copy()
    else:
        return arr
