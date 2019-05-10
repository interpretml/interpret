# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

# TODO: Add unit tests for internal EBM interfacing
import sys
import ctypes as ct
from numpy.ctypeslib import ndpointer
import numpy as np
import os
from sys import platform
import struct
import logging

log = logging.getLogger(__name__)


# Native library path finding
def get_ebm_lib_path(debug=False):
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
        msg = "Platform {0} at {1} bit not supported for EBM".format(platform, bitsize)
        log.error(msg)
        raise Exception(msg)


# Load correct library
def load_library(debug=None):
    gettrace = getattr(sys, "gettrace", None)

    if debug is None:
        is_debug = False
        if gettrace():
            is_debug = True
    else:
        is_debug = debug

    lib = ct.cdll.LoadLibrary(get_ebm_lib_path(debug=is_debug))
    return lib


Lib = load_library(debug=False)

# C-level interface

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


InitializeTrainingRegression = Lib.InitializeTrainingRegression
InitializeTrainingRegression.argtypes = [
    # int64_t randomSeed
    ct.c_longlong,
    # int64_t countAttributes
    ct.c_longlong,
    # Attribute * attributes
    ct.POINTER(Attribute),
    # int64_t countAttributeSets
    ct.c_longlong,
    # AttributeSet * attributeSets
    ct.POINTER(AttributeSet),
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
InitializeTrainingRegression.restype = ct.c_void_p


InitializeTrainingClassification = Lib.InitializeTrainingClassification
InitializeTrainingClassification.argtypes = [
    # int64_t randomSeed
    ct.c_longlong,
    # int64_t countAttributes
    ct.c_longlong,
    # Attribute * attributes
    ct.POINTER(Attribute),
    # int64_t countAttributeSets
    ct.c_longlong,
    # AttributeSet2 * attributeSets
    ct.POINTER(AttributeSet),
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
InitializeTrainingClassification.restype = ct.c_void_p

TrainingStep = Lib.TrainingStep
TrainingStep.argtypes = [
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
TrainingStep.restype = ct.c_longlong


GetCurrentModel = Lib.GetCurrentModel
GetCurrentModel.argtypes = [
    # void * tml
    ct.c_void_p,
    # int64_t indexAttributeSet
    ct.c_longlong,
]
GetCurrentModel.restype = ct.POINTER(ct.c_double)


GetBestModel = Lib.GetBestModel
GetBestModel.argtypes = [
    # void * tml
    ct.c_void_p,
    # int64_t indexAttributeSet
    ct.c_longlong,
]
GetBestModel.restype = ct.POINTER(ct.c_double)


FreeTraining = Lib.FreeTraining
FreeTraining.argtypes = [
    # void * tml
    ct.c_void_p
]


CancelTraining = Lib.CancelTraining
CancelTraining.argtypes = [
    # void * tml
    ct.c_void_p
]

InitializeInteractionClassification = Lib.InitializeInteractionClassification
InitializeInteractionClassification.argtypes = [
    # int64_t countAttributes
    ct.c_longlong,
    # Attribute * attributes
    ct.POINTER(Attribute),
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
InitializeInteractionClassification.restype = ct.c_void_p


InitializeInteractionRegression = Lib.InitializeInteractionRegression
InitializeInteractionRegression.argtypes = [
    # int64_t countAttributes
    ct.c_longlong,
    # Attribute * attributes
    ct.POINTER(Attribute),
    # int64_t countCases
    ct.c_longlong,
    # double * targets
    ndpointer(dtype=ct.c_double, flags="F_CONTIGUOUS", ndim=1),
    # int64_t * data
    ndpointer(dtype=ct.c_longlong, flags="F_CONTIGUOUS", ndim=2),
    # double * predictionScores
    ndpointer(dtype=ct.c_double, flags="F_CONTIGUOUS", ndim=1),
]
InitializeInteractionRegression.restype = ct.c_void_p

GetInteractionScore = Lib.GetInteractionScore
GetInteractionScore.argtypes = [
    # void * tmlInteraction
    ct.c_void_p,
    # int64_t countAttributesInCombination
    ct.c_longlong,
    # int64_t * attributeIndexes
    ndpointer(dtype=ct.c_longlong, flags="F_CONTIGUOUS", ndim=1),
    # double * interactionScoreReturn
    ct.POINTER(ct.c_double),
]
GetInteractionScore.restype = ct.c_longlong

FreeInteraction = Lib.FreeInteraction
FreeInteraction.argtypes = [
    # void * tmlInteraction
    ct.c_void_p
]


CancelInteraction = Lib.CancelInteraction
CancelInteraction.argtypes = [
    # void * tmlInteraction
    ct.c_void_p
]


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
        log.debug("Allocation start")

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

        log.debug("Allocation end")

    def _convert_attribute_info_to_c(self, attributes, attribute_sets):
        # Create C form of attributes
        attribute_ar = (Attribute * len(attributes))()
        for idx, attribute in enumerate(attributes):
            if attribute["type"] == "categorical":
                attribute_ar[idx].attributeType = AttributeTypeNominal
            else:
                attribute_ar[idx].attributeType = AttributeTypeOrdinal
            attribute_ar[idx].hasMissing = 1 * attribute["has_missing"]
            attribute_ar[idx].countStates = attribute["n_bins"]

        attribute_set_indexes = []
        attribute_sets_ar = (AttributeSet * len(attribute_sets))()
        for idx, attribute_set in enumerate(attribute_sets):
            attribute_sets_ar[idx].countAttributes = attribute_set["n_attributes"]

            for attr_idx in attribute_set["attributes"]:
                attribute_set_indexes.append(attr_idx)

        attribute_set_indexes = np.array(attribute_set_indexes, dtype="int64")

        return attribute_ar, attribute_sets_ar, attribute_set_indexes

    def _initialize_interaction_regression(self):
        self.interaction_pointer = InitializeInteractionRegression(
            len(self.attribute_array),
            self.attribute_array,
            self.X_train.shape[0],
            self.y_train,
            self.X_train_f,
            self.training_scores,
        )

    def _initialize_interaction_classification(self):
        self.interaction_pointer = InitializeInteractionClassification(
            len(self.attribute_array),
            self.attribute_array,
            self.num_classification_states,
            self.X_train.shape[0],
            self.y_train,
            self.X_train_f,
            self.training_scores,
        )

    def _initialize_training_regression(self):
        self.model_pointer = InitializeTrainingRegression(
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
        self.model_pointer = InitializeTrainingClassification(
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
        log.debug("Deallocation start")
        FreeTraining(self.model_pointer)
        FreeInteraction(self.interaction_pointer)
        log.debug("Deallocation end")

    def fast_interaction_score(self, attribute_index_tuple):
        """ Provides score for an attribute interaction. Higher is better."""
        log.debug("Fast interaction score start")
        score = ct.c_double(0.0)
        GetInteractionScore(
            self.interaction_pointer,
            len(attribute_index_tuple),
            np.array(attribute_index_tuple, dtype=np.int64),
            ct.byref(score),
        )
        log.debug("Fast interaction score end")
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
        log.debug("Training step start")

        metric_output = ct.c_double(0.0)
        for i in range(training_step_episodes):
            TrainingStep(
                self.model_pointer,
                attribute_set_index,
                learning_rate,
                max_tree_splits,
                min_cases_for_split,
                training_weights,
                validation_weights,
                ct.byref(metric_output),
            )

        log.debug("Training step end")
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
        array_p = GetBestModel(self.model_pointer, attribute_set_index)
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
        array_p = GetCurrentModel(self.model_pointer, attribute_set_index)
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
