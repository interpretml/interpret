# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines the Constant strings."""


class ExplanationDashboardInterface(object):
    """Dictonary properties shared between the python and javascript object."""
    LOCAL_EXPLANATIONS = "localExplanations"
    PREDICTED_Y = "predictedY"
    TRAINING_DATA = "trainingData"
    GLOBAL_EXPLANATION = "globalExplanation"
    IS_CLASSIFIER = "isClassifier"
    FEATURE_NAMES = "featureNames"
    CLASS_NAMES = "classNames"
    PROBABILITY_Y = "probabilityY"
    TRUE_Y = "trueY"
    CUSTOM_VISUALS = "customVis"
    EBM_EXPLANATION = "ebmGlobalExplanation"
    MLI_LOCAL_EXPLANATION_KEY = "local_feature_importance"
    MLI_GLOBAL_EXPLANATION_KEY = "global_feature_importance"
    MLI_EBM_GLOBAL_EXPLANATION_KEY = "ebm_global"
    MLI_EXPLANATION_TYPE_KEY = "explanation_type"
    MLI_EXPLANATION_DATASET_KEY = "evaluation_dataset"
    MLI_DATASET_X_KEY = "dataset_x"
    MLI_DATASET_Y_KEY = "dataset_y"


class WidgetRequestResponseConstants(object):
    """Strings used to pass messages between python and javascript."""
    ID = "id"
    DATA = "data"
    ERROR = "error"
    REQUEST = "request"
