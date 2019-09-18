# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines the Explanation dashboard class."""

from .ExplanationWidget import ExplanationWidget
from ._internal.constants import ExplanationDashboardInterface, WidgetRequestResponseConstants
from IPython.display import display
from scipy.sparse import issparse
import numpy as np
import pandas as pd


class ExplanationDashboard(object):
    """The dashboard class, wraps the dashboard component."""

    def __init__(self, explanationObject, learner=None, dataset=None, trueY=None):
        """Initialize the Explanation Dashboard.

        :param explanationObject: An object that represents an explanation.
        :type explanationObject: ExplanationMixin
        :param learner: An object that represents a model. It is assumed that for the classification case
            it has a method of predict_proba() returning the prediction probabilities for each
            class and for the regression case a method of predict() returning the prediction value.
        :type learner: object
        :param dataset:  A matrix of feature vector examples (# examples x # features), the same sampels
            used to build the explanationObject. Will be overwritten if set on explanation object already
        :type dataset: numpy.array or list[][]
        :param trueY: The true labels for the provided dataset. Will be overwritten if set on explanation object already
        :tpye trueY: numpy.array or list[]
        """
        self._widget_instance = ExplanationWidget()
        self._learner = learner
        self._is_classifier = hasattr(learner, 'predict_proba') and learner.predict_proba is not None
        self._dataframeColumns = None
        if isinstance(dataset, pd.DataFrame) and hasattr(dataset, 'columns'):
            self._dataframeColumns = dataset.columns
        try:
            list_dataset = self._convertToList(dataset)
        except:
            raise ValueError("Unsupported dataset type")
        try:
            y_pred = learner.predict(dataset)
        except:
            raise ValueError("Model does not support predict method for given dataset type")
        try:
            y_pred = self._convertToList(y_pred)
        except:
            raise ValueError("Model prediction output of unsupported type")
        try:
            trueY = self._convertToList(trueY)
        except:
            raise ValueError("True Y array of unsupported type")

        dataArg = {}

        row_length, feature_length = np.shape(list_dataset)
        if row_length > 100000:
            raise ValueError("Exceeds maximum number of rows for visualization (100000)")
        if feature_length > 1000:
            raise ValueError("Exceeds maximum number of features for visualization (1000)")
        # List of explanations, key of explanation type is "explanation_type"
        self._mli_explanations = explanationObject.data(-1)["mli"]
        local_explanation = self._find_first_explanation(ExplanationDashboardInterface.MLI_LOCAL_EXPLANATION_KEY)
        global_explanation = self._find_first_explanation(ExplanationDashboardInterface.MLI_GLOBAL_EXPLANATION_KEY)
        ebm_explanation = self._find_first_explanation(ExplanationDashboardInterface.MLI_EBM_GLOBAL_EXPLANATION_KEY)
        dataset_explanation = self._find_first_explanation(ExplanationDashboardInterface.EXPLANATION_DATASET_KEY)

        dataset_x = None
        dataset_y = None
        predicted_y = None
        if dataset_explanation is not None:
            dataset_x = dataset_explanation[ExplanationDashboardInterface.MLI_DATASET_X_KEY]
            dataset_y = dataset_explanation[ExplanationDashboardInterface.MLI_DATASET_Y_KEY]
        if dataset_x is None and dataset is not None:
            dataset_x = dataset
        if dataset_y is None and trueY is not None:
            dataset_y = trueY
        if isinstance(dataset_x, pd.DataFrame) and hasattr(dataset_x, 'columns'):
            self._dataframeColumns = dataset_x.columns
        try:
            list_dataset = self._convertToList(dataset_x)
        except:
            raise ValueError("Unsupported dataset type")
        if predicted_y is None and dataset_x is not None and learner is not None:
            try:
                predicted_y = learner.predict(dataset_x)
            except:
                raise ValueError("Model does not support predict method for given dataset type")
            try:
                predicted_y = self._convertToList(predicted_y)
            except:
                raise ValueError("Model prediction output of unsupported type")
        if predicted_y is not None:
            dataArg[ExplanationDashboardInterface.PREDICTED_Y] = y_pred
        if dataset_x is not None:
            try:
                list_dataset = self._convertToList(dataset_x)
            except:
                raise ValueError("Unsupported dataset type")
            ExplanationDashboardInterface.TRAINING_DATA: list_dataset
            ExplanationDashboardInterface.IS_CLASSIFIER: self._is_classifier

        local_dim = None

        if trueY is not None and len(trueY) == row_length:
            dataArg[ExplanationDashboardInterface.TRUE_Y] = trueY

        if local_explanation is not None:
            try:
                local_explanation["scores"] = self._convertToList(local_explanation["scores"])
                dataArg[ExplanationDashboardInterface.LOCAL_EXPLANATIONS] = local_explanation
            except:
                raise ValueError("Unsupported local explanation type")
            local_dim = np.shape(local_explanation["scores"])
            if len(local_dim) != 2 and len(local_dim) != 3:
                raise ValueError("Local explanation expected to be a 2D or 3D list")
            if len(local_dim) == 2 and (local_dim[1] != feature_length or local_dim[0] != row_length):
                raise ValueError("Shape mismatch: local explanation length differs from dataset")
            if len(local_dim) == 3 and (local_dim[2] != feature_length or local_dim[1] != row_length):
                raise ValueError("Shape mismatch: local explanation length differs from dataset")
        if local_explanation is None and global_explanation is not None:
            try:
                global_explanation["scores"] = self._convertToList(global_explanation["scores"])
                dataArg[ExplanationDashboardInterface.GLOBAL_EXPLANATION] = global_explanation
            except:
                raise ValueError("Unsupported global explanation type")
        if ebm_explanation is not None:
            try:
                dataArg[ExplanationDashboardInterface.EBM_EXPLANATION] = ebm_explanation
            except:
                raise ValueError("Unsupported ebm explanation type")
        if hasattr(explanationObject, 'features') and explanationObject.features is not None:
            features = self._convertToList(explanationObject.features)
            if len(features) != feature_length:
                raise ValueError("Feature vector length mismatch: \
                    feature names length differs from local explanations dimension")
            dataArg[ExplanationDashboardInterface.FEATURE_NAMES] = features
        if hasattr(explanationObject, 'classes') and explanationObject.classes is not None:
            classes = self._convertToList(explanationObject.classes)
            if local_dim is not None and len(classes) != local_dim[0]:
                raise ValueError("Class vector length mismatch: \
                    class names length differs from local explanations dimension")
            dataArg[ExplanationDashboardInterface.CLASS_NAMES] = self._convertToList(explanationObject.classes)
        if hasattr(learner, 'predict_proba') and learner.predict_proba is not None:
            try:
                probability_y = learner.predict_proba(dataset)
            except:
                raise ValueError("Model does not support predict_proba method for given dataset type")
            try:
                probability_y = self._convertToList(probability_y)
            except:
                raise ValueError("Model predict_proba output of unsupported type")
            dataArg[ExplanationDashboardInterface.PROBABILITY_Y] = probability_y
        self._widget_instance.value = dataArg
        self._widget_instance.observe(self._on_request, names=WidgetRequestResponseConstants.REQUEST)
        display(self._widget_instance)

    def _on_request(self, change):
        try:
            data = change.new[WidgetRequestResponseConstants.DATA]
            if self._dataframeColumns is not None:
                data = pd.DataFrame(data, columns=self._dataframeColumns)
            if (self._is_classifier):
                prediction = self._convertToList(self._learner.predict_proba(data))
            else:
                prediction = self._convertToList(self._learner.predict(data))
            self._widget_instance.response = {
                WidgetRequestResponseConstants.DATA: prediction,
                WidgetRequestResponseConstants.ID: change.new[WidgetRequestResponseConstants.ID]}
        except:
            self._widget_instance.response = {
                WidgetRequestResponseConstants.ERROR: "Model threw exeption while predicting",
                WidgetRequestResponseConstants.DATA: [],
                WidgetRequestResponseConstants.ID: change.new[WidgetRequestResponseConstants.ID]}

    def _show(self):
        display(self._widget_instance)

    def _convertToList(self, array):
        if issparse(array):
            if array.shape[1] > 1000:
                raise ValueError("Exceeds maximum number of features for visualization (1000)")
            return array.toarray().tolist()
        if (isinstance(array, pd.DataFrame)):
            return array.values.tolist()
        if (isinstance(array, np.ndarray)):
            return array.tolist()
        return array

    def _find_first_explanation(self, key):
        new_array = [explanation for explanation in self._mli_explanations
        if explanation[ExplanationDashboardInterface.MLI_EXPLANATION_TYPE_KEY] == key]
        if len(new_array) > 0:
            return new_array[0]["value"]
        return None
