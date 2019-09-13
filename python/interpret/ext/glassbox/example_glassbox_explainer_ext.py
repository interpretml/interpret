# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from interpret.api.base import ExplainerMixin, ExplanationMixin


class ExampleExplanation(ExplanationMixin):
    explanation_type = "local"

    def __init__(self):
        self._internal_obj = None

    def data(self, key=None):
        return None

    def visualize(self, key=None):
        return None


class ExampleExplainer(ExplainerMixin):
    available_explanations = ["local"]
    explainer_type = "glassbox"

    def __init__(
        self, feature_names=None, feature_types=None
    ):
        self.feature_names = feature_names
        self.feature_types = feature_types

    def fit(self, X, y):
        return self

    def predict(self, X):
        return None

    def predict_proba(self, X):
        return None

    def explain_local(self, X, y=None, name=None):
        return ExampleExplanation()
