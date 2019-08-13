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
    explainer_type = "blackbox"

    def __init__(
            self,
            predict_fn,
            data,
            sampler=None,
            feature_names=None,
            feature_types=None,
    ):
        self.predict_fn = predict_fn
        self.data = data

    def explain_local(self, X, y=None, name=None):
        return ExampleExplanation()
