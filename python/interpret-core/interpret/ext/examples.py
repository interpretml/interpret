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


class ExampleDataExplainer(ExplainerMixin):
    available_explanations = ["data"]
    explainer_type = "data"

    def __init__(self, feature_names=None, feature_types=None):
        pass

    def explain_data(self, X, y, name=None):
        return ExampleExplanation()


class ExamplePerfExplainer(ExplainerMixin):
    available_explanations = ["perf"]
    explainer_type = "perf"

    def __init__(self, predict_fn, feature_names=None, feature_types=None, **kwargs):
        pass

    def explain_perf(self, X, y, name=None):
        return ExampleExplanation()


class ExampleGlassboxExplainer(ExplainerMixin):
    available_explanations = ["local"]
    explainer_type = "model"

    def __init__(self, feature_names=None, feature_types=None):
        pass

    def fit(self, X, y):
        return self

    def predict(self, X):
        return None

    def explain_local(self, X, y=None, name=None):
        return ExampleExplanation()


class ExampleGreyboxExplainer(ExplainerMixin):
    available_explanations = ["local"]
    explainer_type = "specific"

    def __init__(self, model, data, feature_names=None, feature_types=None):
        pass

    def explain_local(self, X, y=None, name=None):
        return ExampleExplanation()


class ExampleBlackboxExplainer(ExplainerMixin):
    available_explanations = ["local"]
    explainer_type = "blackbox"

    def __init__(
        self, predict_fn, data, sampler=None, feature_names=None, feature_types=None
    ):
        pass

    def explain_local(self, X, y=None, name=None):
        return ExampleExplanation()


class ExampleVisualizeProvider:
    def render(self, explanation, key=-1, **kwargs):
        return None
