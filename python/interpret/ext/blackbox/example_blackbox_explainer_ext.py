# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from interpret.api.base import ExplainerMixin


class ExampleExplanation(object):
    available_explanations = ["not_important"]
    explainer_type = "demo"

    def data(self, key=None):
        return {"How": {"do": {"you": "do?"}}}

    def visualize(self, key=None):
        return {"here": "I am"}


class ExampleExplainer(ExplainerMixin):
    def explain_local(self, X, y=None, name=None):
        print("Hello world")
        return ExampleExplanation()
