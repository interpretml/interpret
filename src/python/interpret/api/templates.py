# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from .base import ExplanationMixin
from ..visual.plot import plot_line, plot_bar, plot_horizontal_bar
from ..visual.plot import sort_take, plot_pairwise_heatmap


class FeatureValueExplanation(ExplanationMixin):
    """ Visualizes explanation given it matches following criteria.
    """

    explanation_type = None

    def __init__(
        self,
        explanation_type,
        internal_obj,
        feature_names=None,
        feature_types=None,
        name=None,
        selector=None,
    ):
        self.explanation_type = explanation_type
        self._internal_obj = internal_obj
        self.feature_names = feature_names
        self.feature_types = feature_types
        self.name = name
        self.selector = selector

    def data(self, key=None):
        if key is None:
            return self._internal_obj["overall"]

        if self._internal_obj["specific"] is None:
            return None
        return self._internal_obj["specific"][key]

    def visualize(self, key=None):
        data_dict = self.data(key)
        if data_dict is None:
            return None

        # Handle overall graphs
        if key is None:
            data_dict = sort_take(
                data_dict, sort_fn=lambda x: -abs(x), top_n=15, reverse_results=True
            )
            return plot_horizontal_bar(data_dict)

        # Handle local instance graphs
        if self.explanation_type == "local":
            data_dict = sort_take(
                data_dict, sort_fn=lambda x: -abs(x), top_n=15, reverse_results=True
            )
            return plot_horizontal_bar(data_dict)

        # Handle global feature graphs
        feature_type = self.feature_types[key]
        title = self.feature_names[key]
        if feature_type == "continuous":
            return plot_line(data_dict, title=title)
        elif feature_type == "categorical":
            return plot_bar(data_dict, title=title)
        elif feature_type == "pairwise":
            xtitle = title.split(" x ")[0]
            ytitle = title.split(" x ")[1]
            return plot_pairwise_heatmap(
                data_dict, title=title, xtitle=xtitle, ytitle=ytitle
            )

        # Handle everything else as invalid
        raise Exception(
            "Not supported configuration: {0}, {1}".format(
                self.explanation_type, feature_type
            )
        )
