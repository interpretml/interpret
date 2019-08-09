# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from .base import ExplanationMixin


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
        # NOTE: Currently returns full internal object, open to change.
        if key == -1:
            return self._internal_obj

        if key is None:
            return self._internal_obj["overall"]

        if self._internal_obj["specific"] is None:  # pragma: no cover
            return None
        return self._internal_obj["specific"][key]

    def visualize(self, key=None):
        from ..visual.plot import plot_line, plot_bar, plot_horizontal_bar, mli_plot_horizontal_bar
        from ..visual.plot import get_sort_indexes, get_explanation_index, sort_take, mli_sort_take, plot_pairwise_heatmap
        data_dict = self.data(key)
        if data_dict is None:  # pragma: no cover
            return None

        # Handle overall graphs
        if key is None:
            data_dict = sort_take(
                data_dict, sort_fn=lambda x: -abs(x), top_n=15, reverse_results=True
            )
            return plot_horizontal_bar(data_dict)

        # Handle local instance graphs
        if self.explanation_type == "local":
            if "mli" in self.data(-1):
                explanation_list = self.data(-1)["mli"]
                explanation_index = get_explanation_index(explanation_list, "local_feature_importance")
                local_explanation = explanation_list[explanation_index]["value"]
                scores = local_explanation["scores"]
                perf = local_explanation["perf"]
                sort_indexes = get_sort_indexes(scores[key], sort_fn=lambda x: -abs(x), top_n=15)
                sorted_scores = mli_sort_take(scores[key], sort_indexes, reverse_results=True)
                sorted_names = mli_sort_take(self.feature_names, sort_indexes, reverse_results=True)
                instances = explanation_list[1]["value"]["dataset_x"]
                return mli_plot_horizontal_bar(sorted_scores, sorted_names, values=instances[key], perf=perf[key])
            else:
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
            # TODO: Generalize this out.
            xtitle = title.split(" x ")[0]
            ytitle = title.split(" x ")[1]
            return plot_pairwise_heatmap(
                data_dict, title=title, xtitle=xtitle, ytitle=ytitle
            )

        # Handle everything else as invalid
        raise Exception(  # pragma: no cover
            "Not supported configuration: {0}, {1}".format(
                self.explanation_type, feature_type
            )
        )
