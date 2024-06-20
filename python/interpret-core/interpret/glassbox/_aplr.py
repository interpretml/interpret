# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license
import numpy as np
from typing import List
import aplr
from ..api.base import ExplainerMixin
from ..api.templates import FeatureValueExplanation
from ..utils._explanation import (
    gen_name_from_class,
    gen_global_selector,
    gen_local_selector,
    gen_perf_dicts,
)

FloatVector = List[float]
FloatMatrix = List[List[float]]
IntVector = List[int]
IntMatrix = List[List[int]]
StrVector = List[str]


class APLRRegressor(aplr.APLRRegressor, ExplainerMixin):
    available_explanations = ["local", "global"]
    explainer_type = "model"

    def fit(
        self,
        X: FloatMatrix,
        y: FloatVector,
        sample_weight: FloatVector = [],
        X_names: StrVector = [],
        cv_observations: IntMatrix = [],
        prioritized_predictors_indexes: IntVector = [],
        monotonic_constraints: IntVector = [],
        group: FloatVector = [],
        interaction_constraints: List[List[int]] = [],
        other_data: FloatMatrix = [],
        predictor_learning_rates: FloatVector = [],
        predictor_penalties_for_non_linearity: FloatVector = [],
        predictor_penalties_for_interactions: FloatVector = [],
    ):
        self.bin_counts:List[int]=[]
        self.bin_edges:List[float]=[]
        for col in X.T:
            counts,bin_edges=np.histogram(col,bins="doane")
            self.bin_counts.append(counts)
            self.bin_edges.append(bin_edges)

        return super().fit(X, y, sample_weight, X_names, cv_observations, prioritized_predictors_indexes, monotonic_constraints, group, interaction_constraints, other_data, predictor_learning_rates, predictor_penalties_for_non_linearity, predictor_penalties_for_interactions)

    def explain_global(self, name=None):
        """Provides global explanation for model.

        Args:
            name: User-defined explanation name.

        Returns:
            An explanation object,
            visualizing feature-value pairs as horizontal bar chart.
        """
        overall_dict = {"names": self.get_unique_term_affiliations(),"scores": self.get_feature_importance()}

        data_dicts = []


    def explain_local(self, X, y=None, name=None):
        """Provides local explanations for provided instances.

        Args:
            X: Numpy array for X to explain.
            y: Numpy vector for y to explain.
            name: User-defined explanation name.

        Returns:
            An explanation object, visualizing feature-value pairs
            for each instance as horizontal bar charts.
        """
        ...


class APLRClassifier(aplr.APLRClassifier, ExplainerMixin):
    available_explanations = ["local", "global"]
    explainer_type = "model"

    def explain_global(self, name=None):
        """Provides global explanation for model.

        Args:
            name: User-defined explanation name.

        Returns:
            An explanation object,
            visualizing feature-value pairs as horizontal bar chart.
        """
        ...

    def explain_local(self, X, y=None, name=None):
        """Provides local explanations for provided instances.

        Args:
            X: Numpy array for X to explain.
            y: Numpy vector for y to explain.
            name: User-defined explanation name.

        Returns:
            An explanation object, visualizing feature-value pairs
            for each instance as horizontal bar charts.
        """
        ...


class APLRExplanation(FeatureValueExplanation):
    """Visualizes specifically for APLR."""

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
        """Initializes class.

        Args:
            explanation_type:  Type of explanation.
            internal_obj: A jsonable object that backs the explanation.
            feature_names: List of feature names.
            feature_types: List of feature types.
            name: User-defined name of explanation.
            selector: A dataframe whose indices correspond to explanation entries.
        """
        super(APLRExplanation, self).__init__(
            explanation_type,
            internal_obj,
            feature_names=feature_names,
            feature_types=feature_types,
            name=name,
            selector=selector,
        )

    def visualize(self, key=None):
        """Provides interactive visualizations.

        Args:
            key: Either a scalar or list
                that indexes the internal object for sub-plotting.
                If an overall visualization is requested, pass None.

        Returns:
            A Plotly figure.
        """
        from visual.plot import (
            is_multiclass_global_data_dict,
            plot_continuous_bar,
            plot_horizontal_bar,
            sort_take,
        )

        data_dict = self.data(key)
        if data_dict is None:
            return None

        # Overall global explanation
        if self.explanation_type == "global" and key is None:
            data_dict = sort_take(
                data_dict, sort_fn=lambda x: -abs(x), top_n=15, reverse_results=True
            )
            title = "Global Term/Feature Importances"

            figure = plot_horizontal_bar(
                data_dict,
                title=title,
                start_zero=True,
                xtitle="Mean Absolute Score (Weighted)",
            )

            figure._interpret_help_text = (
                "The term importances are the mean absolute "
                "contribution (score) each term (feature or interaction) makes to predictions "
                "averaged across the training dataset. Contributions are weighted by the number "
                "of samples in each bin, and by the sample weights (if any). The 15 most "
                "important terms are shown."
            )
            figure._interpret_help_link = "https://github.com/interpretml/interpret/blob/develop/docs/interpret/python/examples/group-importances.ipynb"

            return figure

        # Per term global explanation
        if self.explanation_type == "global":
            title = "Term: {0} ({1})".format(
                self.feature_names[key], self.feature_types[key]
            )

            if self.feature_types[key] == "continuous":
                xtitle = self.feature_names[key]

                if is_multiclass_global_data_dict(data_dict):
                    figure = plot_continuous_bar(
                        data_dict,
                        multiclass=True,
                        show_error=False,
                        title=title,
                        xtitle=xtitle,
                    )
                else:
                    figure = plot_continuous_bar(data_dict, title=title, xtitle=xtitle)

            elif (
                self.feature_types[key] == "nominal"
                or self.feature_types[key] == "ordinal"
                or self.feature_types[key] == "interaction"
            ):
                figure = super().visualize(key, title)
                figure._interpret_help_text = (
                    "The contribution (score) of the term {0} to predictions "
                    "made by the model.".format(self.feature_names[key])
                )
            else:  # pragma: no cover
                raise Exception(
                    "Not supported configuration: {0}, {1}".format(
                        self.explanation_type, self.feature_types[key]
                    )
                )

            figure._interpret_help_text = (
                "The contribution (score) of the term "
                "{0} to predictions made by the model. For classification, "
                "scores are on a log scale (logits). For regression, scores are on the same "
                "scale as the outcome being predicted (e.g., dollars when predicting cost). "
                "Each graph is centered vertically such that average prediction on the train "
                "set is 0.".format(self.feature_names[key])
            )

            return figure

        # Local explanation graph
        if self.explanation_type == "local":
            figure = super().visualize(key)
            figure.update_layout(
                title="Local Explanation (" + figure.layout.title.text + ")",
                xaxis_title="Contribution to Prediction",
            )
            figure._interpret_help_text = (
                "A local explanation shows the breakdown of how much "
                "each term contributed to the prediction for a single sample. The intercept "
                "reflects the average case. In regression, the intercept is the average y-value "
                "of the train set (e.g., $5.51 if predicting cost). In classification, the "
                "intercept is the log of the base rate (e.g., -2.3 if the base rate is 10%). The "
                "15 most important terms are shown."
            )

            return figure