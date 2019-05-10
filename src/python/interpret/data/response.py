# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from ..api.base import ExplainerMixin, ExplanationMixin
from ..utils import unify_data, gen_name_from_class, gen_global_selector
from ..visual.plot import plot_density, COLORS

import plotly.graph_objs as go
import numpy as np
from scipy.stats import pearsonr


class Marginal(ExplainerMixin):
    available_explanations = ["data"]
    explainer_type = "data"

    def __init__(
        self,
        feature_names=None,
        feature_types=None,
        max_scatter_samples=400,
        random_state=1,
        **kwargs
    ):
        self.max_scatter_samples = max_scatter_samples
        self.random_state = random_state
        self.kwargs = kwargs
        self.feature_names = feature_names
        self.feature_types = feature_types

    def explain_data(self, X, y, name=None):
        if name is None:
            name = gen_name_from_class(self)

        X, y, self.feature_names, self.feature_types = unify_data(
            X, y, self.feature_names, self.feature_types
        )

        global_selector = gen_global_selector(
            X, self.feature_names, self.feature_types, None
        )
        counts, values = np.histogram(y, bins="doane")
        response_density_data_dict = {"names": values, "scores": counts}
        overall_dict = {
            "type": "hist",
            "density": response_density_data_dict,
            "X": X,
            "y": y,
        }

        # Sample down
        n_samples = (
            self.max_scatter_samples if len(y) > self.max_scatter_samples else len(y)
        )
        np.random.seed(self.random_state)
        idx = np.random.choice(np.arange(len(y)), n_samples, replace=False)
        X_sample = X[idx, :]
        y_sample = y[idx]
        specific_dicts = []
        for feat_idx, feature_name in enumerate(self.feature_names):
            feature_type = self.feature_types[feat_idx]
            if feature_type == "continuous":
                counts, values = np.histogram(X[:, feat_idx], bins="doane")
                corr = pearsonr(X[:, feat_idx], y)[0]
            elif feature_type == "categorical":
                values, counts = np.unique(X[:, feat_idx], return_counts=True)
                corr = None
            else:
                raise Exception("Cannot support type: {0}".format(feature_type))

            feat_density_data_dict = {"names": values, "scores": counts}
            specific_dict = {
                "type": "marginal",
                "feature_density": feat_density_data_dict,
                "response_density": response_density_data_dict,
                "feature_samples": X_sample[:, feat_idx],
                "response_samples": y_sample,
                "correlation": corr,
            }

            specific_dicts.append(specific_dict)

        internal_obj = {"overall": overall_dict, "specific": specific_dicts}

        return MarginalExplanation(
            "data",
            internal_obj,
            feature_names=self.feature_names,
            feature_types=self.feature_types,
            name=name,
            selector=global_selector,
        )


class MarginalExplanation(ExplanationMixin):
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
        specific_dict = self._internal_obj["specific"][key].copy()
        specific_dict["x"] = self._internal_obj["overall"]["X"][:, key]
        specific_dict["y"] = self._internal_obj["overall"]["y"]
        return specific_dict

    def visualize(self, key=None):
        data_dict = self.data(key)
        if data_dict is None:
            return None

        if key is None:
            figure = plot_density(data_dict["density"], title="Response")
            return figure

        # Show feature graph
        density_dict = data_dict["feature_density"]

        bin_size = density_dict["names"][1] - density_dict["names"][0]
        is_categorical = self.feature_types[key] == "categorical"

        if is_categorical:
            trace1 = go.Histogram(x=data_dict["x"], name="x density", yaxis="y2")
        else:
            trace1 = go.Histogram(
                x=data_dict["x"],
                name="x density",
                yaxis="y2",
                autobinx=False,
                xbins=dict(
                    start=density_dict["names"][0],
                    end=density_dict["names"][-1],
                    size=bin_size,
                ),
            )
        data = []
        resp_density_dict = data_dict["response_density"]
        resp_bin_size = density_dict["names"][1] - density_dict["names"][0]

        trace2 = go.Histogram(
            y=data_dict["y"],
            name="y density",
            xaxis="x2",
            autobiny=False,
            ybins=dict(
                start=resp_density_dict["names"][0],
                end=resp_density_dict["names"][-1],
                size=resp_bin_size,
            ),
        )
        data.append(trace1)
        data.append(trace2)
        x = data_dict["feature_samples"]
        y = data_dict["response_samples"]
        if not is_categorical:
            trace3 = go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name="points",
                marker=dict(size=5, opacity=0.5),
            )
            data.append(trace3)
        else:
            trace5 = go.Box(x=x, y=y, name="density")
            data.append(trace5)

        corr = data_dict["correlation"]
        do_lo = 0.75
        do_hi = 0.85
        layout = go.Layout(
            showlegend=False,
            autosize=True,
            xaxis=dict(
                title=self.feature_names[key],
                type="category" if is_categorical else "-",
                domain=[0, do_hi],
                showgrid=False,
                zeroline=False,
            ),
            yaxis=dict(
                title="Response", domain=[0, do_lo], showgrid=False, zeroline=False
            ),
            hovermode="closest",
            xaxis2=dict(domain=[do_hi, 1], showgrid=False, zeroline=False),
            yaxis2=dict(domain=[do_hi, 1], showgrid=False, zeroline=False),
            title="Pearson Correlation: {0:.3f}".format(corr)
            if corr is not None
            else "",
        )
        fig = go.Figure(data=data, layout=layout)
        return fig


class ClassHistogram(ExplainerMixin):
    available_explanations = ["data"]
    explainer_type = "data"

    def __init__(self, feature_names=None, feature_types=None, **kwargs):
        self.kwargs = kwargs
        self.feature_names = feature_names
        self.feature_types = feature_types

    def explain_data(self, X, y, name=None):
        if name is None:
            name = gen_name_from_class(self)

        X, y, self.feature_names, self.feature_types = unify_data(
            X, y, self.feature_names, self.feature_types
        )

        global_selector = gen_global_selector(
            X, self.feature_names, self.feature_types, None
        )

        overall_dict = {"type": "hist", "X": X, "y": y}
        internal_obj = {
            "overall": overall_dict,
            "specific": None,  # NOTE: Will be generated at data call
        }

        return ClassHistogramExplanation(
            "data",
            internal_obj,
            feature_names=self.feature_names,
            feature_types=self.feature_types,
            name=name,
            selector=global_selector,
        )


class ClassHistogramExplanation(ExplanationMixin):
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

        specific_dict = {}
        specific_dict["x"] = self._internal_obj["overall"]["X"][:, key]
        specific_dict["y"] = self._internal_obj["overall"]["y"]
        return specific_dict

    def visualize(self, key=None):
        data_dict = self.data(key)
        if data_dict is None:
            return None

        # Show overall graph
        if key is None:
            y = data_dict["y"]
            values, counts = np.unique(y, return_counts=True)
            trace1 = go.Bar(x=[values[0]], y=[counts[0]])
            trace2 = go.Bar(x=[values[1]], y=[counts[1]])
            layout = go.Layout(
                title="Response Distribution",
                showlegend=False,
                xaxis=dict(type="category"),
            )
            fig = go.Figure(data=[trace1, trace2], layout=layout)
            return fig

        # Show feature graph
        x = data_dict["x"]
        y = data_dict["y"]

        column_name = self.feature_names[key]

        classes = list(sorted(set(y)))
        data = []
        is_categorical = self.feature_types[key] == "categorical"
        if not is_categorical:
            _, bins = np.histogram(x, bins="doane")
        for idx, current_class in enumerate(classes):
            x_filt = x[y == current_class]
            if is_categorical:
                values, counts = np.unique(x_filt, return_counts=True)
            else:
                counts, values = np.histogram(x_filt, bins=bins)

            data_dict = {"data_type": "univariate", "names": values, "scores": counts}
            fig = plot_density(
                data_dict,
                color=COLORS[idx],
                name=str(current_class),
                is_categorical=is_categorical,
            )
            data.append(fig["data"][0])

        layout = go.Layout(
            title=column_name, barmode="stack", hoverlabel=dict(font=dict(size=25))
        )
        figure = go.Figure(data, layout)
        figure["layout"]["yaxis1"].update(title="Density")

        return figure
