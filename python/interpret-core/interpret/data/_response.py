# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

from ..api.base import ExplainerMixin, ExplanationMixin
from ..utils._explanation import gen_name_from_class, gen_global_selector

import numpy as np
from scipy.stats import pearsonr

from ..utils._clean_x import preclean_X
from ..utils._clean_simple import clean_dimensions, typify_classification

from ..utils._unify_data import unify_data


class Marginal(ExplainerMixin):
    """Provides a marginal plot for provided data."""

    available_explanations = ["data"]
    explainer_type = "data"

    def __init__(
        self,
        feature_names=None,
        feature_types=None,
        max_scatter_samples=400,
    ):
        """Initializes class.

        Args:
            feature_names: List of feature names.
            feature_types: List of feature types.
            max_scatter_samples: Number of sample points in visualization.
        """
        self.feature_names = feature_names
        self.feature_types = feature_types
        self.max_scatter_samples = max_scatter_samples

    def explain_data(self, X, y, name=None):
        """Explains data as visualizations.

        Args:
            X: Numpy array for X to explain.
            y: Numpy vector for y to explain.
            name: User-defined explanation name.

        Returns:
            An explanation object.
        """
        if name is None:
            name = gen_name_from_class(self)

        y = clean_dimensions(y, "y")
        if y.ndim != 1:
            raise ValueError("y must be 1 dimensional")

        try:
            y = y.astype(np.float64, copy=False)
        except (TypeError, ValueError):
            # we get a TypeError whenever we have an np.object_ array and numpy attempts to call float(), but the
            # object doesn't have a __float__ function.  We get a ValueError when either a str object inside an
            # np.object_ array or when an np.unicode_ array attempts to convert a string to a float and fails

            y = typify_classification(y)

        X, n_samples = preclean_X(X, self.feature_names, self.feature_types, len(y))

        X, feature_names, feature_types = unify_data(
            X, n_samples, self.feature_names, self.feature_types, False, 0
        )

        unique_val_counts = np.zeros(len(feature_names), dtype=np.int64)
        for col_idx in range(len(feature_names)):
            X_col = X[:, col_idx]
            unique_val_counts[col_idx] = len(np.unique(X_col))

        global_selector = gen_global_selector(
            len(feature_names),
            feature_names,
            feature_types,
            unique_val_counts,
            None,
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
        idx = np.random.choice(np.arange(len(y)), n_samples, replace=False)
        X_sample = X[idx, :]
        y_sample = y[idx]
        specific_dicts = []
        for feat_idx, feature_name in enumerate(feature_names):
            feature_type = feature_types[feat_idx]
            if feature_type == "continuous":
                counts, values = np.histogram(X[:, feat_idx], bins="doane")
                corr = pearsonr(X[:, feat_idx].astype(np.float64, copy=False), y)[0]
            elif feature_type == "nominal" or feature_type == "ordinal":
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
            feature_names=feature_names,
            feature_types=feature_types,
            name=name,
            selector=global_selector,
        )


class MarginalExplanation(ExplanationMixin):
    """Explanation object specific to marginal explainer."""

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
        self.explanation_type = explanation_type
        self._internal_obj = internal_obj
        self.feature_names = feature_names
        self.feature_types = feature_types
        self.name = name
        self.selector = selector

    def data(self, key=None):
        """Provides specific explanation data.

        Args:
            key: A number/string that references a specific data item.

        Returns:
            A serializable dictionary.
        """
        if key is None:
            return self._internal_obj["overall"]
        specific_dict = self._internal_obj["specific"][key].copy()
        specific_dict["x"] = self._internal_obj["overall"]["X"][:, key]
        specific_dict["y"] = self._internal_obj["overall"]["y"]
        return specific_dict

    def visualize(self, key=None):
        """Provides interactive visualizations.

        Args:
            key: Either a scalar or list
                that indexes the internal object for sub-plotting.
                If an overall visualization is requested, pass None.

        Returns:
            A Plotly figure.
        """
        from ..visual.plot import plot_density
        import plotly.graph_objs as go

        data_dict = self.data(key)
        if data_dict is None:
            return None

        if key is None:
            figure = plot_density(
                data_dict["density"], title="Response", ytitle="Density"
            )
            return figure

        # Show feature graph
        density_dict = data_dict["feature_density"]

        bin_size = density_dict["names"][1] - density_dict["names"][0]
        is_categorical = (
            self.feature_types[key] == "nominal" or self.feature_types[key] == "ordinal"
        )

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
            title=(
                "Pearson Correlation: {0:.3f}".format(corr) if corr is not None else ""
            ),
        )
        fig = go.Figure(data=data, layout=layout)
        return fig


class ClassHistogram(ExplainerMixin):
    """Provides histogram visualizations for classification problems."""

    available_explanations = ["data"]
    explainer_type = "data"

    def __init__(self, feature_names=None, feature_types=None):
        """Initializes class.

        Args:
            feature_names: List of feature names.
            feature_types: List of feature types.
        """
        self.feature_names = feature_names
        self.feature_types = feature_types

    def explain_data(self, X, y, name=None):
        """Generates data explanations (exploratory data analysis)

        Args:
            X: Numpy array for X to explain.
            y: Numpy vector for y to explain.
            name: User-defined explanation name.

        Returns:
            An explanation object.
        """
        if name is None:
            name = gen_name_from_class(self)

        y = clean_dimensions(y, "y")
        if y.ndim != 1:
            raise ValueError("y must be 1 dimensional")

        try:
            y = y.astype(np.float64, copy=False)
        except (TypeError, ValueError):
            # we get a TypeError whenever we have an np.object_ array and numpy attempts to call float(), but the
            # object doesn't have a __float__ function.  We get a ValueError when either a str object inside an
            # np.object_ array or when an np.unicode_ array attempts to convert a string to a float and fails

            y = typify_classification(y)

        X, n_samples = preclean_X(X, self.feature_names, self.feature_types, len(y))

        X, feature_names, feature_types = unify_data(
            X, n_samples, self.feature_names, self.feature_types, False, 0
        )

        unique_val_counts = np.zeros(len(feature_names), dtype=np.int64)
        for col_idx in range(len(feature_names)):
            X_col = X[:, col_idx]
            unique_val_counts[col_idx] = len(np.unique(X_col))

        global_selector = gen_global_selector(
            len(feature_names),
            feature_names,
            feature_types,
            unique_val_counts,
            None,
        )

        overall_dict = {"type": "hist", "X": X, "y": y}
        internal_obj = {
            "overall": overall_dict,
            "specific": None,  # NOTE: Will be generated at data call
        }

        return ClassHistogramExplanation(
            "data",
            internal_obj,
            feature_names=feature_names,
            feature_types=feature_types,
            name=name,
            selector=global_selector,
        )


class ClassHistogramExplanation(ExplanationMixin):
    """Explanation object specific to class histogram explainer."""

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
        self.explanation_type = explanation_type
        self._internal_obj = internal_obj
        self.feature_names = feature_names
        self.feature_types = feature_types
        self.name = name
        self.selector = selector

    def data(self, key=None):
        """Provides specific explanation data.

        Args:
            key: A number/string that references a specific data item.

        Returns:
            A serializable dictionary.
        """
        if key is None:
            return self._internal_obj["overall"]

        specific_dict = {}
        specific_dict["x"] = self._internal_obj["overall"]["X"][:, key]
        specific_dict["y"] = self._internal_obj["overall"]["y"]
        return specific_dict

    def visualize(self, key=None):
        """Provides interactive visualizations.

        Args:
            key: Either a scalar or list
                that indexes the internal object for sub-plotting.
                If an overall visualization is requested, pass None.

        Returns:
            A Plotly figure.
        """
        from ..visual.plot import plot_density, COLORS
        import plotly.graph_objs as go

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
        is_categorical = (
            self.feature_types[key] == "nominal" or self.feature_types[key] == "ordinal"
        )
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
            title=column_name,
            barmode="stack",
            # hoverlabel=dict(font=dict(size=25))
        )
        figure = go.Figure(data, layout)
        figure["layout"]["yaxis1"].update(title="Density")

        return figure
