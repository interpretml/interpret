# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

from ..api.base import ExplainerMixin, ExplanationMixin
import numpy as np
from ..utils._explanation import gen_name_from_class, gen_global_selector

from ..utils._clean_x import preclean_X
from ..utils._unify_predict import determine_classes, unify_predict_fn
from ..utils._unify_data import unify_data


def _unique_grid_points(values):
    unique_points = np.unique(values)
    unique_points.sort()
    return unique_points


def _percentile_grid_points(values, num_points=10):
    percentiles = np.linspace(0, 100, num=num_points)
    grid_points = np.percentile(values, percentiles)
    return grid_points


# def _equal_spaced_grid_points(values, num_points=10):
#     grid_points = np.linspace(min(values), max(values), num=num_points)
#     return grid_points


def _gen_pdp(
    X,
    predict_fn,
    col_idx,
    feature_type,
    num_points=10,
    std_coef=1.0,
    num_ice_samples=10,
):
    num_uniq_vals = len(np.unique(X[:, col_idx]))
    if (
        feature_type == "nominal"
        or feature_type == "ordinal"
        or num_uniq_vals <= num_points
    ):
        grid_points = _unique_grid_points(X[:, col_idx])
        values, counts = np.unique(X[:, col_idx], return_counts=True)
    else:
        grid_points = _percentile_grid_points(X[:, col_idx], num_points=num_points)
        counts, values = np.histogram(X[:, col_idx], bins="doane")

    X_mut = X.copy()
    ice_lines = np.zeros((X.shape[0], grid_points.shape[0]))
    for idx, grid_point in enumerate(grid_points):
        X_mut[:, col_idx] = grid_point
        ice_lines[:, idx] = predict_fn(X_mut)
    mean = np.mean(ice_lines, axis=0)
    std = np.std(ice_lines, axis=0)

    ice_lines = ice_lines[
        np.random.choice(ice_lines.shape[0], num_ice_samples, replace=False), :
    ]

    return {
        "type": "univariate",
        "names": grid_points,
        "scores": mean,
        # TODO: can we get rid of this column of X?
        "values": X[:, col_idx],
        "density": {"names": values, "scores": counts},
        # NOTE: We can take either bounds or background values, picked one.
        "upper_bounds": mean + std * std_coef,
        "lower_bounds": mean - std * std_coef,
        "background_scores": ice_lines,
    }


class PartialDependence(ExplainerMixin):
    """Partial dependence plots as defined in Friedman's paper on
    "Greedy function approximation: a gradient boosting machine".

    Friedman, Jerome H. "Greedy function approximation: a gradient boosting machine."
    Annals of statistics (2001): 1189-1232.
    """

    available_explanations = ["global"]
    explainer_type = "blackbox"

    def __init__(
        self,
        model,
        data,
        feature_names=None,
        feature_types=None,
        num_points=10,
        std_coef=1.0,
    ):
        """Initializes class.

        Args:
            model: model or prediction function of model (predict_proba for classification or predict for regression)
            data: Data used to initialize PartialDependence with.
            feature_names: List of feature names.
            feature_types: List of feature types.
            num_points: Number of grid points for the x axis.
            std_coef: Co-efficient for standard deviation.
        """

        data, n_samples = preclean_X(data, feature_names, feature_types)

        predict_fn, n_classes, _ = determine_classes(model, data, n_samples)
        if 3 <= n_classes:
            raise Exception("multiclass PDP not supported")
        predict_fn = unify_predict_fn(predict_fn, data, 1 if n_classes == 2 else -1)

        data, self.feature_names_in_, self.feature_types_in_ = unify_data(
            data, n_samples, feature_names, feature_types, False, 0
        )

        # Fortran ordered float data is faster since we go by columns, so use that
        data = data.astype(np.float64, order="F", copy=False)

        pdps = []
        unique_val_counts = np.zeros(len(self.feature_names_in_), dtype=np.int64)
        for col_idx, feature in enumerate(self.feature_names_in_):
            feature_type = self.feature_types_in_[col_idx]
            pdp = _gen_pdp(
                data,
                predict_fn,
                col_idx,
                feature_type,
                num_points=num_points,
                std_coef=std_coef,
            )
            pdps.append(pdp)

            X_col = data[:, col_idx]
            unique_val_counts[col_idx] = len(np.unique(X_col))

        # TODO: we can probably extract the data in pdps_ to be less opaque
        # to this class and construct the JSONable data later
        self.pdps_ = pdps
        self.unique_val_counts_ = unique_val_counts

    def explain_global(self, name=None):
        """Provides approximate global explanation for blackbox model.

        Args:
            name: User-defined explanation name.

        Returns:
            An explanation object, visualizes dependence plots.
        """
        if name is None:
            name = gen_name_from_class(self)

        data_dicts = []
        feature_list = []
        density_list = []
        for col_idx, feature in enumerate(self.feature_names_in_):
            pdp = self.pdps_[col_idx]
            feature_dict = {
                "feature_values": pdp["values"],
                "scores": pdp["scores"],
                "upper_bounds": pdp["upper_bounds"],
                "lower_bounds": pdp["lower_bounds"],
            }
            feature_list.append(feature_dict)
            density_list.append(pdp["density"])
            data_dicts.append(pdp)

        internal_obj = {
            "overall": None,
            "specific": data_dicts,
            "mli": [
                {"explanation_type": "pdp", "value": {"feature_list": feature_list}},
                {"explanation_type": "density", "value": {"density": density_list}},
            ],
        }

        selector = gen_global_selector(
            len(self.feature_names_in_),
            self.feature_names_in_,
            self.feature_types_in_,
            self.unique_val_counts_,
            None,
        )

        return PDPExplanation(
            "global",
            internal_obj,
            feature_names=self.feature_names_in_,
            feature_types=self.feature_types_in_,
            name=name,
            selector=selector,
        )


class PDPExplanation(ExplanationMixin):
    """Visualizes explanation as a partial dependence plot."""

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
        return self._internal_obj["specific"][key]

    def visualize(self, key=None):
        """Provides interactive visualizations.

        Args:
            key: Either a scalar or list
                that indexes the internal object for sub-plotting.
                If an overall visualization is requested, pass None.

        Returns:
            A Plotly figure.
        """
        from ..visual.plot import plot_line, plot_bar

        data_dict = self.data(key)
        if data_dict is None:
            return None

        feature_type = self.feature_types[key]
        feature_name = self.feature_names[key]
        if feature_type == "continuous":
            figure = plot_line(data_dict, title=feature_name)
        elif feature_type == "nominal" or feature_type == "ordinal":
            figure = plot_bar(data_dict, title=feature_name)
        else:
            raise Exception("Feature type {0} is not supported.".format(feature_type))

        figure["layout"]["yaxis1"].update(title="Average Response")
        return figure
