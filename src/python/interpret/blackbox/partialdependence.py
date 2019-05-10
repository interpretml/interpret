# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from ..api.base import ExplainerMixin, ExplanationMixin
import numpy as np
import warnings
from ..utils import gen_name_from_class, gen_global_selector
from ..utils import unify_data, unify_predict_fn
from ..visual.plot import plot_line, plot_bar


class PartialDependence(ExplainerMixin):
    available_explanations = ["global"]
    explainer_type = "blackbox"

    def __init__(
        self,
        predict_fn,
        data,
        sampler=None,
        feature_names=None,
        feature_types=None,
        num_points=10,
        std_coef=1.0,
    ):

        self.data, _, self.feature_names, self.feature_types = unify_data(
            data, None, feature_names, feature_types
        )
        self.predict_fn = unify_predict_fn(predict_fn, self.data)
        self.num_points = num_points
        self.std_coef = std_coef

        if sampler is not None:  # pragma: no cover
            warnings.warn("Sampler interface not currently supported.")
        self.sampler = sampler

    @classmethod
    def _unique_grid_points(cls, values):
        unique_points = np.unique(values)
        unique_points.sort()
        return unique_points

    @classmethod
    def _percentile_grid_points(cls, values, num_points=10):
        percentiles = np.linspace(0, 100, num=num_points)
        grid_points = np.percentile(values, percentiles)
        return grid_points

    # @classmethod
    # def _equal_spaced_grid_points(cls, values, num_points=10):
    #     grid_points = np.linspace(min(values), max(values), num=num_points)
    #     return grid_points

    @classmethod
    def _gen_pdp(
        cls,
        X,
        predict_fn,
        col_idx,
        feature_type,
        num_points=10,
        std_coef=1.0,
        num_ice_samples=10,
    ):

        num_uniq_vals = len(np.unique(X[:, col_idx]))
        if feature_type == "categorical" or num_uniq_vals <= num_points:
            grid_points = cls._unique_grid_points(X[:, col_idx])
            values, counts = np.unique(X[:, col_idx], return_counts=True)
        else:
            grid_points = cls._percentile_grid_points(
                X[:, col_idx], num_points=num_points
            )
            counts, values = np.histogram(X[:, col_idx], bins="doane")

        X_mut = X.copy()
        ice_lines = np.zeros((X.shape[0], grid_points.shape[0]))
        for idx, grid_point in enumerate(grid_points):
            X_mut[:, col_idx] = grid_point
            ice_lines[:, idx] = predict_fn(X_mut)
        mean = np.mean(ice_lines, axis=0)
        # std = np.std(ice_lines, axis=0)

        ice_lines = ice_lines[
            np.random.choice(ice_lines.shape[0], num_ice_samples, replace=False), :
        ]

        return {
            "type": "univariate",
            "names": grid_points,
            "scores": mean,
            "values": X[:, col_idx],
            "density": {"names": values, "scores": counts},
            # NOTE: We can take either bounds or background values, picked one.
            # 'upper_bounds': mean + std * std_coef,
            # 'lower_bounds': mean - std * std_coef,
            "background_scores": ice_lines,
        }

    def explain_global(self, name=None):
        if name is None:
            name = gen_name_from_class(self)

        data_dicts = []
        for col_idx, feature in enumerate(self.feature_names):
            feature_type = self.feature_types[col_idx]
            pdp = PartialDependence._gen_pdp(
                self.data,
                self.predict_fn,
                col_idx,
                feature_type,
                num_points=self.num_points,
                std_coef=self.std_coef,
            )
            data_dicts.append(pdp)

        internal_obj = {"overall": None, "specific": data_dicts}

        selector = gen_global_selector(
            self.data, self.feature_names, self.feature_types, None
        )

        return PDPExplanation(
            "global",
            internal_obj,
            feature_names=self.feature_names,
            feature_types=self.feature_types,
            name=name,
            selector=selector,
        )


# TODO: State criteria in docs.
class PDPExplanation(ExplanationMixin):
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
        return self._internal_obj["specific"][key]

    def visualize(self, key=None):
        data_dict = self.data(key)
        if data_dict is None:
            return None

        feature_type = self.feature_types[key]
        feature_name = self.feature_names[key]
        if feature_type == "continuous":
            figure = plot_line(data_dict, title=feature_name)
        elif feature_type == "categorical":
            figure = plot_bar(data_dict, title=feature_name)
        else:
            raise Exception("Feature type {0} is not supported.".format(feature_type))

        figure["layout"]["yaxis1"].update(title="Average Response")
        return figure
