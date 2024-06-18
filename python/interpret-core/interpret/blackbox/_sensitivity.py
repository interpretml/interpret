# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

from ..api.base import ExplainerMixin
from ..api.templates import FeatureValueExplanation
from ..utils._explanation import gen_name_from_class, gen_global_selector

from abc import ABC, abstractmethod
import numpy as np

from ..utils._clean_x import preclean_X
from ..utils._unify_predict import determine_classes, unify_predict_fn
from ..utils._unify_data import unify_data


# TODO: move this to a more general location where other blackbox methods can access it
class SamplerMixin(ABC):
    @abstractmethod
    def sample(self, data, feature_names, feature_types):
        # if the blackbox or greybox underlying method accepts a sampling
        # function or abstract class they may want to pass us additional
        # options, which the class that derrives from SamplerMixin may
        # want to add as either explicit arguments or kwargs
        pass  # pragma: no cover


class MorrisSampler(SamplerMixin):
    def __init__(self, N=1000, num_levels=4, **kwargs):
        self.N = N
        self.num_levels = num_levels
        self.kwargs = kwargs

    def sample(self, data, feature_names, feature_types):
        # data, feature_names, and feature_types are materialized after unify_data has been called

        from SALib.sample import morris as morris_sampler

        problem = _gen_problem_from_data(data, feature_names)
        return morris_sampler.sample(
            problem, N=self.N, num_levels=self.num_levels, **self.kwargs
        )


class MorrisSensitivity(ExplainerMixin):
    """Method of Morris for analyzing blackbox systems.
    If using this please cite the package owners as can be found here: https://github.com/SALib/SALib

    Morris, Max D. "Factorial sampling plans for preliminary computational experiments."
    Technometrics 33.2 (1991): 161-174.
    """

    available_explanations = ["global"]
    explainer_type = "blackbox"

    def __init__(
        self,
        model,
        data,
        feature_names=None,
        feature_types=None,
        sampler=None,
        **kwargs
    ):
        """Initializes class.

        Args:
            model: model or prediction function of model (predict_proba for classification or predict for regression)
            data: Data used to initialize LIME with.
            feature_names: List of feature names.
            feature_types: List of feature types.
            sampler: A SamplerMixin derrived class that can generate samples from data
            **kwargs: Kwargs that will be sent to SALib.analyze.morris.analyze
        """

        from SALib.analyze import morris

        data, n_samples = preclean_X(data, feature_names, feature_types)

        predict_fn, n_classes, _ = determine_classes(model, data, n_samples)
        if 3 <= n_classes:
            raise Exception("multiclass MorrisSensitivity not supported")
        predict_fn = unify_predict_fn(predict_fn, data, 1 if n_classes == 2 else -1)

        data, self.feature_names_in_, self.feature_types_in_ = unify_data(
            data, n_samples, feature_names, feature_types, False, 0
        )

        # SALib does not support string categoricals, and np.object_ is slower,
        # so convert to np.float64 until we implement some automatic categorical handling
        # Fortran ordered is faster since we go by columns, so use that
        data = data.astype(np.float64, order="F", copy=False)

        if sampler is None:
            sampler = MorrisSampler()

        samples = sampler.sample(data, self.feature_names_in_, self.feature_types_in_)
        problem = _gen_problem_from_data(data, self.feature_names_in_)
        analysis = morris.analyze(
            problem, samples, predict_fn(samples).astype(float), **kwargs
        )

        # TODO: see if we can clean up these datatypes to simpler and easier to understand

        self.mu_ = analysis["mu"]
        self.mu_star_ = analysis["mu_star"]
        self.sigma_ = analysis["sigma"]
        self.mu_star_conf_ = analysis["mu_star_conf"]

        mask = self.mu_star_ > 0
        self.convergence_index_ = np.max(
            np.array(self.mu_star_conf_)[mask] / np.array(self.mu_star_)[mask]
        )

        unique_val_counts = np.zeros(len(self.feature_names_in_), dtype=np.int64)
        for col_idx, feature in enumerate(self.feature_names_in_):
            X_col = data[:, col_idx]
            unique_val_counts[col_idx] = len(np.unique(X_col))

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

        overall_data_dict = {
            "names": self.feature_names_in_,
            "scores": self.mu_star_,
            "convergence_index": self.convergence_index_,
        }

        specific_data_dicts = []
        for feat_idx, feature_name in enumerate(self.feature_names_in_):
            specific_data_dict = {
                "type": "morris",
                "mu": self.mu_[feat_idx],
                "mu_star": self.mu_star_[feat_idx],
                "sigma": self.sigma_[feat_idx],
                "mu_star_conf": self.mu_star_conf_[feat_idx],
            }
            specific_data_dicts.append(specific_data_dict)

        internal_obj = {"overall": overall_data_dict, "specific": specific_data_dicts}

        global_selector = gen_global_selector(
            len(self.feature_names_in_),
            self.feature_names_in_,
            self.feature_types_in_,
            self.unique_val_counts_,
            self.mu_star_,
        )

        return MorrisExplanation(
            "global",
            internal_obj,
            feature_names=self.feature_names_in_,
            feature_types=self.feature_types_in_,
            name=name,
            selector=global_selector,
        )


class MorrisExplanation(FeatureValueExplanation):
    """Visualizations specific to Method of Morris."""

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

        super(MorrisExplanation, self).__init__(
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
            HTML as a string.
        """
        from ..visual.plot import plot_horizontal_bar, sort_take

        data_dict = self.data(key)
        if data_dict is None:
            return None

        if self.explanation_type == "global" and key is None:
            data_dict = sort_take(
                data_dict, sort_fn=lambda x: -abs(x), top_n=15, reverse_results=True
            )
            title = "Morris Sensitivity<br>Convergence Index: {0:.3f}".format(
                data_dict["convergence_index"]
            )
            figure = plot_horizontal_bar(data_dict, start_zero=True, title=title)
            return figure

        if self.explanation_type == "global" and key is not None:
            multi_html_template = r"""
                <style>
                .container {{
                    display: flex;
                    justify-content: center;
                    flex-direction: column;
                    text-align: center;
                    align-items: center;
                }}
                .row {{
                    width: 50%;
                    flex: none;
                }}
                .dotted-hr {{
                    border: none;
                    border-top: 1px dotted black;
                }}
                </style>
                <div class='container'>
                <div class='row'>
                    <div>
                        <p>
                            <h1>Morris Analysis<br/>{feature_name}</h1>
                        </p>
                    </div>
                    <hr>
                    {analyses}
                </div>
                </div>
            """
            analysis_template = r"""
                <p>
                    <h2>
                    Mu: {mu:.3f}
                    <br/>
                    Mu_star: {mu_star:.3f}
                    <br/>
                    Sigma: {sigma:.3f}
                    <br/>
                    Mu_star Confidence: {mu_star_conf:.3f}
                    </h2>
                </p>
                <hr class='dotted-hr'/>
            """

            analysis = analysis_template.format(
                mu=data_dict["mu"],
                mu_star=data_dict["mu_star"],
                sigma=data_dict["sigma"],
                mu_star_conf=data_dict["mu_star_conf"],
            )

            html_str = multi_html_template.format(
                feature_name=self.feature_names[key], analyses=analysis
            )
            return html_str

        return super().visualize(key)


def _soft_min_max(values, soft_add=1, soft_bounds=1):
    """Returns [min, max + soft_add] if difference of min and max is less
        than the soft bound.

    Args:
        values: Iterable of numeric.
        soft_add:  Increment to max if difference is too small.
        soft_bounds:  If difference is smaller than this, add a soft increment.

    Returns:
        A list [min, max + soft_add] if abs(max-min) is less than soft bound.
    """
    min_val = min(values)
    max_val = max(values)

    diff_val = max_val - min_val
    max_increment = soft_add if abs(diff_val) < soft_bounds else 0
    return [min_val, max_val + max_increment]


def _gen_problem_from_data(data, feature_names):
    bounds = [_soft_min_max(data[:, i]) for i, _ in enumerate(feature_names)]
    problem = {
        "num_vars": len(feature_names),
        "names": feature_names,
        "bounds": bounds,
    }
    return problem
