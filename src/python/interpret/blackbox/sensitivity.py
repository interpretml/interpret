# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from ..api.base import ExplainerMixin
from ..api.templates import FeatureValueExplanation
from ..utils import unify_predict_fn, unify_data
from ..utils import gen_name_from_class, gen_global_selector
from ..visual.plot import plot_horizontal_bar, sort_take

from abc import ABC, abstractmethod
from SALib.sample import morris as morris_sampler
from SALib.analyze import morris
import numpy as np


class SamplerMixin(ABC):
    @abstractmethod
    def sample(self):
        pass  # pragma: no cover


class MorrisSampler(SamplerMixin):
    def __init__(self, data, feature_names, N=1000, **kwargs):
        self.data = data
        self.feature_names = feature_names
        self.N = N
        self.kwargs = kwargs

    def sample(self):
        kwargs = {"num_levels": 4}
        kwargs.update(self.kwargs)

        problem = self.gen_problem_from_data(self.data, self.feature_names)
        return morris_sampler.sample(problem, N=self.N, **kwargs)

    @classmethod
    def gen_problem_from_data(cls, data, feature_names):
        bounds = [soft_min_max(data[:, i]) for i, _ in enumerate(feature_names)]
        problem = {
            "num_vars": len(feature_names),
            "names": feature_names,
            "bounds": bounds,
        }
        return problem


class MorrisSensitivity(ExplainerMixin):
    available_explanations = ["global"]
    explainer_type = "blackbox"

    def __init__(
        self, predict_fn, data, sampler=None, feature_names=None, feature_types=None
    ):
        self.data, _, self.feature_names, self.feature_types = unify_data(
            data, None, feature_names, feature_types
        )
        self.predict_fn = unify_predict_fn(predict_fn, self.data)
        self.sampler = sampler

        if self.sampler is None:
            self.sampler = MorrisSampler(self.data, self.feature_names)

    def explain_global(self, name=None):
        if name is None:
            name = gen_name_from_class(self)

        samples = self.sampler.sample()
        problem = self.sampler.gen_problem_from_data(self.data, self.feature_names)
        analysis = morris.analyze(problem, samples, self.predict_fn(samples))

        mu = analysis["mu"]
        mu_star = analysis["mu_star"]
        sigma = analysis["sigma"]
        mu_star_conf = analysis["mu_star_conf"]

        mask = mu_star > 0
        convergence_index = np.max(
            np.array(mu_star_conf)[mask] / np.array(mu_star)[mask]
        )

        overall_data_dict = {
            "names": self.feature_names,
            "scores": mu_star,
            "convergence_index": convergence_index,
        }

        specific_data_dicts = []
        for feat_idx, feature_name in enumerate(self.feature_names):
            specific_data_dict = {
                "type": "morris",
                "mu": mu[feat_idx],
                "mu_star": mu_star[feat_idx],
                "sigma": sigma[feat_idx],
                "mu_star_conf": mu_star_conf[feat_idx],
            }
            specific_data_dicts.append(specific_data_dict)

        internal_obj = {"overall": overall_data_dict, "specific": specific_data_dicts}

        global_selector = gen_global_selector(
            self.data, self.feature_names, self.feature_types, mu_star
        )

        return MorrisExplanation(
            "global",
            internal_obj,
            feature_names=self.feature_names,
            feature_types=self.feature_types,
            name=name,
            selector=global_selector,
        )


class MorrisExplanation(FeatureValueExplanation):
    """ Visualizes specifically for SA.
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

        super(MorrisExplanation, self).__init__(
            explanation_type,
            internal_obj,
            feature_names=feature_names,
            feature_types=feature_types,
            name=name,
            selector=selector,
        )

    def visualize(self, key=None):
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


def soft_min_max(values, soft_add=1, soft_bounds=1):
    """ Returns [min, max + soft_add] if difference of min and max is less
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
