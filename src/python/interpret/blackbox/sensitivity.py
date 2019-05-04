# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from ..api.base import ExplainerMixin
from ..api.templates import FeatureValueExplanation
from ..utils import unify_predict_fn, unify_data

from abc import ABC, abstractmethod
from SALib.sample import morris as morris_sampler
from SALib.analyze import morris
from ..utils import gen_name_from_class, gen_global_selector

class SamplerMixin(ABC):
    @abstractmethod
    def sample(self):
        pass # pragma: no cover


class MorrisSampler(SamplerMixin):
    def __init__(self, data, feature_names, N=1000, **kwargs):
        self.data = data
        self.feature_names = feature_names
        self.N = N
        self.kwargs = kwargs

    def sample(self):
        kwargs = {
            'num_levels': 4,
        }
        kwargs.update(self.kwargs)

        problem = self.gen_problem_from_data(self.data, self.feature_names)
        return morris_sampler.sample(problem, N=self.N, **kwargs)

    @classmethod
    def gen_problem_from_data(cls, data, feature_names):
        bounds = [
            soft_min_max(data[:, i]) for i, _ in enumerate(feature_names)
        ]
        problem = {
            'num_vars': len(feature_names),
            'names': feature_names,
            'bounds': bounds,
        }
        return problem


class MorrisSensitivity(ExplainerMixin):
    available_explanations = ['global']
    explainer_type = 'blackbox'

    def __init__(self, predict_fn, data, sampler=None,
                 feature_names=None, feature_types=None):
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
        problem = self.sampler.gen_problem_from_data(
            self.data, self.feature_names
        )
        analysis = morris.analyze(
            problem, samples, self.predict_fn(samples)
        )

        contributions = analysis['mu_star']
        data_dict = {
            'names': self.feature_names,
            'scores': contributions,
        }
        internal_obj = {
            'overall': data_dict,
            'specific': None,
        }

        global_selector = gen_global_selector(
            self.data, self.feature_names, self.feature_types,
            contributions
        )

        return FeatureValueExplanation(
            'global', internal_obj,
            feature_names=self.feature_names,
            feature_types=self.feature_types,
            name=name,
            selector=global_selector
        )


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
