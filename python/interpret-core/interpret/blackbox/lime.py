# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from ..api.base import ExplainerMixin
from ..api.templates import FeatureValueExplanation

from ..utils import gen_name_from_class, gen_local_selector
from ..utils import gen_perf_dicts
from ..utils import unify_data, unify_predict_fn
import warnings


# TODO: Make kwargs explicit.
class LimeTabular(ExplainerMixin):
    """ Exposes LIME tabular explainer from lime package, in interpret API form.
    If using this please cite the original authors as can be found here: https://github.com/marcotcr/lime/blob/master/citation.bib
    """

    available_explanations = ["local"]
    explainer_type = "blackbox"

    def __init__(
        self,
        predict_fn,
        data,
        sampler=None,
        feature_names=None,
        feature_types=None,
        explain_kwargs={},
        n_jobs=1,
        **kwargs
    ):
        """ Initializes class.

        Args:
            predict_fn: Function of blackbox that takes input, and returns prediction.
            data: Data used to initialize LIME with.
            sampler: Currently unused. Due for deprecation.
            feature_names: List of feature names.
            feature_types: List of feature types.
            explain_kwargs: Kwargs that will be sent to lime's explain_instance.
            n_jobs: Number of jobs to run in parallel.
            **kwargs: Kwargs that will be sent to lime at initialization time.
        """
        from lime.lime_tabular import LimeTabularExplainer

        self.data, _, self.feature_names, self.feature_types = unify_data(
            data, None, feature_names, feature_types
        )
        self.predict_fn = unify_predict_fn(predict_fn, self.data)
        self.n_jobs = n_jobs

        if sampler is not None:  # pragma: no cover
            warnings.warn("Sampler interface not currently supported.")

        self.sampler = sampler
        self.explain_kwargs = explain_kwargs

        self.kwargs = kwargs
        final_kwargs = {"mode": "regression"}
        if self.feature_names:
            final_kwargs["feature_names"] = self.feature_names
        final_kwargs.update(self.kwargs)

        self.lime = LimeTabularExplainer(self.data, **final_kwargs)

    def explain_local(self, X, y=None, name=None):
        """ Generates local explanations for provided instances.

        Args:
            X: Numpy array for X to explain.
            y: Numpy vector for y to explain.
            name: User-defined explanation name.

        Returns:
            An explanation object, visualizing feature-value pairs
            for each instance as horizontal bar charts.
        """
        if name is None:
            name = gen_name_from_class(self)
        X, y, _, _ = unify_data(X, y, self.feature_names, self.feature_types)

        predictions = self.predict_fn(X)
        pred_fn = self.predict_fn

        data_dicts = []
        scores_list = []
        perf_list = []
        perf_dicts = gen_perf_dicts(predictions, y, False)
        for i, instance in enumerate(X):
            lime_explanation = self.lime.explain_instance(
                instance, pred_fn, **self.explain_kwargs
            )

            names = []
            scores = []
            values = []
            feature_idx_imp_pairs = lime_explanation.as_map()[1]
            for feat_idx, imp in feature_idx_imp_pairs:
                names.append(self.feature_names[feat_idx])
                scores.append(imp)
                values.append(instance[feat_idx])
            intercept = lime_explanation.intercept[1]

            perf_dict_obj = None if perf_dicts is None else perf_dicts[i]

            scores_list.append(scores)
            perf_list.append(perf_dict_obj)

            data_dict = {
                "type": "univariate",
                "names": names,
                "perf": perf_dict_obj,
                "scores": scores,
                "values": values,
                "extra": {"names": ["Intercept"], "scores": [intercept], "values": [1]},
            }
            data_dicts.append(data_dict)

        internal_obj = {
            "overall": None,
            "specific": data_dicts,
            "mli": [
                {
                    "explanation_type": "local_feature_importance",
                    "value": {
                        "scores": scores_list,
                        "intercept": intercept,
                        "perf": perf_list,
                    },
                }
            ],
        }
        internal_obj["mli"].append(
            {
                "explanation_type": "evaluation_dataset",
                "value": {"dataset_x": X, "dataset_y": y},
            }
        )
        selector = gen_local_selector(data_dicts, is_classification=False)

        return FeatureValueExplanation(
            "local",
            internal_obj,
            feature_names=self.feature_names,
            feature_types=self.feature_types,
            name=name,
            selector=selector,
        )
