# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from ..api.base import ExplainerMixin, ExplanationMixin
from ..utils import (
    gen_name_from_class,
    gen_global_selector,
    gen_local_selector,
    gen_perf_dicts,
)

from copy import deepcopy
import numpy as np
import pandas as pd
import re

from ..utils._binning import (
    preclean_X,
    unify_data2,
    clean_dimensions,
    typify_classification,
)

import logging

log = logging.getLogger(__name__)


class RulesExplanation(ExplanationMixin):
    """Visualizes rules as HTML for both global and local explanations."""

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
            HTML as string.
        """
        from ..visual.plot import rules_to_html

        data_dict = self.data(key)
        if data_dict is None:
            return None

        # Handle overall graphs
        if key is None and self.explanation_type == "global":
            return rules_to_html(data_dict, title="All Rules")

        # Handle local instance graphs
        if self.explanation_type == "local":
            return rules_to_html(data_dict, title="Triggered Rule")

        # Handle global feature graphs
        elif self.explanation_type == "global":
            return rules_to_html(
                data_dict,
                title="Rules with Feature: {0}".format(self.feature_names[key]),
            )
        # Handle everything else as invalid
        else:  # pragma: no cover
            msg = "Not suppported: {0}, {1}".format(self.explanation_type, key)
            log.error(msg)
            raise Exception(msg)


class DecisionListClassifier(ClassifierMixin, ExplainerMixin):
    """Decision List Classifier

    Currently a slight variant of SkopeRules from skope-rules.
    https://github.com/scikit-learn-contrib/skope-rules

    """

    available_explanations = ["global", "local"]
    explainer_type = "model"

    def __init__(self, feature_names=None, feature_types=None, **kwargs):
        """Initializes class.

        Args:
            feature_names: List of feature names.
            feature_types: List of feature types.
            **kwargs: Kwargs passed to wrapped SkopeRules at initialization time.
        """
        self.feature_names = feature_names
        self.feature_types = feature_types
        self.kwargs = kwargs

    def fit(self, X, y):
        """Fits model to provided instances.

        Args:
            X: Numpy array for training instances.
            y: Numpy array as training labels.

        Returns:
            Itself.
        """
        try:
            from skrules import SkopeRules as SR
        except ImportError:  # NOTE: skoperules loves six, shame it's deprecated.
            import six
            import sys

            sys.modules["sklearn.externals.six"] = six
            from skrules import SkopeRules as SR

        y = clean_dimensions(y, "y")
        if y.ndim != 1:
            msg = "y must be 1 dimensional"
            log.error(msg)
            raise ValueError(msg)
        if len(y) == 0:
            msg = "y cannot have 0 samples"
            log.error(msg)
            raise ValueError(msg)

        X, n_samples = preclean_X(X, self.feature_names, self.feature_types, len(y))

        X, self.feature_names_in_, self.feature_types_in_ = unify_data2(
            X, n_samples, self.feature_names, self.feature_types, False, 0
        )

        y = typify_classification(y)
        # skope-rules handles binary and multiclass classification.
        # It checks this, so we do not need to.

        # TODO: it might be cleaner to pass is no feature names, then use mapping property.  That way if they change their names we'll use their new names

        # skope-rules seems to want to record things in their names, so replicate their format
        # they seem to have changed this to make rules_ use the given feature_names, but just
        # in case we want to look at any of their other internals, use their naming
        BASE_FEATURE_NAME = "__C__"
        self.feature_index_ = [
            BASE_FEATURE_NAME + x
            for x in np.arange(len(self.feature_names_in_)).astype(str)
        ]
        self.feature_map_ = {
            key: name for key, name in zip(self.feature_index_, self.feature_names_in_)
        }
        self.sk_model_ = SR(feature_names=self.feature_index_, **self.kwargs)

        # TODO: it might be better to pass the typified y into skope-rules so that the self.sk_model_
        #       class is more capable, but this would be true only if it's able to handle strings
        self.classes_, y = np.unique(y, return_inverse=True)
        self._class_idx_ = {x: index for index, x in enumerate(self.classes_)}

        self.sk_model_.fit(X, y)
        self.pos_ratio_ = np.mean(y)

        # Extract rules
        (
            self.internal_rules_,
            self.rules_,
            self.prec_,
            self.recall_,
            self.feat_rule_map_,
        ) = self._extract_rules(self.sk_model_.rules_)

        # TODO: move this call into the explain_global function and extract the information needed
        #       in a cleaner way.  Also, look over the above fields to see if we can simplify
        self.global_selector_ = gen_global_selector(
            X, self.feature_names_in_, self.feature_types_in_, None
        )

        self.has_fitted_ = True

        return self

    def predict(self, X):
        """Predicts on provided instances.

        Args:
            X: Numpy array for instances.

        Returns:
            Predicted class label per instance.
        """

        # let predict_proba clean X and check if we are fitted
        scores = self.predict_proba(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def _scores(self, X):
        df = pd.DataFrame(X, columns=self.feature_index_)
        selected_rules = self.internal_rules_

        scores = np.ones(X.shape[0]) * np.inf
        for k, r in enumerate(selected_rules):
            matched_idx = list(df.query(r[0]).index)
            scores[matched_idx] = np.minimum(k, scores[matched_idx])
        scores[np.isinf(scores)] = len(selected_rules)
        scores = scores.astype("int64")

        return scores

    def predict_proba(self, X):
        """Provides probability estimates on provided instances.

        Args:
            X: Numpy array for instances.

        Returns:
            Probability estimate of instance for each class.
        """

        check_is_fitted(self, "has_fitted_")

        X, n_samples = preclean_X(X, self.feature_names_in_, self.feature_types_in_)

        X, _, _ = unify_data2(
            X, n_samples, self.feature_names_in_, self.feature_types_in_, False, 0
        )

        scores = self._scores(X)
        prec_ar = np.array(self.prec_)
        return np.c_[1.0 - prec_ar[scores], prec_ar[scores]]

    def _extract_rules(self, rules):
        rules = deepcopy(rules)
        rules = list(sorted(rules, key=lambda x: x[1][0], reverse=True))

        rule_li = []
        prec_li = []
        recall_li = []
        predict_li = []
        features_dict = {feat: [] for feat in self.feature_names_in_}

        def extract_orig_features(pattern, rule):
            feature_set = set()
            for m in re.finditer(pattern, rule):
                orig_feature = self.feature_map_[m.group(1)]
                feature_set.add(orig_feature)
            return feature_set

        for indx, rule_rec in enumerate(rules):
            rule = rule_rec[0]
            rule_round = " ".join(
                [
                    "{0:.2f}".format(float(x)) if x.replace(".", "", 1).isdigit() else x
                    for x in rule.split(" ")
                ]
            )
            pattern = r"(feature_[0-9]+)"
            feature_set = extract_orig_features(pattern, rule_round)
            rule_fix = re.sub(
                pattern, lambda m: self.feature_map_[m.group(1)], rule_round
            )
            rule_li.append(rule_fix)
            prec_li.append(rule_rec[1][0])
            recall_li.append(rule_rec[1][1])
            predict_li.append(1.0)

            for feat in feature_set:
                features_dict[feat].append(indx)

        # Provide default rule
        rule_li.append("No Rules Triggered")
        prec_li.append(self.pos_ratio_)
        recall_li.append(1.0)
        predict_li.append(0.0)

        return rules, rule_li, prec_li, recall_li, features_dict

    def explain_local(self, X, y=None, name=None):
        """Provides local explanations for provided instances.

        Args:
            X: Numpy array for X to explain.
            y: Numpy vector for y to explain.
            name: User-defined explanation name.

        Returns:
            An explanation object.
        """

        check_is_fitted(self, "has_fitted_")

        if name is None:
            name = gen_name_from_class(self)

        n_samples = None
        if y is not None:
            y = clean_dimensions(y, "y")
            if y.ndim != 1:
                raise ValueError("y must be 1 dimensional")
            n_samples = len(y)

            y = typify_classification(y)
            y = np.array([self._class_idx_[el] for el in y], dtype=np.int64)

        X, n_samples = preclean_X(
            X, self.feature_names_in_, self.feature_types_in_, n_samples
        )

        # predict and predict_proba call unify_data already, so call them before unifying
        # but after doing the light cleaning for iterators
        outcomes = self.predict(X)
        predictions = self.predict_proba(X)

        X, _, _ = unify_data2(
            X, n_samples, self.feature_names_in_, self.feature_types_in_, False, 0
        )

        scores = self._scores(X)

        perf_dicts = gen_perf_dicts(predictions, y, True)
        data_dicts = []
        for idx, score in enumerate(scores):
            data_dict = {
                "type": "rule",
                "rule": [self.rules_[score]],
                "precision": [self.prec_[score]],
                "recall": [self.recall_[score]],
                "outcome": [outcomes[idx]],
            }
            perf_dict_obj = None if perf_dicts is None else perf_dicts[idx]
            data_dict["perf"] = perf_dict_obj
            data_dicts.append(data_dict)

        internal_obj = {"overall": None, "specific": data_dicts}
        selector = gen_local_selector(data_dicts, is_classification=True)

        return RulesExplanation(
            "local",
            internal_obj,
            feature_names=self.feature_names_in_,
            feature_types=self.feature_types_in_,
            name=name,
            selector=selector,
        )

    def explain_global(self, name=None):
        """Provides global explanation for model.

        Args:
            name: User-defined explanation name.

        Returns:
            An explanation object.
        """

        check_is_fitted(self, "has_fitted_")

        if name is None:
            name = gen_name_from_class(self)

        # Extract rules
        rules, prec, recall, feat_rule_map = (
            self.rules_,
            self.prec_,
            self.recall_,
            self.feat_rule_map_,
        )

        outcomes = [self.classes_[1]] * (len(self.rules_) - 1)
        # Add the zero case for the default rule
        outcomes.append(self.classes_[0])
        overall_data_dict = {
            "type": "rule",
            "rule": rules,
            "precision": prec,
            "recall": recall,
            "outcome": outcomes,
        }
        data_dicts = [
            {
                "type": "rule",
                "rule": [rules[i] for i in feat_rule_map[feature]],
                "precision": [prec[i] for i in feat_rule_map[feature]],
                "recall": [recall[i] for i in feat_rule_map[feature]],
                "outcome": [outcomes[i] for i in feat_rule_map[feature]],
            }
            for feature in self.feature_names_in_
        ]

        internal_obj = {"overall": overall_data_dict, "specific": data_dicts}

        return RulesExplanation(
            "global",
            internal_obj,
            feature_names=self.feature_names_in_,
            feature_types=self.feature_types_in_,
            name=name,
            selector=self.global_selector_,
        )
