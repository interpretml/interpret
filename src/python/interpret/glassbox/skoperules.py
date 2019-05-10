# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from sklearn.base import ClassifierMixin
from ..api.base import ExplainerMixin, ExplanationMixin
from ..utils import gen_name_from_class, gen_global_selector, gen_local_selector
from ..utils import unify_data
from ..visual.plot import rules_to_html

from skrules import SkopeRules as SR
from copy import deepcopy
import numpy as np
import pandas as pd
import re

import logging

log = logging.getLogger(__name__)


class RulesExplanation(ExplanationMixin):
    """
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
    """ Decision List Classifier

    Currently a slight variant of SkopeRules from skope-rules.
    https://github.com/scikit-learn-contrib/skope-rules

    """

    available_explanations = ["global", "local"]
    explainer_type = "model"

    def __init__(self, feature_names=None, feature_types=None, **kwargs):
        """ Initializes skope rules.

        Args:
            **kwargs: Keyword arguments to be passed to SkopeRules
                in skope-rules.
        """
        self.feature_names = feature_names
        self.feature_types = feature_types
        self.kwargs = kwargs

    def fit(self, X, y):
        """ Fits model to provided instances.

        Args:
            X: Numpy array for training instances.
            y: Numpy array as training labels.

        Returns:
            Itself.
        """
        X, y, self.feature_names, self.feature_types = unify_data(
            X, y, self.feature_names, self.feature_types
        )
        self.feature_index_ = [
            "feature_" + str(i) for i, v in enumerate(self.feature_names)
        ]
        self.feature_map_ = {
            v: self.feature_names[i] for i, v in enumerate(self.feature_index_)
        }
        self.sk_model_ = SR(feature_names=self.feature_index_, **self.kwargs)

        self.classes_, y = np.unique(y, return_inverse=True)
        self.sk_model_.fit(X, y)
        self.pos_ratio_ = np.mean(y)

        # Extract rules
        self.internal_rules_, self.rules_, self.prec_, self.recall_, self.feat_rule_map_ = self._extract_rules(
            self.sk_model_.rules_
        )

        self.global_selector = gen_global_selector(
            X, self.feature_names, self.feature_types, None
        )

        return self

    def predict(self, X):
        """ Predicts on provided instances.

        Args:
            X: Numpy array for instances.

        Returns:
            Predicted class label per instance.
        """

        X, _, _, _ = unify_data(X, None, self.feature_names, self.feature_types)
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
        """ Provides probability estimates on provided instances.

        Args:
            X: Numpy array for instances.

        Returns:
            Probability estimate of instance for each class.
        """

        X, _, _, _ = unify_data(X, None, self.feature_names, self.feature_types)
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
        features_dict = {feat: [] for feat in self.feature_names}

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
        if name is None:
            name = gen_name_from_class(self)

        X, y, _, _ = unify_data(X, y, self.feature_names, self.feature_types)

        scores = self._scores(X)
        outcomes = self.predict(X)
        prob_scores = self.predict_proba(X)

        data_dicts = []
        for idx, score in enumerate(scores):
            data_dict = {
                "type": "rule",
                "rule": [self.rules_[score]],
                "precision": [self.prec_[score]],
                "recall": [self.recall_[score]],
                "outcome": [outcomes[idx]],
            }
            data_dicts.append(data_dict)

        internal_obj = {"overall": None, "specific": data_dicts}
        selector = gen_local_selector(X, y, prob_scores[:, 1])

        return RulesExplanation(
            "local",
            internal_obj,
            feature_names=self.feature_names,
            feature_types=self.feature_types,
            name=name,
            selector=selector,
        )

    def explain_global(self, name=None):
        """ Provides global explanation for model.

        Args:
            name: User-defined explanation name.

        Returns:
            An explanation object,
            visualizing feature-value pairs as horizontal bar chart.
        """
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
            for feature in self.feature_names
        ]

        internal_obj = {"overall": overall_data_dict, "specific": data_dicts}

        return RulesExplanation(
            "global",
            internal_obj,
            feature_names=self.feature_names,
            feature_types=self.feature_types,
            name=name,
            selector=self.global_selector,
        )
