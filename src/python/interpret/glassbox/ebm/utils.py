# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license
# TODO: Test EBMUtils

from sklearn.utils.extmath import softmax
import numpy as np

import logging

log = logging.getLogger(__name__)


# TODO: Clean up
class EBMUtils:
    @staticmethod
    def gen_attributes(col_types, col_n_bins):
        # Create Python form of attributes
        # Undocumented.
        attributes = [None] * len(col_types)
        for col_idx, _ in enumerate(attributes):
            attributes[col_idx] = {
                # NOTE: Ordinal only handled at native, override.
                # 'type': col_types[col_idx],
                "type": "continuous",
                # NOTE: Missing not implemented at native, always set to false.
                "has_missing": False,
                "n_bins": col_n_bins[col_idx],
            }
        return attributes

    @staticmethod
    def gen_attribute_sets(attribute_indices):
        attribute_sets = [None] * len(attribute_indices)
        for i, indices in enumerate(attribute_indices):
            attribute_set = {"n_attributes": len(indices), "attributes": indices}
            attribute_sets[i] = attribute_set
        return attribute_sets

    @staticmethod
    def scores_by_attrib_set(
        X, attribute_sets, attribute_set_models, skip_attr_set_idxs=[]
    ):

        for set_idx, attribute_set in enumerate(attribute_sets):
            if set_idx in skip_attr_set_idxs:
                continue
            tensor = attribute_set_models[set_idx]

            # Get the current column(s) to process
            attr_idxs = attribute_set["attributes"]
            sliced_X = X[:, attr_idxs]
            scores = tensor[tuple(sliced_X.T)]

            yield set_idx, attribute_set, scores

    @staticmethod
    def decision_function(
        X, attribute_sets, attribute_set_models, intercept, skip_attr_set_idxs=[]
    ):

        if X.ndim == 1:
            X = X.reshape(1, X.shape[0])

        # Foreach column, add log odds per instance
        score_vector = np.zeros(X.shape[0])
        score_vector += intercept

        scores_gen = EBMUtils.scores_by_attrib_set(
            X, attribute_sets, attribute_set_models, skip_attr_set_idxs
        )
        for _, _, scores in scores_gen:
            score_vector += scores

        if not np.all(np.isfinite(score_vector)):  # pragma: no cover
            msg = "Non-finite values present in log odds vector."
            log.error(msg)
            raise Exception(msg)

        return score_vector

    @staticmethod
    def classifier_predict_proba(X, estimator, skip_attr_set_idxs=[]):
        log_odds_vector = EBMUtils.decision_function(
            X,
            estimator.attribute_sets_,
            estimator.attribute_set_models_,
            estimator.intercept_,
            skip_attr_set_idxs,
        )
        log_odds_trans = np.c_[-log_odds_vector, log_odds_vector]
        scores = softmax(log_odds_trans, copy=True)
        return scores

    @staticmethod
    def classifier_predict(X, estimator, skip_attr_set_idxs=[]):
        scores = EBMUtils.classifier_predict_proba(X, estimator, skip_attr_set_idxs)
        return estimator.classes_[np.argmax(scores, axis=1)]

    @staticmethod
    def regressor_predict(X, estimator, skip_attr_set_idxs=[]):
        scores = EBMUtils.decision_function(
            X,
            estimator.attribute_sets_,
            estimator.attribute_set_models_,
            estimator.intercept_,
            skip_attr_set_idxs,
        )
        return scores

    @staticmethod
    def gen_feature_name(attr_idxs, col_names):
        feature_name = []
        for attribute_index in attr_idxs:
            feature_name.append(col_names[attribute_index])
        feature_name = " x ".join(feature_name)
        return feature_name

    @staticmethod
    def gen_feature_type(attr_idxs, col_types):
        if len(attr_idxs) == 1:
            return col_types[attr_idxs[0]]
        else:
            return "pairwise"
