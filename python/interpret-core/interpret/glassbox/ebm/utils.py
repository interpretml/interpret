# Copyright (c) 2019 Microsoft Corporation

# Distributed under the MIT software license
# TODO: Test EBMUtils

# from scipy.special import expit
from sklearn.utils.extmath import softmax
from sklearn.model_selection import train_test_split
import numbers
import numpy as np


import logging

log = logging.getLogger(__name__)


# TODO: Clean up
class EBMUtils:
    @staticmethod
    def get_count_scores_c(n_classes):
        # this should reflect how the C code represents scores
        return 1 if n_classes <= 2 else n_classes

    @staticmethod
    def ebm_train_test_split(X, y, test_size, random_state, is_classification):
        # all test/train splits should be done with this function to ensure that
        # if we re-generate the train/test splits that they are generated exactly
        # the same as before
        if test_size > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=y if is_classification else None,
            )
        elif test_size == 0:
            X_train = X
            y_train = y
            X_val = np.empty(shape=(0, 0), dtype=X.dtype)
            y_val = np.empty(shape=(0), dtype=y.dtype)
        else:  # pragma: no cover
            raise Exception("test_size must be between 0 and 1.")

        return X_train, X_val, y_train, y_val

    @staticmethod
    def gen_features(col_types, col_n_bins):
        # Create Python form of features
        # Undocumented.
        features = [None] * len(col_types)
        for col_idx, _ in enumerate(features):
            features[col_idx] = {
                # NOTE: Ordinal only handled at native, override.
                # 'type': col_types[col_idx],
                "type": "continuous",
                # NOTE: Missing not implemented at native, always set to false.
                "has_missing": False,
                "n_bins": col_n_bins[col_idx],
            }
        return features

    @staticmethod
    def gen_feature_combinations(feature_indices):
        feature_combinations = [None] * len(feature_indices)
        for i, indices in enumerate(feature_indices):
            # TODO PK v.2 remove n_attributes (this is the only place it is used, but it's public)
            # TODO PK v.2 rename all instances of "attributes" -> "features"
            feature_combination = {"n_attributes": len(indices), "attributes": indices}
            feature_combinations[i] = feature_combination
        return feature_combinations

    @staticmethod
    def scores_by_feature_combination(
        X, feature_combinations, model, skip_feature_combination_idxs=[]
    ):
        for set_idx, feature_combination in enumerate(feature_combinations):
            if set_idx in skip_feature_combination_idxs:
                continue
            tensor = model[set_idx]

            # Get the current column(s) to process
            feature_idxs = feature_combination["attributes"]
            feature_idxs = list(reversed(feature_idxs))
            sliced_X = X[feature_idxs, :]
            scores = tensor[tuple(sliced_X)]

            yield set_idx, feature_combination, scores

    @staticmethod
    def decision_function(
        X, feature_combinations, model, intercept, skip_feature_combination_idxs=[]
    ):
        if X.ndim == 1:
            X = X.reshape(X.shape[0], 1)

        # Initialize empty vector for predictions
        if isinstance(intercept, numbers.Number) or len(intercept) == 1:
            score_vector = np.empty(X.shape[1])
        else:
            score_vector = np.empty((X.shape[1], len(intercept)))

        np.copyto(score_vector, intercept)

        scores_gen = EBMUtils.scores_by_feature_combination(
            X, feature_combinations, model, skip_feature_combination_idxs
        )
        for _, _, scores in scores_gen:
            score_vector += scores

        if not np.all(np.isfinite(score_vector)):  # pragma: no cover
            msg = "Non-finite values present in log odds vector."
            log.error(msg)
            raise Exception(msg)

        return score_vector

    @staticmethod
    def classifier_predict_proba(X, feature_combinations, model, intercept, skip_feature_combination_idxs=[]):
        log_odds_vector = EBMUtils.decision_function(
            X,
            feature_combinations,
            model,
            intercept,
            skip_feature_combination_idxs,
        )

        # Handle binary classification case -- softmax only works with 0s appended
        if log_odds_vector.ndim == 1:
            log_odds_vector = np.c_[np.zeros(log_odds_vector.shape), log_odds_vector]

        return softmax(log_odds_vector)

    @staticmethod
    def classifier_predict(X, feature_combinations, model, intercept, classes):
        log_odds_vector = EBMUtils.decision_function(
            X,
            feature_combinations,
            model,
            intercept
        )
        if log_odds_vector.ndim == 1:
            log_odds_vector = np.c_[np.zeros(log_odds_vector.shape), log_odds_vector]

        return classes[np.argmax(log_odds_vector, axis=1)]

    @staticmethod
    def regressor_predict(X, feature_combinations, model, intercept, skip_feature_combination_idxs=[]):
        scores = EBMUtils.decision_function(
            X,
            feature_combinations,
            model,
            intercept,
            skip_feature_combination_idxs,
        )
        return scores

    @staticmethod
    def gen_feature_name(feature_idxs, col_names):
        feature_name = []
        for feature_index in feature_idxs:
            col_name = col_names[feature_index]
            feature_name.append("feature_" + str(col_name) if isinstance(col_name, int) else str(col_name))
        feature_name = " x ".join(feature_name)
        return feature_name

    @staticmethod
    def gen_feature_type(feature_idxs, col_types):
        if len(feature_idxs) == 1:
            return col_types[feature_idxs[0]]
        else:
            # TODO PK we should consider changing the feature type to the same " x " separator
            # style as gen_feature_name, for human understanability
            return "pairwise"
