# Copyright (c) 2019 Microsoft Corporation

# Distributed under the MIT software license
# TODO: Test EBMUtils

# from scipy.special import expit
from sklearn.utils.extmath import softmax
from sklearn.model_selection import train_test_split
import numbers
import numpy as np
import warnings


import logging

log = logging.getLogger(__name__)


# TODO: Clean up
class EBMUtils:
    @staticmethod
    def convert_to_intervals(cuts):
        cuts = np.array(cuts, dtype=np.float64)

        if np.isnan(cuts).any():
            raise Exception("cuts cannot contain nan")

        if np.isinf(cuts).any():
            raise Exception("cuts cannot contain infinity")

        smaller = np.insert(cuts, 0, -np.inf)
        larger = np.append(cuts, np.inf)
        intervals = list(zip(smaller, larger))

        if any(x[1] <= x[0] for x in intervals):
            raise Exception("cuts must contain increasing values")

        return intervals

    @staticmethod
    def convert_to_cuts(intervals):
        if len(intervals) == 0:
            raise Exception("intervals must have at least one interval")

        if any(len(x) != 2 for x in intervals):
            raise Exception("intervals must be a list of tuples")

        if intervals[0][0] != -np.inf:
            raise Exception("intervals must start from -inf")

        if intervals[-1][-1] != np.inf:
            raise Exception("intervals must end with inf")

        cuts = [x[0] for x in intervals[1:]]
        cuts_verify = [x[1] for x in intervals[:-1]]

        if np.isnan(cuts).any():
            raise Exception("intervals cannot contain NaN")

        if any(x[0] != x[1] for x in zip(cuts, cuts_verify)):
            raise Exception("intervals must contain adjacent sections")

        if any(higher <= lower for lower, higher in zip(cuts, cuts[1:])):
            raise Exception("intervals must contain increasing sections")

        return cuts

    @staticmethod
    def get_count_scores_c(n_classes):
        # this should reflect how the C code represents scores
        return 1 if n_classes <= 2 else n_classes

    @staticmethod
    def ebm_train_test_split(
        X, y, test_size, random_state, is_classification, is_train=True
    ):
        # all test/train splits should be done with this function to ensure that
        # if we re-generate the train/test splits that they are generated exactly
        # the same as before
        if test_size == 0:
            X_train, y_train = X, y
            X_val = np.empty(shape=(0, X.shape[1]), dtype=X.dtype)
            y_val = np.empty(shape=(0,), dtype=y.dtype)
        elif test_size > 0:
            # Adapt test size if too small relative to number of classes
            if is_classification:
                y_uniq = len(set(y))
                n_test_samples = test_size if test_size >= 1 else len(y) * test_size
                if n_test_samples < y_uniq:  # pragma: no cover
                    warnings.warn(
                        "Too few samples per class, adapting test size to guarantee 1 sample per class."
                    )
                    test_size = y_uniq

            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=y if is_classification else None,
            )
        else:  # pragma: no cover
            raise Exception("test_size must be between 0 and 1.")

        if not is_train:
            X_train, y_train = None, None

        # TODO PK doing a fortran re-ordering here (and an extra copy) isn't the most efficient way
        #         push the re-ordering right to our first call to fit(..) AND stripe convert
        #         groups of rows at once and they process them in fortran order after that
        # change to Fortran ordering on our data, which is more efficient in terms of memory accesses
        # AND our C code expects it in that ordering
        if X_train is not None:
            X_train = np.ascontiguousarray(X_train.T)
        X_val = np.ascontiguousarray(X_val.T)

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
    def scores_by_feature_combination(X, feature_combinations, model):
        for set_idx, feature_combination in enumerate(feature_combinations):
            tensor = model[set_idx]

            # Get the current column(s) to process
            feature_idxs = feature_combination
            sliced_X = X[feature_idxs, :]
            scores = tensor[tuple(sliced_X)]

            yield set_idx, feature_combination, scores

    @staticmethod
    def decision_function(X, feature_combinations, model, intercept):
        if X.ndim == 1:
            X = X.reshape(X.shape[0], 1)

        # Initialize empty vector for predictions
        if isinstance(intercept, numbers.Number) or len(intercept) == 1:
            score_vector = np.empty(X.shape[1])
        else:
            score_vector = np.empty((X.shape[1], len(intercept)))

        np.copyto(score_vector, intercept)

        scores_gen = EBMUtils.scores_by_feature_combination(
            X, feature_combinations, model
        )
        for _, _, scores in scores_gen:
            score_vector += scores

        if not np.all(np.isfinite(score_vector)):  # pragma: no cover
            msg = "Non-finite values present in log odds vector."
            log.error(msg)
            raise Exception(msg)

        return score_vector

    @staticmethod
    def classifier_predict_proba(X, feature_combinations, model, intercept):
        log_odds_vector = EBMUtils.decision_function(
            X, feature_combinations, model, intercept
        )

        # Handle binary classification case -- softmax only works with 0s appended
        if log_odds_vector.ndim == 1:
            log_odds_vector = np.c_[np.zeros(log_odds_vector.shape), log_odds_vector]

        return softmax(log_odds_vector)

    @staticmethod
    def classifier_predict(X, feature_combinations, model, intercept, classes):
        log_odds_vector = EBMUtils.decision_function(
            X, feature_combinations, model, intercept
        )
        if log_odds_vector.ndim == 1:
            log_odds_vector = np.c_[np.zeros(log_odds_vector.shape), log_odds_vector]

        return classes[np.argmax(log_odds_vector, axis=1)]

    @staticmethod
    def regressor_predict(X, feature_combinations, model, intercept):
        scores = EBMUtils.decision_function(X, feature_combinations, model, intercept)
        return scores

    @staticmethod
    def gen_feature_name(feature_idxs, col_names):
        feature_name = []
        for feature_index in feature_idxs:
            col_name = col_names[feature_index]
            feature_name.append(
                "feature_" + str(col_name)
                if isinstance(col_name, int)
                else str(col_name)
            )
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
