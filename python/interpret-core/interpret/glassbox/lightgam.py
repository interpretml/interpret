# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

import numbers
import pandas as pd
import numpy as np
from ..utils import unify_data
from joblib import Parallel, delayed
from sklearn.utils.extmath import softmax
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from .ebm.ebm import EBMPreprocessor, EBMExplanation

from ..utils import gen_name_from_class, gen_global_selector, gen_local_selector, gen_perf_dicts
from sklearn.base import is_classifier
from sklearn.utils.validation import check_is_fitted


def _convert_to_category(X, feature_names, feature_types):
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    cat_cols = [
        feature_names[index] for index, ftype in enumerate(feature_types)
        if ftype == "categorical"
    ]

    # Convert to categoricals
    if len(cat_cols) > 0:
        for col in cat_cols:
            X[col] = pd.Categorical(X[col])
    return X


# TODO: Clean up function signature.
def _fit(estimator, X, y, random_state, holdout_split, early_stopping_rounds,
         feature_names, feature_types):
    # X_new = _convert_to_category(X_new, feature_names, feature_types)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        random_state=random_state,
        test_size=holdout_split,
    )
    cat_cols = [
        index for index, ftype in enumerate(feature_types)
        if ftype == "categorical"
    ]
    estimator.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  early_stopping_rounds=early_stopping_rounds,
                  # categorical_feature=cat_cols,
                  verbose=False
                  )

    return estimator


class LightGAMRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, feature_names=None, feature_types=None,
                 holdout_split=0.15, early_stopping_rounds=50,
                 max_depth=1, min_child_samples=2, num_leaves=3, max_rounds=5000,
                 learning_rate=0.01, colsample_bytree=0.00001, outer_bags=2,
                 n_jobs=-1, random_state=1):

        self.feature_names = feature_names
        self.feature_types = feature_types
        self.holdout_split = holdout_split
        self.early_stopping_rounds = early_stopping_rounds
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.num_leaves = num_leaves
        self.max_rounds = max_rounds
        self.learning_rate = learning_rate
        self.colsample_bytree = colsample_bytree
        self.outer_bags = outer_bags
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y):
        from lightgbm import LGBMRegressor

        X_orig, y, self.feature_names, self.feature_types = unify_data(
            X, y, self.feature_names, self.feature_types
        )
        self.preprocessor_ = EBMPreprocessor(
            feature_names=self.feature_names,
            feature_types=self.feature_types,
            random_state=1337,  # TODO: Arg
            binning="quantile",
        )
        self.preprocessor_.fit(X_orig)
        X = self.preprocessor_.transform(X_orig)

        if self.outer_bags > 1:
            estimators = []
            for bag in range(self.outer_bags):
                learner = LGBMRegressor(max_depth=self.max_depth,
                                        n_estimators=self.max_rounds * X.shape[1],
                                        learning_rate=self.learning_rate,
                                        colsample_bytree=self.colsample_bytree,
                                        n_jobs=1,
                                        random_state=self.random_state + bag,
                                        verbose=-1)
                estimators.append(learner)

            self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit)(estimator, X, y, self.random_state + i, self.holdout_split, self.early_stopping_rounds,
                              self.feature_names, self.feature_types)
                for i, estimator in enumerate(estimators)
            )
            return self

        else:
            learner = LGBMRegressor(max_depth=self.max_depth,
                                    n_estimators=self.max_rounds * X.shape[1],
                                    learning_rate=self.learning_rate,
                                    colsample_bytree=self.colsample_bytree,
                                    n_jobs=self.n_jobs,
                                    random_state=self.random_state,
                                    verbose=-1)

            self.estimators_ = [
                _fit(learner, X, y, self.random_state, self.holdout_split, self.early_stopping_rounds,
                     self.feature_names, self.feature_types)]
            return self

    def predict(self, X):
        # X_cat = _convert_to_category(X, self.feature_names, self.feature_types)
        # Should be predicting from consolidated GAM, not many bagged estimators
        X_orig, _, _, _ = unify_data(X, None, self.feature_names, self.feature_types)
        X_cat = self.preprocessor_.transform(X_orig)
        return np.mean([i.predict(X_cat) for i in self.estimators_], axis=0)


class LightGAMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, feature_names=None, feature_types=None,
                 holdout_split=0.15, early_stopping_rounds=50,
                 max_depth=1, min_child_samples=2, num_leaves=3, max_rounds=5000,
                 learning_rate=0.01, colsample_bytree=0.00001, outer_bags=2,
                 n_jobs=-1, random_state=1):

        self.feature_names = feature_names
        self.feature_types = feature_types
        self.holdout_split = holdout_split
        self.early_stopping_rounds = early_stopping_rounds
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.num_leaves = num_leaves
        self.max_rounds = max_rounds
        self.learning_rate = learning_rate
        self.colsample_bytree = colsample_bytree
        self.outer_bags = outer_bags
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y):
        from lightgbm import LGBMClassifier

        X_orig, y, self.feature_names, self.feature_types = unify_data(
            X, y, self.feature_names, self.feature_types
        )
        self.preprocessor_ = EBMPreprocessor(
            feature_names=self.feature_names,
            feature_types=self.feature_types,
            random_state=1337,  # TODO: Arg
            binning="quantile",
        )
        self.preprocessor_.fit(X_orig)
        X = self.preprocessor_.transform(X_orig)

        self.classes_, y = np.unique(y, return_inverse=True)
        self._class_idx_ = {x: index for index, x in enumerate(self.classes_)}

        if self.outer_bags > 1:
            estimators = []
            for bag in range(self.outer_bags):
                learner = LGBMClassifier(max_depth=self.max_depth,
                                         n_estimators=self.max_rounds * X.shape[1],
                                         learning_rate=self.learning_rate,
                                         colsample_bytree=self.colsample_bytree,
                                         n_jobs=1,
                                         random_state=self.random_state + bag,
                                         verbose=-1)
                estimators.append(learner)

            self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit)(estimator, X, y, self.random_state + i, self.holdout_split, self.early_stopping_rounds,
                              self.feature_names, self.feature_types)
                for i, estimator in enumerate(estimators)
            )
        else:
            learner = LGBMClassifier(max_depth=self.max_depth,
                                     n_estimators=self.max_rounds * X.shape[1],
                                     learning_rate=self.learning_rate,
                                     colsample_bytree=self.colsample_bytree,
                                     n_jobs=self.n_jobs,
                                     random_state=self.random_state,
                                     verbose=-1)

            self.estimators_ = [
                _fit(learner, X, y, self.random_state, self.holdout_split, self.early_stopping_rounds,
                     self.feature_names, self.feature_types)]

        # TODO: Introduce a merge operator to build GAM graphs
        self.additive_terms_ = []
        self.term_standard_deviations_ = []

        baselines = []
        for estimator in self.estimators_:
            X_synthetic = np.zeros(len(self.feature_names))
            baseline = estimator.predict(X_synthetic.reshape(1, -1), raw_score=True)
            baselines.append(baseline)
        self.intercept_ = np.average(baselines, axis=0)

        for index, _ in enumerate(self.feature_names):
            term_contribs = []
            for estimator in self.estimators_:
                X_synthetic = np.zeros(len(self.feature_names))
                # X_synthetic = X[0]
                # X_baseline = self.preprocessor_.transform(X_synthetic)
                baseline = estimator.predict(X_synthetic.reshape(1, -1), raw_score=True)

                bin_labels = self.preprocessor_.get_bin_labels(index)
                term_contrib = np.zeros(len(bin_labels))
                for bin_index, bin_label in enumerate(self.preprocessor_.get_bin_labels(index)):
                    X_synthetic[index] = bin_index
                    # X_transformed = self.preprocessor_.transform(X_synthetic)
                    X_transformed = estimator.predict(X_synthetic.reshape(1, -1), raw_score=True)
                    term_contrib[bin_index] = X_transformed - baseline

                term_contribs.append(term_contrib)

            averaged_model = np.average(np.array(term_contribs), axis=0)
            model_errors = np.std(np.array(term_contribs), axis=0)

            self.additive_terms_.append(averaged_model)
            self.term_standard_deviations_.append(model_errors)

        # TODO: Model centering

        # Generate overall importance
        if X.ndim == 1:
            X = X.reshape(1, -1)
        intercept = self.intercept_
        if isinstance(intercept, numbers.Number) or len(intercept) == 1:
            score_vector = np.empty(X.shape[0])
        else:
            score_vector = np.empty((X.shape[0], len(intercept)))
        np.copyto(score_vector, intercept)
        self.feature_importances_ = []
        model = self.additive_terms_
        for feature_index, _ in enumerate(self.feature_names):
            tensor = model[feature_index]
            sliced_X = X[:, feature_index]
            unknowns = (sliced_X < 0)
            sliced_X[unknowns] = 0
            scores = tensor[sliced_X]
            scores[unknowns] = 0
            score_vector += scores
            mean_abs_score = np.mean(np.abs(scores))
            self.feature_importances_.append(mean_abs_score)
        # Generate selector
        self.global_selector = gen_global_selector(
            X_orig, self.feature_names, self.feature_types, None
        )

        return self

    # TODO: Remove this.
    def _old_decision_function(self, X):
        # X_cat = _convert_to_category(X, self.feature_names, self.feature_types)
        X_orig, _, _, _ = unify_data(X, None, self.feature_names, self.feature_types)
        X_cat = self.preprocessor_.transform(X_orig)
        # TODO: Verify this can work for alternative loss functions
        raw_scores_vector = np.mean([i.predict_proba(X_cat, raw_score=True) for i in self.estimators_], axis=0)

        return raw_scores_vector

    def predict_proba(self, X):
        raw_scores_vector = self.decision_function(X)
        if raw_scores_vector.ndim == 1:
            raw_scores_vector = np.c_[np.zeros(raw_scores_vector.shape), raw_scores_vector]
        return softmax(raw_scores_vector)

    def decision_function(self, X):
        X_orig, _, _, _ = unify_data(X, None, self.feature_names, self.feature_types)
        X = self.preprocessor_.transform(X_orig)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        intercept = self.intercept_
        if isinstance(intercept, numbers.Number) or len(intercept) == 1:
            score_vector = np.empty(X.shape[0])
        else:
            score_vector = np.empty((X.shape[0], len(intercept)))
        np.copyto(score_vector, intercept)

        model = self.additive_terms_
        for feature_index, _ in enumerate(self.feature_names):
            tensor = model[feature_index]
            sliced_X = X[:, feature_index]
            unknowns = (sliced_X < 0)
            sliced_X[unknowns] = 0
            scores = tensor[sliced_X]
            scores[unknowns] = 0
            score_vector += scores

        return score_vector

        # if score_vector.ndim == 1:
        #     score_vector = np.c_[np.zeros(score_vector.shape), score_vector]
        #
        # return softmax(score_vector)

    def predict(self, X):
        X_orig, _, _, _ = unify_data(X, None, self.feature_names, self.feature_types)
        X_cat = self.preprocessor_.transform(X_orig)
        # X_cat = _convert_to_category(X, self.feature_names, self.feature_types)
        preds = self.predict_proba(X_cat)
        return self.classes_[np.argmax(preds, axis=1)]

    def explain_local(self, X, y=None, name=None):
        """ Provides local explanations for provided samples.

        Args:
            X: Numpy array for X to explain.
            y: Numpy vector for y to explain.
            name: User-defined explanation name.

        Returns:
            An explanation object, visualizing feature-value pairs
            for each sample as horizontal bar charts.
        """

        # Produce feature value pairs for each sample.
        # Values are the model graph score per respective feature group.
        if name is None:
            name = gen_name_from_class(self)

        X_orig, y, _, _ = unify_data(X, y, self.feature_names, self.feature_types)
        X = self.preprocessor_.transform(X_orig)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Transform y if classifier
        if is_classifier(self) and y is not None:
            y = np.array([self._class_idx_[el] for el in y])

        n_rows = X.shape[0]
        data_dicts = []
        intercept = self.intercept_
        if not is_classifier(self) or len(self.classes_) <= 2:
            if isinstance(self.intercept_, np.ndarray) or isinstance(
                    self.intercept_, list
            ):
                intercept = intercept[0]

        for _ in range(n_rows):
            data_dict = {
                "type": "univariate",
                "names": [],
                "scores": [],
                "values": [],
                "extra": {"names": ["Intercept"], "scores": [intercept], "values": [1]},
            }
            if is_classifier(self):
                data_dict["meta"] = {
                    "label_names": self.classes_.tolist()  # Classes should be numpy array, convert to list.
                }
            data_dicts.append(data_dict)

        model = self.additive_terms_
        for feature_index, _ in enumerate(self.feature_names):
            tensor = model[feature_index]
            sliced_X = X[:, feature_index]
            unknowns = (sliced_X < 0)
            sliced_X[unknowns] = 0
            scores = tensor[sliced_X]
            scores[unknowns] = 0

            for row_idx in range(n_rows):
                feature_name = self.feature_names[feature_index]
                data_dicts[row_idx]["names"].append(feature_name)
                data_dicts[row_idx]["scores"].append(scores[row_idx])
                data_dicts[row_idx]["values"].append("")

        is_classification = is_classifier(self)
        if is_classification:
            scores = self.predict_proba(X_orig)
        else:
            scores = self.predict(X_orig)

        perf_list = []
        perf_dicts = gen_perf_dicts(scores, y, is_classification)
        for row_idx in range(n_rows):
            perf = None if perf_dicts is None else perf_dicts[row_idx]
            perf_list.append(perf)
            data_dicts[row_idx]["perf"] = perf

        selector = gen_local_selector(data_dicts, is_classification=is_classification)
        internal_obj = {
            "overall": None,
            "specific": data_dicts,
            "mli": [
                {
                    "explanation_type": "ebm_local",
                    "value": {
                        "scores": self.additive_terms_,
                        "intercept": self.intercept_,
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

        return EBMExplanation(
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

        # Obtain min/max for model scores
        lower_bound = np.inf
        upper_bound = -np.inf
        for feature_name_index, _ in enumerate(self.feature_names):
            errors = self.term_standard_deviations_[feature_name_index]
            scores = self.additive_terms_[feature_name_index]

            lower_bound = min(lower_bound, np.min(scores - errors))
            upper_bound = max(upper_bound, np.max(scores + errors))

        bounds = (lower_bound, upper_bound)

        # Add per feature graph
        data_dicts = []
        feature_list = []
        density_list = []
        for feature_index, feature_name in enumerate(
                self.feature_names
        ):
            model_graph = self.additive_terms_[feature_index]

            # NOTE: This uses stddev. for bounds, consider issue warnings.
            errors = self.term_standard_deviations_[feature_index]

            bin_labels = self.preprocessor_.get_bin_labels(feature_index)
            # bin_counts = self.preprocessor_.get_bin_counts(
            #     feature_indexes[0]
            # )
            scores = list(model_graph)
            upper_bounds = list(model_graph + errors)
            lower_bounds = list(model_graph - errors)
            density_dict = {
                "names": self.preprocessor_.get_hist_edges(feature_index),
                "scores": self.preprocessor_.get_hist_counts(feature_index),
            }

            feature_dict = {
                "type": "univariate",
                "names": bin_labels,
                "scores": scores,
                "scores_range": bounds,
                "upper_bounds": upper_bounds,
                "lower_bounds": lower_bounds,
            }
            feature_list.append(feature_dict)
            density_list.append(density_dict)

            data_dict = {
                "type": "univariate",
                "names": bin_labels,
                "scores": model_graph,
                "scores_range": bounds,
                "upper_bounds": model_graph + errors,
                "lower_bounds": model_graph - errors,
                "density": {
                    "names": self.preprocessor_.get_hist_edges(feature_index),
                    "scores": self.preprocessor_.get_hist_counts(
                        feature_index
                    ),
                },
            }
            if is_classifier(self):
                data_dict["meta"] = {
                    "label_names": self.classes_.tolist()  # Classes should be numpy array, convert to list.
                }

            data_dicts.append(data_dict)

        overall_dict = {
            "type": "univariate",
            "names": self.feature_names,
            "scores": self.feature_importances_,
        }
        internal_obj = {
            "overall": overall_dict,
            "specific": data_dicts,
            "mli": [
                {
                    "explanation_type": "ebm_global",
                    "value": {"feature_list": feature_list},
                },
                {"explanation_type": "density", "value": {"density": density_list}},
            ],
        }

        return EBMExplanation(
            "global",
            internal_obj,
            feature_names=self.feature_names,
            feature_types=self.feature_types,
            name=name,
            selector=self.global_selector,
        )
