# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

import pandas as pd
import numpy as np
from ..utils import unify_data
from joblib import Parallel, delayed
from sklearn.utils.extmath import softmax
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin


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


def _fit(estimator, X, y, random_state, holdout_split, early_stopping_rounds,
         feature_names, feature_types):
    X, y, feature_names, feature_types = unify_data(
        X, y, feature_names, feature_types
    )
    X_cat = _convert_to_category(X, feature_names, feature_types)

    X_train, X_val, y_train, y_val = train_test_split(
        X_cat, y,
        random_state=random_state,
        test_size=holdout_split,
    )

    estimator.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  early_stopping_rounds=early_stopping_rounds,
                  categorical_feature='auto', verbose=False)

    return estimator


class LightGAMRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, feature_names=None, feature_types=None,
                 holdout_split=0.15, early_stopping_rounds=50,
                 max_depth=1, min_child_samples=2, num_leaves=3, max_rounds=5000,
                 learning_rate=0.01, colsample_bytree=0.00001, outer_bags=16,
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

            self.bagged_estimators_ = Parallel(n_jobs=self.n_jobs)(
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

            self.bagged_estimators_ = [
                _fit(learner, X, y, self.random_state, self.holdout_split, self.early_stopping_rounds,
                     self.feature_names, self.feature_types)]
            return self

    # TODO: Introduce a merge operator to build GAM graphs

    def predict(self, X):
        X_cat = _convert_to_category(X, self.feature_names, self.feature_types)
        # Should be predicting from consolidated GAM, not many bagged estimators
        return np.mean([i.predict(X_cat) for i in self.bagged_estimators_], axis=0)


class LightGAMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, feature_names=None, feature_types=None,
                 holdout_split=0.15, early_stopping_rounds=50,
                 max_depth=1, min_child_samples=2, num_leaves=3, max_rounds=5000,
                 learning_rate=0.01, colsample_bytree=0.00001, outer_bags=16,
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

        self.classes_, y = np.unique(y, return_inverse=True)

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
            return self

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
            return self

    def predict_proba(self, X):
        X_cat = _convert_to_category(X, self.feature_names, self.feature_types)
        # TODO: Verify this can work for alternative loss functions
        raw_scores_vector = np.mean([i.predict_proba(X_cat, raw_score=True) for i in self.estimators_], axis=0)

        if raw_scores_vector.ndim == 1:
            raw_scores_vector = np.c_[np.zeros(raw_scores_vector.shape), raw_scores_vector]

        return softmax(raw_scores_vector)

    def predict(self, X):
        X_cat = _convert_to_category(X, self.feature_names, self.feature_types)
        preds = self.predict_proba(X_cat)
        return self.classes_[np.argmax(preds, axis=1)]
