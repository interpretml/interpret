# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

import math
from collections import Counter
from itertools import count, repeat, groupby
from warnings import warn
import numpy as np
import numpy.ma as ma
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
)
from sklearn.utils.validation import check_is_fitted
from sklearn.base import is_classifier, is_regressor

import logging

_log = logging.getLogger(__name__)

from ._clean_x import unify_columns, unify_feature_names

_none_list = [None]


def unify_data(
    X,
    n_samples,
    feature_names=None,
    feature_types=None,
    missing_data_allowed=False,
    min_unique_continuous=0,
):
    _log.info("Unifying data")

    if n_samples == 0:
        # for some callers this might be legal if they've been fitted before, but we'll let the caller
        # decide how they want to handle this condition
        msg = "X has 0 samples"
        _log.error(msg)
        raise ValueError(msg)

    # if feature_names_in and feature_types_in were generated in a call to fit(..) then unify_feature_names
    # and unify_columns will return the identical names and types
    feature_names_in = unify_feature_names(X, feature_names, feature_types)
    feature_types_in = _none_list * len(feature_names_in)

    # TODO: this could be made more efficient by storing continuous and categorical values in separate numpy arrays
    # and merging afterwards.  Categoricals are going to share the same objects, but we don't want object
    # fragmentation for continuous values which generates a lot of garbage to collect later
    X_unified = np.empty((n_samples, len(feature_names_in)), np.object_, order="F")

    for feature_idx, (feature_type_in, X_col, categories, bad) in enumerate(
        unify_columns(
            X,
            zip(range(len(feature_names_in)), repeat(None)),
            feature_names_in,
            feature_types,
            min_unique_continuous,
            False,
        )
    ):
        if n_samples != len(X_col):
            msg = "The columns of X are mismatched in the number of of samples"
            _log.error(msg)
            raise ValueError(msg)

        feature_types_in[feature_idx] = feature_type_in
        if categories is None:
            # continuous feature
            if bad is not None:
                msg = f"Feature {feature_names_in[feature_idx]} is indicated as continuous, but has non-numeric data"
                _log.error(msg)
                raise ValueError(msg)

            if not missing_data_allowed and np.isnan(X_col).any():
                msg = f"X cannot contain missing values"
                _log.error(msg)
                raise ValueError(msg)

            X_unified[:, feature_idx] = X_col
        else:
            # categorical feature
            if bad is not None:
                msg = f"Feature {feature_names_in[feature_idx]} has unrecognized ordinal values"
                _log.error(msg)
                raise ValueError(msg)

            if not missing_data_allowed and np.count_nonzero(X_col) != len(X_col):
                msg = f"X cannot contain missing values"
                _log.error(msg)
                raise ValueError(msg)

            mapping = np.empty(len(categories) + 1, np.object_)
            mapping.itemset(0, np.nan)
            for category, idx in categories.items():
                mapping.itemset(idx, category)
            X_unified[:, feature_idx] = mapping[X_col]

    return X_unified, feature_names_in, feature_types_in
