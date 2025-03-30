# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

import logging
from itertools import repeat, count

import numpy as np

from ._clean_x import unify_columns, unify_feature_names, categorical_encode

_log = logging.getLogger(__name__)

_none_list = [None]
_none_ndarray = np.array(None)


def unify_data(
    X,
    n_samples,
    feature_names=None,
    feature_types=None,
    missing_data_allowed=False,
    unseen_data_allowed=False,
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

    # fill with np.nan for missing values and None for unseen values
    X_unified = np.empty((n_samples, len(feature_names_in)), np.object_, "F")

    get_col = unify_columns(
        X,
        n_samples,
        feature_names_in,
        feature_types,
        min_unique_continuous,
        True,
        False,
    )
    for feature_idx in range(len(feature_names_in)):
        feature_type_in, nonmissings, uniques, X_col, bad = get_col(feature_idx)

        feature_types_in[feature_idx] = feature_type_in
        if X_col is None:
            # feature_type is "ignore"

            if not missing_data_allowed:
                msg = "X cannot contain missing values"
                _log.error(msg)
                raise ValueError(msg)

            X_unified[:, feature_idx] = np.nan
        elif uniques is None:
            # continuous feature

            if not missing_data_allowed and np.isnan(X_col).any():
                msg = "X cannot contain missing values"
                _log.error(msg)
                raise ValueError(msg)

            if bad is not None:
                if not unseen_data_allowed:
                    msg = f"Feature {feature_names_in[feature_idx]} is indicated as continuous, but has non-numeric data"
                    _log.error(msg)
                    raise ValueError(msg)
                X_col[bad != _none_ndarray] = None

            X_unified[:, feature_idx] = X_col
        else:
            # categorical feature

            categories = dict(zip(uniques, count(1)))

            X_col, bad = categorical_encode(uniques, X_col, nonmissings, categories)

            if not missing_data_allowed and np.count_nonzero(X_col) != n_samples:
                msg = "X cannot contain missing values"
                _log.error(msg)
                raise ValueError(msg)

            mapping = np.empty(len(categories) + 1, np.object_)
            mapping[0] = np.nan
            for category, idx in categories.items():
                mapping[idx] = category

            X_col = mapping[X_col]

            if bad is not None:
                if not unseen_data_allowed:
                    msg = f"Feature {feature_names_in[feature_idx]} has unrecognized ordinal values"
                    _log.error(msg)
                    raise ValueError(msg)
                X_col[bad != _none_ndarray] = None

            X_unified[:, feature_idx] = X_col

    return X_unified, feature_names_in, feature_types_in
