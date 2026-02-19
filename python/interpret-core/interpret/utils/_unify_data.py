# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

import logging
from itertools import repeat, count

import numpy as np

from ._clean_x import (
    unify_columns_nonschematized,
    unify_columns_schematized,
    unify_feature_names,
    categorical_encode,
)

_log = logging.getLogger(__name__)

_none_list = [None]


def unify_data(
    X,
    n_samples,
    feature_names=None,
    feature_types=None,
    missing_data_allowed=False,
    unseen_data_allowed=False,
    min_unique_continuous=0,
    is_schematized=False,
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

    if is_schematized:
        get_col = unify_columns_schematized(
            X,
            n_samples,
            feature_names_in,
            feature_types,
            min_unique_continuous,
            False,
        )
    else:
        get_col = unify_columns_nonschematized(
            X,
            n_samples,
            feature_names_in,
            feature_types,
            min_unique_continuous,
            False,
        )

    for feature_idx in range(len(feature_names_in)):
        if feature_types is not None and feature_types[feature_idx] == "ignore":
            # TODO: we should drop these columns instead of passing them to the dependent model
            # since many models cannot handle missing values.

            if not missing_data_allowed:
                msg = "X cannot contain missing values"
                _log.error(msg)
                raise ValueError(msg)

            X_unified[:, feature_idx] = np.nan
            feature_types_in[feature_idx] = "ignore"
        else:
            feature_types_in[feature_idx], nonmissings, uniques, X_col, bad = get_col(
                feature_idx
            )
            if uniques is None:
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
                    X_col[bad] = None  # use None for unseen. np.nan is for missing

                X_unified[:, feature_idx] = X_col
            else:
                # categorical feature

                categories = dict(zip(uniques, count(1)))

                X_col = categorical_encode(uniques, X_col, nonmissings, categories)

                if not missing_data_allowed and np.count_nonzero(X_col) != n_samples:
                    msg = "X cannot contain missing values"
                    _log.error(msg)
                    raise ValueError(msg)

                if not unseen_data_allowed and (X_col == -1).any():
                    msg = f"Feature {feature_names_in[feature_idx]} has unrecognized ordinal values"
                    _log.error(msg)
                    raise ValueError(msg)

                mapping = np.empty(len(categories) + 2, np.object_)
                mapping[0] = np.nan  # use np.nan for missing
                mapping[-1] = None  # use None for unseen
                for category, idx in categories.items():
                    mapping[idx] = category

                X_unified[:, feature_idx] = mapping[X_col]

    return (
        X_unified,
        feature_names_in,
        feature_types if is_schematized else feature_types_in,
    )
