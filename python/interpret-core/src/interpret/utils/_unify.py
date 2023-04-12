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

from ._binning import unify_columns, unify_feature_names, clean_dimensions

_none_list = [None]

try:
    import pandas as pd

    _pandas_installed = True
except ImportError:
    _pandas_installed = False


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


def determine_classes(model, data, n_samples):
    if n_samples == 0:
        msg = "data cannot have 0 samples"
        _log.error(msg)
        raise ValueError(msg)

    classes = None
    if is_classifier(model):
        classes = model.classes_
        model = model.predict_proba
        preds = clean_dimensions(model(data), "model")
        if n_samples == 1:  # then the sample dimension would have been eliminated
            if preds.ndim != 1:
                msg = f"model.predict_proba(data) returned inconsistent number of dimensions"
                _log.error(msg)
                raise ValueError(msg)
            n_classes = preds.shape[0]
        else:
            if preds.shape[0] == 0:
                # we have at least 2 samples, so this means classes was an empty dimension
                n_classes = 0
            elif preds.shape[0] != n_samples:
                msg = f"model.predict_proba(data) returned inconsistent number of samples compared to data"
                _log.error(msg)
                raise ValueError(msg)
            elif preds.ndim == 1:
                # we have at least 2 samples, so the one dimension must be for samples, and the other dimension must have been 1 class (mono-classification)
                n_classes = 1
            else:
                n_classes = preds.shape[1]
        if n_classes != len(classes):
            msg = "class number mismatch"
            _log.error(msg)
            raise ValueError(msg)
    elif is_regressor(model):
        n_classes = -1
        model = model.predict
        preds = clean_dimensions(model(data), "model")
        if preds.ndim != 1:
            msg = f"model.predict(data) must have only 1 dimension"
            _log.error(msg)
            raise ValueError(msg)
        elif preds.shape[0] != n_samples:
            msg = f"model.predict(data) returned inconsistent number of samples compared to data"
            _log.error(msg)
            raise ValueError(msg)
    else:
        preds = clean_dimensions(model(data), "model")
        if n_samples == 1:  # then the sample dimension would have been eliminated
            if preds.ndim != 1:
                msg = f"model(data) has an inconsistent number of samples compared to data"
                _log.error(msg)
                raise ValueError(msg)
            elif preds.shape[0] != 1:
                # regression is always 1, so it's probabilities, and therefore classification
                n_classes = preds.shape[0]
            else:
                # it could be mono-classification, but that's unlikely, so it's regression
                n_classes = -1
        else:
            if preds.shape[0] == 0:
                # we have at least 2 samples, so this means classes was an empty dimension
                n_classes = 0
            elif preds.shape[0] != n_samples:
                msg = f"model(data) has an inconsistent number of samples compared to data"
                _log.error(msg)
                raise ValueError(msg)
            elif preds.ndim == 1:
                # we have at least 2 samples, so the first dimension must be for samples, and the second held 1 value.
                # it could be mono-classification, but that's unlikely, so it's regression
                n_classes = -1
            else:
                # we see a non-1 number of items, so it's probabilities, and therefore classification
                n_classes = preds.shape[1]

    # at this point model has been converted to a predict_fn
    return model, n_classes, classes


def unify_predict_fn(predict_fn, X, class_idx):
    if _pandas_installed and isinstance(X, pd.DataFrame):
        # scikit-learn now wants a Pandas dataframe if the model was trained on a Pandas dataframe,
        # so convert it
        names = X.columns
        if 0 <= class_idx:
            # classification
            def new_predict_fn(x):
                # TODO: at some point we should also handle column position remapping when the column names match
                X_fill = pd.DataFrame(x, columns=names)
                return predict_fn(X_fill)[:, class_idx]

            return new_predict_fn
        else:
            # regression
            def new_predict_fn(x):
                X_fill = pd.DataFrame(x, columns=names)
                return predict_fn(X_fill)

            return new_predict_fn
    else:
        if 0 <= class_idx:
            # classification
            return lambda x: predict_fn(x)[:, class_idx]  # noqa: E731
        else:
            # regression
            return predict_fn
