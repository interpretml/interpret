# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

import itertools
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd

# NOTE: Old versions of pandas have these utils in a different namepsace.
try:
    from pandas.api.types import is_numeric_dtype, is_string_dtype
except ImportError:  # pragma: no cover
    from pandas.core.dtypes.common import is_numeric_dtype, is_string_dtype

from pandas.core.generic import NDFrame
from pandas.core.series import Series
import scipy as sp

import logging

log = logging.getLogger(__name__)


def gen_perf_dicts(scores, y=None, is_classification=True):
    n_dim = len(scores.shape)

    if not is_classification:
        predicted = scores
    else:
        if n_dim == 1:
            scores = np.vstack([1 - scores, scores]).T
        predicted = np.argmax(scores, axis=1)

    records = []
    for i, _ in enumerate(scores):
        di = {}
        di["is_classification"] = is_classification
        di["actual"] = np.nan if y is None else y[i]

        if is_classification:
            di["predicted"] = predicted[i]
            di["actual_score"] = np.nan if y is None else scores[i, y[i]]
            di["predicted_score"] = scores[i, predicted[i]]
        else:
            di["predicted"] = predicted[i]
            di["actual_score"] = np.nan if y is None else y[i]
            di["predicted_score"] = scores[i]

        records.append(di)

    return records


def hist_per_column(arr, feature_types=None):
    counts = []
    bin_edges = []

    if feature_types is not None:
        for i, feat_type in enumerate(feature_types):
            if feat_type == "continuous":
                count, bin_edge = np.histogram(arr[:, i], bins="doane")
                counts.append(count)
                bin_edges.append(bin_edge)
            elif feat_type == "categorical":
                # Todo: check if this call
                bin_edge, count = np.unique(arr[:, i], return_counts=True)
                counts.append(count)
                bin_edges.append(bin_edge)
    else:
        for i in range(arr.shape[1]):
            count, bin_edge = np.histogram(arr[:, i], bins="doane")
            counts.append(count)
            bin_edges.append(bin_edge)
    return counts, bin_edges


def gen_global_selector(X, feature_names, feature_types, importance_scores, round=3):
    records = []

    for feat_idx, _ in enumerate(feature_names):
        record = {}
        record["Name"] = feature_names[feat_idx]
        record["Type"] = feature_types[feat_idx]

        if feat_idx < X.shape[1]:
            col_vals = X[:, feat_idx]
            nz_count = np.count_nonzero(col_vals)
            col_vals = col_vals.astype('U')
            record["# Unique"] = len(np.unique(col_vals))
            record["% Non-zero"] = nz_count / X.shape[0]
        else:
            record["# Unique"] = np.nan
            record["% Non-zero"] = np.nan

        # if importance_scores is None:
        #     record["Importance"] = np.nan
        # else:
        #     record["Importance"] = importance_scores[feat_idx]

        records.append(record)

    # columns = ["Name", "Type", "# Unique", "% Non-zero", "Importance"]
    columns = ["Name", "Type", "# Unique", "% Non-zero"]
    df = pd.DataFrame.from_records(records, columns=columns)
    if round is not None:
        return df.round(round)
    else:  # pragma: no cover
        return df

def gen_global_selector2(n_samples, n_features, term_names, term_types, unique_val_counts, zero_val_counts, round=3):
    records = []
    for term_idx in range(len(term_names)):
        record = {}
        record["Name"] = term_names[term_idx]
        feature_type = term_types[term_idx]
        record["Type"] = 'categorical' if feature_type == 'nominal' or feature_type == 'ordinal' else feature_type

        if term_idx < n_features:
            record["# Unique"] = np.nan if unique_val_counts is None else unique_val_counts[term_idx]
            if n_samples is None or zero_val_counts is None:
                record["% Non-zero"] = np.nan
            else:
                record["% Non-zero"] = (n_samples - zero_val_counts[term_idx]) / n_samples
        else:
            record["# Unique"] = np.nan
            record["% Non-zero"] = np.nan

        records.append(record)

    # columns = ["Name", "Type", "# Unique", "% Non-zero", "Importance"]
    columns = ["Name", "Type", "# Unique", "% Non-zero"]
    df = pd.DataFrame.from_records(records, columns=columns)
    if round is not None:
        return df.round(round)
    else:  # pragma: no cover
        return df

def gen_local_selector(data_dicts, round=3, is_classification=True):
    records = []

    for data_dict in data_dicts:
        perf_dict = data_dict["perf"]
        record = {}
        record["PrScore"] = perf_dict["predicted_score"]
        record["AcScore"] = perf_dict["actual_score"]

        record["Predicted"] = perf_dict["predicted"]
        record["Actual"] = perf_dict["actual"]

        record["Resid"] = record["AcScore"] - record["PrScore"]
        record["AbsResid"] = abs(record["Resid"])

        records.append(record)

    if is_classification:
        columns = ["Actual", "Predicted", "PrScore", "AcScore", "Resid", "AbsResid"]
    else:
        columns = ["Actual", "Predicted", "Resid", "AbsResid"]

    df = pd.DataFrame.from_records(records, columns=columns)
    if round is not None:
        return df.round(round)
    else:  # pragma: no cover
        return df


def gen_name_from_class(obj):
    """ Generates a name for a given class.

    Args:
        obj: An object.

    Returns:
        A generated name as a string that uses
        class name and a static counter.
    """
    class_name = obj.__class__.__name__
    if class_name not in gen_name_from_class.cache:
        gen_name_from_class.cache[class_name] = itertools.count(0)
    identifier = next(gen_name_from_class.cache[class_name])

    return str(obj.__class__.__name__) + "_" + str(identifier)


gen_name_from_class.cache = {}


def gen_feat_val_list(features, values):
    """ Generates feature value lists sorted in descending value.

    Args:
        features: A list of feature names.
        values: A list of values.

    Returns:
        A sorted list of feature-value tuples.
    """
    sorted_feat_val_list = sorted(
        zip(features, values), key=lambda x: abs(x[1]), reverse=True  # noqa: E731
    )
    return sorted_feat_val_list


def reverse_map(map):
    """ Inverts a dictionary.

    Args:
        map: Target dictionary to invert.

    Returns:
        A dictionary where keys and values are swapped.
    """
    return dict(reversed(item) for item in map.items())


def sort_feature_value_pairs_list(feature_value_pairs_list):
    """ Sorts feature value pairs list.

    Args:
        feature_value_pairs_list: List of feature value pairs (list in itself)

    Returns:
        A Feature value pairs list sorted by score descending.
    """
    sorted_list = [
        sorted(x, key=lambda x: abs(x[1]), reverse=True)
        for x in feature_value_pairs_list
    ]
    return sorted_list


def gen_feature_names_from_df(df):
    return list(df.columns)


def unify_predict_fn(predict_fn, X):
    predictions = predict_fn(X[:1])
    if predictions.ndim == 2:
        new_predict_fn = lambda x: predict_fn(x)[:, 1]  # noqa: E731
        return new_predict_fn
    else:
        return predict_fn


def unify_vector(data):
    if data is None:
        return None

    if isinstance(data, Series):
        new_data = data.values
    elif isinstance(data, np.ndarray):
        if data.ndim > 1:
            new_data = data.ravel()
        else:
            new_data = data
    elif isinstance(data, list):
        new_data = np.array(data)
    elif isinstance(data, NDFrame) and data.shape[1] == 1:
        new_data = data.iloc[:, 0].values
    else:  # pragma: no cover
        msg = "Could not unify data of type: {0}".format(type(data))
        log.warning(msg)
        raise Exception(msg)

    return new_data


def _get_new_feature_names(data, feature_names):
    if feature_names is None:
        return [f'feature_{i:04}' for i in range(1, 1 + data.shape[1])]
    else:
        return feature_names


def _get_new_feature_types(data, feature_types, new_feature_names):
    if feature_types is None:
        unique_val_counts = np.apply_along_axis(lambda a: len(set(a)), axis=0, arr=data)
        return [
            _assign_feature_type(feature_type, unique_val_counts[index])
            for index, feature_type in enumerate([data.dtype] * len(new_feature_names))
        ]
    else:
        return feature_types


# TODO: Docs for unify_data.
def unify_data(data, labels=None, feature_names=None, feature_types=None, missing_data_allowed=False):
    """ Attempts to unify data into a numpy array with feature names and types.

    If it cannot unify, returns the original data structure.

    Args:
        data:
        labels:
        feature_names:
        feature_types:

    Returns:

    """
    # TODO: Clean up code to have less duplication.
    if isinstance(data, NDFrame):
        # NOTE: Workaround for older versions of pandas.
        try:
            new_data = data.to_numpy()
        except AttributeError:  # pragma: no cover
            new_data = data.values

        if feature_names is None:
            new_feature_names = list(data.columns)
        else:
            new_feature_names = feature_names

        if feature_types is None:
            # unique_val_counts = np.apply_along_axis(lambda a: len(set(a)), axis=0, arr=data)
            bool_indicator = [data[col].isin([np.nan, 0, 1]).all() for col in data.columns]
            new_feature_types = [
                _assign_feature_type(feature_type, bool_indicator[index])
                for index, feature_type in enumerate(data.dtypes)
            ]
        else:
            new_feature_types = feature_types
    elif isinstance(data, list):
        new_data = np.array(data)

        new_feature_names = _get_new_feature_names(new_data, feature_names)
        new_feature_types = _get_new_feature_types(
            new_data, feature_types, new_feature_names
        )
    elif isinstance(data, np.ndarray):
        new_data = data

        new_feature_names = _get_new_feature_names(data, feature_names)
        new_feature_types = _get_new_feature_types(
            data, feature_types, new_feature_names
        )
    elif sp.sparse.issparse(data):
        # Add warning message for now prior to converting the data to dense format
        warn_msg = (
            "Sparse data not fully supported, will be densified for now, may cause OOM"
        )
        warnings.warn(warn_msg, RuntimeWarning)
        new_data = data.toarray()

        new_feature_names = _get_new_feature_names(new_data, feature_names)
        new_feature_types = _get_new_feature_types(
            new_data, feature_types, new_feature_names
        )
    else:  # pragma: no cover
        msg = "Could not unify data of type: {0}".format(type(data))
        log.error(msg)
        raise ValueError(msg)

    new_labels = unify_vector(labels)

    # NOTE: Until missing handling is introduced, all methods will fail at data unification stage if present.
    new_data_has_na = (
        True if new_data is not None and pd.isnull(new_data).any() else False
    )
    new_labels_has_na = (
        True if new_labels is not None and pd.isnull(new_labels).any() else False
    )

    if (new_data_has_na and not missing_data_allowed) or new_labels_has_na:
        msg = "Missing values are currently not supported."
        log.error(msg)
        raise ValueError(msg)

    return new_data, new_labels, new_feature_names, new_feature_types


def autogen_schema(X, ordinal_max_items=2, feature_names=None, feature_types=None):
    """ Generates data schema for a given dataset as JSON representable.

    Args:
        X: Dataframe/ndarray to build schema from.
        ordinal_max_items: If a numeric column's cardinality
            is at most this integer,
            consider it as ordinal instead of continuous.
        feature_names: Feature names
        feature_types: Feature types

    Returns:
        A dictionary - schema that encapsulates column information,
        such as type and domain.
    """
    schema = OrderedDict()
    col_number = 0
    if isinstance(X, np.ndarray):
        log.warning(
            "Passing a numpy array to schema autogen when it should be dataframe."
        )
        if feature_names is None:
            feature_names = [f'feature_{i:04}' for i in range(1, 1 + X.shape[1])]

        # NOTE: Use rolled out infer_objects for old pandas.
        # As used from SO:
        # https://stackoverflow.com/questions/47393134/attributeerror-dataframe-object-has-no-attribute-infer-objects
        X = pd.DataFrame(X, columns=feature_names)
        try:
            X = X.infer_objects()
        except AttributeError:
            for k in list(X):
                X[k] = pd.to_numeric(X[k], errors="ignore")

    if isinstance(X, NDFrame):
        for name, col_dtype in zip(X.dtypes.index, X.dtypes):
            schema[name] = {}
            if is_numeric_dtype(col_dtype):
                if X[name].isin([np.nan, 0, 1]).all():
                    schema[name]["type"] = "categorical"
                else:
                    schema[name]["type"] = "continuous"
            elif is_string_dtype(col_dtype):
                schema[name]["type"] = "categorical"
            else:  # pragma: no cover
                warnings.warn("Unknown column: " + name, RuntimeWarning)
                schema[name]["type"] = "unknown"
            schema[name]["column_number"] = col_number
            col_number += 1

        # Override if feature_types is passed as arg.
        if feature_types is not None:
            for idx, name in enumerate(X.dtypes.index):
                schema[name]["type"] = feature_types[idx]
    else:  # pragma: no cover
        raise TypeError("EBMs only supports numpy arrays or pandas dataframes.")

    return schema


def _assign_feature_type(feature_type, is_boolean=False):
    if is_boolean or is_string_dtype(feature_type):
        return 'categorical'
    elif is_numeric_dtype(feature_type):
        return "continuous"
    else:  # pragma: no cover
        return "unknown"