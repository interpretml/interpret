""" Task related measures. """

from math import log, e
from numbers import Number
from typing import Iterable, List, Tuple
import numpy as np
import pandas as pd


def entropy(labels: Iterable, base: Number = None, normalized: bool = False) -> Number:
    """Computes entropy of label distribution.

    Args:
        labels (Iterable): Labels to compute entropy.
        base (Number, optional): Logarithmic base. Defaults to None.
        normalized (bool, optional): Return normalized entropy instead. Defaults to False.

    Returns:
        Number: Entropy.
    """

    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    value, counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.0

    # Compute entropy
    base = e if base is None else base
    for i in probs:
        ent -= i * log(i, base)

    if normalized:
        return ent / log(len(value), base)
    else:
        return ent


def class_stats(y: pd.Series, meta):
    """Compute classification label statistics.

    Args:
        y (pd.Series): Labels.

    Returns:
        List[Tuple[str, str, float, bool]]: Tuples of form: (name, description, value, is_lower_better).
    """
    labels = y.values
    labels_unique = np.unique(labels, return_counts=True)
    labels_min_cnt = np.min(labels_unique[1])
    labels_max_cnt = np.max(labels_unique[1])

    meta["class_normalized_entropy"] = float(entropy(labels, normalized=True))
    meta["num_classes"] = int(len(labels_unique))
    meta["min_class_count"] = int(labels_min_cnt)
    meta["max_class_count"] = int(labels_max_cnt)
    meta["avg_class_count"] = float(np.average(labels_unique[1]))


def regression_stats(y: pd.Series, meta):
    """Computes regression statistics on response.

    Args:
        y (pd.Series): Response.

    Returns:
        List[Tuple[str, str, float, bool]]: Tuples of form: (name, description, value, is_lower_better).
    """
    labels = y.values
    labels_avg = np.average(labels)
    labels_max = max(labels)
    labels_min = min(labels)
    meta["response_min_val"] = float(labels_min)
    meta["response_avg_val"] = float(labels_avg)
    meta["response_max_val"] = float(labels_max)


def data_stats(
    X: pd.DataFrame, y, is_classification, categorical_mask: Iterable[bool], meta
):
    """Computes data statistics on instances.

    Args:
        X (pd.DataFrame): Instances.
        y (pd.DataFrame or pd.Series): outputs
        categorical_mask (Iterable[bool]): Boolean mask on which columns are categorical.

    Returns:
        List[Tuple[str, str, float, bool]]: Tuples of form: (name, description, value, is_lower_better).
    """
    data = X.values

    avg_prop_special_values = 0.0
    for index, _ in enumerate(X):
        prop_special_values = 0.0
        if categorical_mask[index]:
            for val in data[:, index]:
                if pd.isnull(val) or val == " " or val == "":
                    prop_special_values += 1.0
        else:
            for val in data[:, index]:
                if pd.isnull(val) or val == 0:
                    prop_special_values += 1.0
        prop_special_values /= X.shape[0]
        avg_prop_special_values += prop_special_values
    avg_prop_special_values /= X.shape[1]

    n_classes = 0
    if is_classification and y is not None:
        n_classes = len(np.unique(y))

    prop_cat_cols = float(sum([int(x) for x in categorical_mask]))
    prop_cat_cols /= len(categorical_mask)

    meta["n_rows"] = int(X.shape[0])
    meta["n_cols"] = int(X.shape[1])
    meta["n_classes"] = int(n_classes)
    meta["prop_cat_cols"] = float(prop_cat_cols)
    meta["row_col_ratio"] = float(X.shape[0]) / float(X.shape[1])
    meta["avg_prop_special_values"] = float(avg_prop_special_values)
