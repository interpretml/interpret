"""Task related measures."""

from math import e, log
from numbers import Number
from typing import Iterable

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
    return ent


def class_stats(y: pd.Series, meta):
    """Compute classification label statistics.

    Args:
        y (pd.Series): Labels.

    Returns:
        List[Tuple[str, str, float, bool]]: Tuples of form: (name, description, value, is_lower_better).
    """
    labels = y.values
    _, counts = np.unique(labels, return_counts=True)

    meta["n_classes"] = int(len(counts))
    meta["class_normalized_entropy"] = float(entropy(labels, normalized=True))
    meta["min_class_count"] = int(np.min(counts))
    meta["max_class_count"] = int(np.max(counts))


def regression_stats(y: pd.Series, meta):
    """Computes regression statistics on response.

    Args:
        y (pd.Series): Response.

    Returns:
        List[Tuple[str, str, float, bool]]: Tuples of form: (name, description, value, is_lower_better).
    """
    labels = y.values
    meta["n_classes"] = 0
    meta["response_min_val"] = float(min(labels))
    meta["response_avg_val"] = float(np.average(labels))
    meta["response_max_val"] = float(max(labels))


def data_stats(X: pd.DataFrame, categorical_mask: Iterable[bool], meta):
    """Computes data statistics on instances.

    Args:
        X (pd.DataFrame): Instances.
        categorical_mask (Iterable[bool]): Boolean mask on which columns are categorical.

    Returns:
        List[Tuple[str, str, float, bool]]: Tuples of form: (name, description, value, is_lower_better).
    """

    percent_special_values = 0.0
    max_unique_continuous = 0
    max_categories = 0
    total_categories = 0

    for i in range(X.shape[1]):
        col = X.iloc[:, i]
        if categorical_mask[i]:
            n_unique = col.nunique(dropna=False)
            max_categories = max(max_categories, n_unique)
            total_categories += n_unique
            for val in col.values:
                if pd.isnull(val) or val.strip() == "":
                    percent_special_values += 1.0
        else:
            max_unique_continuous = max(max_unique_continuous, col.nunique())
            for val in col.values:
                if pd.isnull(val) or val == 0:
                    percent_special_values += 1.0
    percent_special_values /= X.shape[0] * X.shape[1]

    percent_categorical = float(sum([int(x) for x in categorical_mask]))
    percent_categorical /= len(categorical_mask)

    meta["n_samples"] = int(X.shape[0])
    meta["n_features"] = int(X.shape[1])
    meta["max_unique_continuous"] = int(max_unique_continuous)
    meta["max_categories"] = int(max_categories)
    meta["total_categories"] = int(total_categories)
    meta["percent_categorical"] = float(percent_categorical)
    meta["percent_special_values"] = float(percent_special_values)
