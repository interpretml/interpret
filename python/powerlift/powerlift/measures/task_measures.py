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


def class_stats(y: pd.Series) -> List[Tuple[str, str, float, bool]]:
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

    return [
        (
            "class_normalized_entropy",
            "Normalized entropy of classes.",
            float(entropy(labels, normalized=True)),
            False,
        ),
        ("num_classes", "Number of distinct classes.", float(len(labels_unique)), True),
        ("min_class_count", "Minimum class count.", float(labels_min_cnt), False),
        ("max_class_count", "Maximum class count.", float(labels_max_cnt), False),
        (
            "avg_class_count",
            "Average class count.",
            float(np.average(labels_unique[1])),
            False,
        ),
    ]


def regression_stats(y: pd.Series) -> List[Tuple[str, str, float, bool]]:
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
    return [
        ("response_min_val", "Minimum value of response.", float(labels_min), True),
        ("response_avg_val", "Average value of response.", float(labels_avg), True),
        ("response_max_val", "Maximum value of response.", float(labels_max), True),
    ]


def data_stats(
    X: pd.DataFrame, categorical_mask: Iterable[bool]
) -> List[Tuple[str, str, float, bool]]:
    """Computes data statistics on instances.

    Args:
        X (pd.DataFrame): Instances.
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

    prop_cat_cols = float(sum([int(x) for x in categorical_mask]))
    prop_cat_cols /= len(categorical_mask)

    return [
        ("n_rows", "Number of rows.", float(X.shape[0]), False),
        ("n_cols", "Number of columns.", float(X.shape[1]), True),
        (
            "prop_cat_cols",
            "Proportion of categorical columns",
            float(prop_cat_cols),
            True,
        ),
        ("row_col_ratio", "Row to column ratio", float(X.shape[0] / X.shape[1]), True),
        (
            "avg_prop_special_values",
            "Average number of special value proportion per column.",
            float(avg_prop_special_values),
            True,
        ),
    ]
