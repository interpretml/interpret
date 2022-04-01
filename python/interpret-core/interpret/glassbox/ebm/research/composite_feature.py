"""Composite Feature Importance module

A composite feature is a set of features. This module adds utility functions to compute
the importances of composite features and append them to Global Explanations.
"""

import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.base import is_classifier
import logging

logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def compute_composite_feature_importance(composite_feature_list, ebm, X, contributions=None):
    """Computes the importance of a composite feature (i.e. a set of features).

    Args:
        composite_feature_list: A list of feature names or feature indices
        ebm: A fitted EBM
        X (numpy array): Samples used to compute the composite feature importance
        contributions (numpy array, optional): Contributions of all features per X's row

    Returns:
        float: The composite feature importance
    """
    check_is_fitted(ebm, "has_fitted_")

    if contributions is None:
        _, contributions = ebm.predict_and_contrib(X)
    ebm_feature_names = ebm.get_feature_names_out()
    composite_feature_indices = []

    for feature in composite_feature_list:
        if isinstance(feature, str):
            try:
                composite_feature_indices.append(ebm_feature_names.index(feature))
            except ValueError:
                log.warning("Feature '{}' not found.".format(feature))
        elif isinstance(feature, int) and 0 <= feature < len(ebm_feature_names):
            composite_feature_indices.append(feature)
        else:
            log.warning("Feature '{}' is not a string or a valid integer.".format(feature))

    if len(composite_feature_indices) == 0:
        raise ValueError("composite_feature_list does not contain any valid features.")

    # For multiclass we take the average of contributions per class
    # TODO this is consistent to what Interpret is doing but might be changed
    if is_classifier(ebm) and 2 < len(ebm.classes_):
        contributions = np.average(np.abs(contributions), axis=-1)

    abs_sum_per_row = np.empty(len(contributions), np.float64)
    for i in range(len(contributions)):
        sum = 0.0
        for j in composite_feature_indices:
            sum += contributions[i][j]
        abs_sum_per_row[i] = abs(sum)

    return np.average(abs_sum_per_row)

def _get_composite_feature_name(composite_feature_list, ebm_feature_names):
    """Returns the composite feature name in the format "feature_name_1 and feature_name_2 and ..."

    Args:
        composite_feature_list: A list of feature names or feature indices
        ebm_feature_names: a list of all ebm feature names

    Returns:
        str: The composite name
    """
    name = ""
    for feature in composite_feature_list:
        if isinstance(feature, str) and feature in ebm_feature_names:
            name += feature if len(name) == 0 else " and " + feature
        elif isinstance(feature, int) and 0 <= feature < len(ebm_feature_names):
            name += ebm_feature_names[feature] if len(name) == 0 else " and " + ebm_feature_names[feature]
        else:
            log.warning("Feature '{}' is not a string or a valid integer.".format(feature))
    return name

def append_composite_feature_importance(composite_feature_list, ebm, global_exp, X, composite_name=None, contributions=None):
    """ Computes and appends a composite feature importance to the global explanation, which
        will only be displayed in the "Summary" Graph

    Args:
        composite_feature_list: A list of feature names or feature indices
        ebm: A fitted EBM
        global_exp: An EBM Global Explanation
        X (numpy array): Samples used to compute the composite feature importance
        composite_name (str, optional): Name to be appended to the global explanation
        contributions (numpy array, optional): Contributions of all features per X's row
    """
    check_is_fitted(ebm, "has_fitted_")

    if global_exp.explanation_type != "global":
        raise ValueError("The provided explanation is {} but a global explanation is expected.".format(global_exp.explanation_type))
    elif global_exp._internal_obj is None or global_exp._internal_obj["overall"] is None:
        raise ValueError("The global explanation object is incomplete.")
    else:
        if composite_name is None:
            composite_name = _get_composite_feature_name(composite_feature_list, ebm.get_feature_names_out())
        composite_importance = compute_composite_feature_importance(composite_feature_list, ebm, X, contributions)
        global_exp._internal_obj["overall"]["names"].append(composite_name)
        global_exp._internal_obj["overall"]["scores"].append(composite_importance)