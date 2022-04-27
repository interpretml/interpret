"""Composite Importance module

A composite is a set of terms (e.g. features). This module adds utility functions to compute
the importances of composites and append them to Global Explanations.
"""

import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.base import is_classifier
import logging

logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def compute_composite_importance(composite_terms, ebm, X, contributions=None):
    """Computes the importance of the composite terms.

    Args:
        composite_terms: A list of term names or term indices
        ebm: A fitted EBM
        X (numpy array): Samples used to compute the composite importance
        contributions (numpy array, optional): Contributions of all terms per X's row

    Returns:
        float: The composite importance
    """
    check_is_fitted(ebm, "has_fitted_")

    if contributions is None:
        _, contributions = ebm.predict_and_contrib(X)
    ebm_term_names = ebm.get_feature_names_out()
    composite_term_indices = []

    for term in composite_terms:
        if isinstance(term, str):
            try:
                composite_term_indices.append(ebm_term_names.index(term))
            except ValueError:
                raise ValueError("Term '{}' not found.".format(term))
        elif isinstance(term, int) and 0 <= term < len(ebm_term_names):
            composite_term_indices.append(term)
        else:
            raise ValueError("Term '{}' is not a string or a valid integer.".format(term))

    if len(composite_term_indices) == 0:
        raise ValueError("composite_terms does not contain any valid terms.")

    # For multiclass we take the average of contributions per class
    # TODO this is consistent to what Interpret is doing but might be changed
    if is_classifier(ebm) and 2 < len(ebm.classes_):
        contributions = np.average(np.abs(contributions), axis=-1)

    abs_sum_per_row = np.empty(len(contributions), np.float64)
    for i in range(len(contributions)):
        sum = 0.0
        for j in composite_term_indices:
            sum += contributions[i][j]
        abs_sum_per_row[i] = abs(sum)

    return np.average(abs_sum_per_row)

def _get_composite_name(composite_terms, ebm_term_names):
    """Returns the composite name in the format "term_name_1 & term_name_2 & ..."

    Args:
        composite_terms: A list of term names or term indices
        ebm_term_names: a list of all ebm term names

    Returns:
        str: The composite name
    """
    name = ""
    for term in composite_terms:
        if isinstance(term, str) and term in ebm_term_names:
            name += term if len(name) == 0 else " & " + term
        elif isinstance(term, int) and 0 <= term < len(ebm_term_names):
            name += ebm_term_names[term] if len(name) == 0 else " & " + ebm_term_names[term]
        else:
            raise ValueError("Term '{}' is not a string or a valid integer.".format(term))
    return name

def append_composite_importance(composite_terms, ebm, global_exp, X, composite_name=None, contributions=None):
    """Computes the composite importance for composite_terms and returns a global explanation containing it
    
    The composite importance will only be displayed in the "Summary" Graph.

    Args:
        composite_terms: A list of term names or term indices
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
            composite_name = _get_composite_name(composite_terms, ebm.get_feature_names_out())
        composite_importance = compute_composite_importance(composite_terms, ebm, X, contributions)
        global_exp._internal_obj["overall"]["names"].append(composite_name)
        global_exp._internal_obj["overall"]["scores"].append(composite_importance)