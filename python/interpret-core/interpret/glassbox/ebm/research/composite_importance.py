"""Composite Importance module

A composite is a set of terms (e.g. features). This module adds utility functions to compute
the importances of composites and append them to Global Explanations.
"""
import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.base import is_classifier

def compute_composite_importance(composite_terms, ebm, X, contributions=None):
    """Computes the importance of the composite_terms.

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

def append_composite_importance(composite_terms, ebm, X, composite_name=None, global_exp=None, global_exp_name=None, contributions=None):
    """Computes the importance of the composite_terms and appends it to a global explanation.

    In case a global explanation is provided, the composite importance will be appended to it and returned.
    Otherwise, a new global explanation will be creted and returned.

    The composite importance will only be displayed in the Summary graph.

    Args:
        composite_terms: A list of term names or term indices
        ebm: A fitted EBM
        X (numpy array): Samples used to compute the composite importance
        composite_name (str, optional): User-defined composite Name
        global_exp (EBMExplanation, optional): User-defined global explanation object
        global_exp_name (str, optional): User-defined name when creating a new global explanation
        contributions (numpy array, optional): Contributions of all terms per X's row

    Returns:
        EBMExplanation: A global explanation with the composite importance appended to it
    """
    check_is_fitted(ebm, "has_fitted_")

    if global_exp is not None:
        if global_exp.explanation_type != "global":
            raise ValueError("The provided explanation is {} but a global explanation is expected.".format(global_exp.explanation_type))
        elif global_exp._internal_obj is None or global_exp._internal_obj["overall"] is None:
            raise ValueError("The global explanation object is incomplete.")
        else:
            global_explanation = global_exp
    else:
        global_explanation = ebm.explain_global(global_exp_name)

    if composite_name is None:
        composite_name = _get_composite_name(composite_terms, ebm.get_feature_names_out())
    composite_importance = compute_composite_importance(composite_terms, ebm, X, contributions)

    global_explanation._internal_obj["overall"]["names"].append(composite_name)
    global_explanation._internal_obj["overall"]["scores"].append(composite_importance)

    return global_explanation

def get_composite_and_individual_terms(composite_terms_list, ebm, X):
    """Returns a dict containing the importances of the composite terms in composite_terms_list as well as
        all other terms in the EBM

    The dict will de sorted in descending order w.r.t. the importances

    Args:
        composite_terms_list: A list of composite terms, which are lists of term names or term indices
            e.g. [["Feature 1", "Feature 2], ["Feature 3", "Feature 4"]]
        ebm: A fitted EBM
        X (numpy array): Samples used to compute the composite importance

    Returns:
       a dict where each entry is in the form 'term_name: term_importance'
    """
    if type(composite_terms_list) is not list:
        raise ValueError("composite_terms_list should be a list.")
    elif len(composite_terms_list) == 0:
        raise ValueError("composite_terms_list should be a non-empty list.")

    _, contributions = ebm.predict_and_contrib(X)
    dict = {}

    for term in ebm.get_feature_names_out():
         dict[term] = compute_composite_importance([term], ebm, X, contributions)

    # If it's not a list of lists, we assume it's one composite term only (e.g. list of strings or ints)
    if type(composite_terms_list[0]) is not list:
        composite_name = _get_composite_name(composite_terms_list, ebm.get_feature_names_out())
        dict[composite_name] = compute_composite_importance(composite_terms_list, ebm, X, contributions)
    else:
        for composite_terms in composite_terms_list:
            composite_name = _get_composite_name(composite_terms, ebm.get_feature_names_out())
            dict[composite_name] = compute_composite_importance(composite_terms, ebm, X, contributions)

    sorted_dict = {k: v for k, v in sorted(dict.items(), key=lambda item: item[1], reverse=True)}
    return sorted_dict