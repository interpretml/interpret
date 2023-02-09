"""Group importance module

This module adds utility functions to compute the importances of groups of
features or terms and append them to Global Explanations.

A term denotes both single features and interactions (pairs).
"""
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.utils.validation import check_is_fitted
from sklearn.base import is_classifier

def compute_group_importance(term_list, ebm, X, contributions=None):
    """Computes the importance of a group of terms.

    Args:
        term_list: A list of term names or indices
        ebm: A fitted EBM
        X (numpy array): Samples used to compute the group importance
        contributions (numpy array, optional): Contributions of all terms per X's row

    Returns:
        float: term_list's group importance
    """
    check_is_fitted(ebm, "has_fitted_")

    if contributions is None:
        _, contributions = ebm.predict_and_contrib(X)
    ebm_term_names = ebm.term_names_
    term_group_indices = []

    for term in term_list:
        if isinstance(term, str):
            try:
                term_group_indices.append(ebm_term_names.index(term))
            except ValueError:
                raise ValueError(f"Term '{term}' not found.")
        elif isinstance(term, int) and 0 <= term < len(ebm_term_names):
            term_group_indices.append(term)
        else:
            raise ValueError(f"Term '{term}' is not a string or a valid integer.")

    if len(term_group_indices) == 0:
        raise ValueError("term_list does not contain any valid terms.")

    # For multiclass we take the average of contributions per class
    # TODO this is consistent to what Interpret is doing but might be changed
    if is_classifier(ebm) and 2 < len(ebm.classes_):
        contributions = np.average(np.abs(contributions), axis=-1)

    abs_sum_per_row = np.empty(len(contributions), np.float64)
    for i in range(len(contributions)):
        sum = 0.0
        for j in term_group_indices:
            sum += contributions[i][j]
        abs_sum_per_row[i] = abs(sum)

    return np.average(abs_sum_per_row)

def _get_group_name(term_list, ebm_term_names):
    """Returns the group's name in the format "term_name_1, term_name_2, ..."

    Args:
        term_list: A list of term names or indices
        ebm_term_names: a list of all ebm term names

    Returns:
        str: The group name
    """
    name = ""
    for term in term_list:
        if isinstance(term, str) and term in ebm_term_names:
            name += term if len(name) == 0 else ", " + term
        elif isinstance(term, int) and 0 <= term < len(ebm_term_names):
            name += ebm_term_names[term] if len(name) == 0 else ", " + ebm_term_names[term]
        else:
            raise ValueError(f"Term '{term}' is not a string or a valid integer.")
    return name

def append_group_importance(term_list, ebm, X, group_name=None, global_exp=None, global_exp_name=None, contributions=None):
    """Computes the importance of a group of terms and appends it to a global explanation.

    In case a global explanation is provided, the group importance will be appended to it and returned.
    Otherwise, a new global explanation will be creted and returned.

    The group importance will only be displayed in the Summary graph.

    Args:
        term_list: A list of term names or indices
        ebm: A fitted EBM
        X (numpy array): Samples used to compute the group importance
        group_name (str, optional): User-defined group name
        global_exp (EBMExplanation, optional): User-defined global explanation object
        global_exp_name (str, optional): User-defined name when creating a new global explanation
        contributions (numpy array, optional): Contributions of all terms per X's row

    Returns:
        EBMExplanation: A global explanation with the group importance appended to it
    """
    check_is_fitted(ebm, "has_fitted_")

    if global_exp is not None:
        if global_exp.explanation_type != "global":
            raise ValueError(f"The provided explanation is {global_exp.explanation_type} but a global explanation is expected.")
        elif global_exp._internal_obj is None or global_exp._internal_obj["overall"] is None:
            raise ValueError("The global explanation object is incomplete.")
        else:
            global_explanation = global_exp
    else:
        global_explanation = ebm.explain_global(global_exp_name)

    if group_name is None:
        group_name = _get_group_name(term_list, ebm.term_names_)

    if group_name in global_explanation._internal_obj["overall"]["names"]:
        raise ValueError(f"The group {group_name} is already in the global explanation.")

    group_importance = compute_group_importance(term_list, ebm, X, contributions)

    global_explanation._internal_obj["overall"]["names"].append(group_name)
    global_explanation._internal_obj["overall"]["scores"].append(group_importance)

    return global_explanation

def get_group_and_individual_importances(term_groups_list, ebm, X, contributions=None):
    """Returns a dict containing the importances of the groups in term_groups_list as well as
        all other EBM terms

    The dict will de sorted in descending order w.r.t. the importances

    Args:
        term_groups_list: A list of term groups, which are lists of term names or indices
            e.g. [["Feature 1", "Feature 2], ["Feature 3", "Feature 4"]]
        ebm: A fitted EBM
        X (numpy array): Samples used to compute the group importance
        contributions (numpy array, optional): Contributions of all terms per X's row

    Returns:
       a dict where each entry is in the form 'term_name: term_importance'
    """
    if type(term_groups_list) is not list:
        raise ValueError("term_groups_list should be a list.")
    elif len(term_groups_list) == 0:
        raise ValueError("term_groups_list should be a non-empty list.")

    if contributions is None:
        _, contributions = ebm.predict_and_contrib(X)

    dict = {}

    for term in ebm.term_names_:
         dict[term] = compute_group_importance([term], ebm, X, contributions)

    # If it's not a list of lists, we assume it's only one term group (e.g. list of strings or ints)
    if type(term_groups_list[0]) is not list:
        group_name = _get_group_name(term_groups_list, ebm.term_names_)
        dict[group_name] = compute_group_importance(term_groups_list, ebm, X, contributions)
    else:
        for term_group in term_groups_list:
            group_name = _get_group_name(term_group, ebm.term_names_)
            dict[group_name] = compute_group_importance(term_group, ebm, X, contributions)

    sorted_dict = {k: v for k, v in sorted(dict.items(), key=lambda item: item[1], reverse=True)}
    return sorted_dict

def get_individual_importances(ebm, X, contributions=None):
    """Returns a dict containing the importances of all EBM terms

    The dict will de sorted in descending order w.r.t. the importances

    Args:
        ebm: A fitted EBM
        X (numpy array): Samples used to compute the group importance
        contributions (numpy array, optional): Contributions of all terms per X's row

    Returns:
       a dict where each entry is in the form 'term_name: term_importance'
    """
    if contributions is None:
        _, contributions = ebm.predict_and_contrib(X)

    dict = {}
    for term in ebm.term_names_:
         dict[term] = compute_group_importance([term], ebm, X, contributions)

    sorted_dict = {k: v for k, v in sorted(dict.items(), key=lambda item: item[1], reverse=True)}
    return sorted_dict

def get_importance_per_top_groups(ebm, X):
    """ Returns a Dataframe with the importances of groups of terms, such that:

    The first group is the term with the highest individual importance (i.e. top term), the second group is
    composed by the top 2 terms, and so on. For example:
        Group 1 - ['Age']
        Group 2 - ['Age', 'MaritalStatus']
        Group 3 - ['Age', 'MaritalStatus', 'CapitalGain']
        ...
        Group N - All terms

    Args:
        ebm: A fitted EBM
        X (numpy array): Samples used to compute the group importance

    Returns:
       a pandas Dataframe with three columns: group_names, terms_per_group and importances
    """
    _, contributions = ebm.predict_and_contrib(X)
    individual_importances = get_individual_importances(ebm, X, contributions)

    # Create groups of terms starting with the most important and adding each subsequent term
    groups_list = []
    temp_group = []
    for key in individual_importances.keys():
        if len(temp_group) > 0:
            temp_group = list(groups_list[-1])
        temp_group.append(key)
        groups_list.append(temp_group)

    # Compute the importance of each group in groups_list
    group_index = 1
    output_dict = {}
    for group in groups_list:
        group_name = f"Group {group_index}"
        output_dict[group_name] = compute_group_importance(group, ebm, X, contributions)
        group_index += 1

    df = pd.DataFrame({
        "groups": output_dict.keys(),
        "terms_per_group": groups_list,
        "importances": output_dict.values(),
    })

    return df

def plot_importance_per_top_groups(ebm, X):
    """ Plots a plotly graph where the x-axis represents groups of top K terms and the y-axis their importances.

    The first group is the terms with the highest individual importance (i.e. top term), the second group is
    composed by the top 2 terms, and so on. For example:
        Group 1 - ['Age']
        Group 2 - ['Age', 'MaritalStatus']
        Group 3 - ['Age', 'MaritalStatus', 'CapitalGain']
        ...
        Group N - All terms

    Args:
        ebm: A fitted EBM
        X (numpy array): Samples used to compute the group importance
    """
    df = get_importance_per_top_groups(ebm, X)

    fig = px.line(df, x="groups", y="importances", title='Group Importances')
    fig.show()