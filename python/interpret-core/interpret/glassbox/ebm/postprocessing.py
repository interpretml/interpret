# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

import copy
import numpy as np


def multiclass_postprocess(
    X_binned, feature_graphs, binned_predict_proba, feature_types
):
    """ Postprocesses multiclass model graphs with desired properties.

    Args:
        X_binned: Training dataset, pre-binned. Contains integer values, 0+. Each value is a unique bin.
        feature_graphs: List of 2d numpy arrays. List is size d for d features. Each numpy array is of size b for b bins in the feature. Each bin has k elements for k classes.
        binned_predict_proba: Function that takes in X_binned, returns 2d numpy array of predictions. Each row in the return vector has k elements, with probability of belonging to class k.
        feature_types: List of strings containing either "categorical" or "numeric" for each feature.

    Returns:
        Dictionary with updated model graphs and new intercepts.
    """

    updated_feature_graphs = copy.deepcopy(feature_graphs)
    K = feature_graphs[0].shape[1]

    # Compute the predicted probability on the original data.
    predprob = binned_predict_proba(X_binned)
    predprob_prev = [None] * len(feature_graphs)

    # Compute the predicted probability on the counterfactual data with each value in feature i decrease by 1.
    for i in range(len(feature_graphs)):
        data_prev = np.copy(X_binned)
        data_prev[i, :] = np.maximum(X_binned[i, :] - 1, 0)
        predprob_prev[i] = binned_predict_proba(data_prev)

    intercepts = np.zeros(K)
    for i in range(len(feature_graphs)):
        bincount = np.bincount(X_binned[i, :].astype(int))
        # TODO: shouldn't this be "continuous" instead of "numeric"?
        if feature_types[i] == "numeric":
            num_bins = feature_graphs[i].shape[0]
            change = np.zeros(num_bins)
            for v in range(1, num_bins):
                subset_index = X_binned[i, :] == v
                ratio = np.divide(
                    predprob[subset_index, :], predprob_prev[i][subset_index, :]
                )
                sum_ratio = np.sum(np.subtract(ratio, 1), axis=0)
                difference = (
                    feature_graphs[i][v, :]
                    - feature_graphs[i][v - 1, :]
                    + change[v - 1]
                )
                change[v] = np.mean(difference)
                new_difference = difference - change[v]
                back_change = 0
                for k in range(K):
                    if new_difference[k] * sum_ratio[k] < 0 and abs(back_change) < abs(
                        new_difference[k]
                    ):
                        back_change = new_difference[k]
                change[v] = change[v] + back_change
            updated_feature_graphs[i] = np.subtract(
                updated_feature_graphs[i], change.reshape((num_bins, -1))
            )
        for k in range(K):
            mean = (
                np.sum(np.multiply(updated_feature_graphs[i][:, k], bincount))
                / X_binned.shape[1]
            )
            updated_feature_graphs[i][:, k] = np.subtract(
                updated_feature_graphs[i][:, k], mean
            )
            intercepts[k] += mean
    return {"feature_graphs": updated_feature_graphs, "intercepts": intercepts}

def multiclass_postprocess2(
    n_classes, term_scores, bin_weights, intercept
):
    """ Postprocesses multiclass model graphs with desired properties.
    """

    # TODO: our existing implementation has a bug where it always uses the simpler method of taking 
    # the mean of the class scores.  Copy this behavior for now since it's a lot simpler when
    # moving to the generator unify_columns function.  Also, this method generalizes to tensors

    # TODO: we can probably do all the classes together, and that would make it generalize to interactions as well
    # TODO: this code, if we continue to do multiclass this way, can be merged with binary and regression handling
    #       Look at the alternate branch in the caller to multiclass_postprocess2

    for i in range(len(term_scores)):
        for k in range(n_classes):
            mean = np.multiply(term_scores[i][:, k], bin_weights[i]).sum() / bin_weights[i].sum()
            term_scores[i][:, k] = np.subtract(term_scores[i][:, k], mean)
            intercept[k] += mean
