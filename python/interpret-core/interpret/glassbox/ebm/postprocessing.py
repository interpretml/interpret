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
