# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

import numpy as np
from sklearn.utils.extmath import softmax  # type: ignore


def link(link_function, link_param, predictions):
    if link_function == "logit":
        maxes = np.amax(predictions, axis=1)
        with np.errstate(divide="ignore"):
            scores = np.log(predictions / maxes[:, np.newaxis])
        if scores.shape[1] == 2:  # binary classification
            scores = scores[:, 1] - scores[:, 0]
        return scores
    elif link_function == "identity":
        return predictions
    elif link_function == "log":
        return np.log(predictions)
    else:
        raise ValueError("Unsupported link function: {}".format(link_function))


def inv_link(link_function, link_param, scores, n_classes):
    if link_function == "logit":
        if n_classes == 1:
            # if there is only one class then all probabilities are 100%
            return np.full((len(scores), 1), 1.0, np.float64)

        if scores.ndim == 1:
            # binary classification requires prepending a 0
            scores = np.c_[np.zeros(scores.shape), scores]
        return softmax(scores)
    elif link_function == "identity":
        return scores
    elif link_function == "log":
        return np.exp(scores)
    else:
        raise ValueError("Unsupported link function: {}".format(link_function))
