# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

import numpy as np
from sklearn.utils.extmath import softmax  # type: ignore


def link(link_function, link_param, predictions):
    if link_function == "logit":
        if predictions.shape[1] == 1:
            # mono-classification
            return np.empty((len(predictions), 0), np.float64)
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


def inv_link(link_function, link_param, scores):
    if link_function == "logit":
        if scores.ndim == 1:
            # binary classification requires prepending a 0
            scores = np.c_[np.zeros(scores.shape), scores]
        elif scores.shape[1] == 0:
            # monoclassification has probability 1.0 for the only class
            return np.full((len(scores), 1), 1.0, np.float64)
        return softmax(scores)
    elif link_function == "identity":
        return scores
    elif link_function == "log":
        return np.exp(scores)
    else:
        raise ValueError("Unsupported link function: {}".format(link_function))
