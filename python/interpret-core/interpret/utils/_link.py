# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

import numpy as np
from sklearn.utils.extmath import softmax  # type: ignore


def link_func(predictions, link, link_param=np.nan):
    """Applies the link function to predictions to generate scores.

    Args:
        predictions: Numpy array of predictions for samples.
        link: string containing the type of link function to use
        link_param: Optional. numeric parameter that is specified by the link function.

    Returns:
        Scores converted by the link function.
    """

    if link == "logit":
        if predictions.ndim != 2:
            msg = f"predictions must have 2 dimensions."
            _log.error(msg)
            raise ValueError(msg)
        if predictions.shape[1] <= 1:
            # mono-classification
            return np.empty((len(predictions), 0), np.float64)
        maxes = np.amax(predictions, axis=1)
        with np.errstate(divide="ignore"):
            scores = np.log(predictions / maxes[:, np.newaxis])
        if scores.shape[1] == 2:  # binary classification
            scores = scores[:, 1] - scores[:, 0]
        return scores
    elif link == "identity":
        if predictions.ndim != 1:
            msg = f"predictions must have 1 dimensions."
            _log.error(msg)
            raise ValueError(msg)
        return predictions
    elif link == "log":
        if predictions.ndim != 1:
            msg = f"predictions must have 1 dimensions."
            _log.error(msg)
            raise ValueError(msg)
        return np.log(predictions)
    else:
        raise ValueError("Unsupported link function: {}".format(link))


def inv_link(scores, link, link_param=np.nan):
    """Applies the inverse link function to scores to generate predictions.

    Args:
        scores: Numpy array of scores for samples.
        link: string containing the type of link function to use
        link_param: Optional. numeric parameter that is specified by the link function.

    Returns:
        Predictions converted by the link function.
    """

    if link == "logit":
        if scores.ndim != 1 and scores.ndim != 2:
            msg = f"scores must have either 1 dimension or two."
            _log.error(msg)
            raise ValueError(msg)
        if scores.ndim == 1 or scores.shape[1] == 1:
            # binary classification requires prepending a 0
            scores = np.c_[np.zeros(scores.shape), scores]
        elif scores.shape[1] == 0:
            # mono classification has probability 1.0 for the only class
            return np.full((len(scores), 1), 1.0, np.float64)
        return softmax(scores)
    elif link == "identity":
        if scores.ndim != 1:
            msg = f"scores must have 1 dimension."
            _log.error(msg)
            raise ValueError(msg)
        return scores
    elif link == "log":
        if scores.ndim != 1:
            msg = f"scores must have 1 dimension."
            _log.error(msg)
            raise ValueError(msg)
        return np.exp(scores)
    else:
        raise ValueError("Unsupported link function: {}".format(link))
