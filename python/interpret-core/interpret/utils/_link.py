# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

import numpy as np

import logging
_log = logging.getLogger(__name__)

def _softmax(x):
    e_x = np.exp(x - x.max(axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def link_func(predictions, link, link_param=np.nan):
    """Applies the link function to predictions to generate scores.

    Args:
        predictions: Numpy array of predictions for samples.
        link: string containing the type of link function to use
        link_param: Optional. numeric parameter that is specified by the link function.

    Returns:
        Scores converted by the link function.
    """

    if link == "monoclassification":
        if 2 <= predictions.shape[-1]:
            msg = f"predictions must have 1 element in the last dimensions, but has {predictions.shape[-1]}."
            _log.error(msg)
            raise ValueError(msg)
        return np.empty(predictions.shape[:-1] + (0,), np.float64)
    elif link == "logit":
        if predictions.shape[-1] != 2:
            msg = f"predictions must have 2 elements in the last dimensions, but has {predictions.shape[-1]}."
            _log.error(msg)
            raise ValueError(msg)
        maxes = predictions.max(axis=-1, keepdims=True)
        with np.errstate(divide="ignore"):
            scores = np.log(predictions / maxes)
        scores = scores[..., 1] - scores[..., 0]
        return scores
    elif link == "mlogit":
        # accept multinominal with 2 classes, even though it's weird
        if predictions.shape[-1] <= 1:
            msg = f"predictions must have 2 or more elements in the last dimensions, but has {predictions.shape[-1]}."
            _log.error(msg)
            raise ValueError(msg)
        maxes = predictions.max(axis=-1, keepdims=True)
        with np.errstate(divide="ignore"):
            scores = np.log(predictions / maxes)
        return scores
    elif link == "identity":
        return predictions.copy()
    elif link == "log":
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

    if link == "monoclassification":
        if scores.shape[-1] != 0:
            msg = f"scores must have 0 elements in the last dimensions, but has {scores.shape[-1]}."
            _log.error(msg)
            raise ValueError(msg)
        return np.full(scores.shape[:-1] + (1,), 1.0, np.float64)
    elif link == "logit":
        scores = np.expand_dims(scores, axis=-1)
        scores = np.insert(scores, 0, 0, axis=-1)
        return _softmax(scores)
    elif link == "mlogit":
        # accept multinominal with 2 classes, even though it's weird
        if scores.shape[-1] <= 1:
            msg = f"scores must have 2 or more elements in the last dimensions, but has {scores.shape[-1]}."
            _log.error(msg)
            raise ValueError(msg)
        return _softmax(scores)
    elif link == "identity":
        return scores.copy()
    elif link == "log":
        return np.exp(scores)
    else:
        raise ValueError("Unsupported link function: {}".format(link))
