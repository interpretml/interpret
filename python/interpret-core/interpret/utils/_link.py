# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

import numpy as np

import logging

_log = logging.getLogger(__name__)


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
        if predictions.shape[-1] == 1:
            val = predictions.reshape(predictions.shape[:-1])
        elif predictions.shape[-1] == 2:
            val = predictions[..., 1]
            val /= predictions.sum(axis=-1)
        else:
            msg = f"predictions must have 2 elements in the last dimensions, but has {predictions.shape[-1]}."
            _log.error(msg)
            raise ValueError(msg)
        with np.errstate(divide="ignore"):
            # val == 1.0 and log(0.0) gives warning otherwise
            val /= 1.0 - val
            np.log(val, out=val)
        return val
    elif link == "mlogit":
        # accept multinominal with 2 classes, even though it's weird
        if predictions.shape[-1] <= 1:
            msg = f"predictions must have 2 or more elements in the last dimensions, but has {predictions.shape[-1]}."
            _log.error(msg)
            raise ValueError(msg)
        val = predictions / predictions.max(axis=-1, keepdims=True)
        with np.errstate(divide="ignore"):
            # log(0.0) gives warning otherwise
            np.log(val, out=val)
        return val
    elif link == "identity":
        return predictions.copy()
    elif link == "log":
        with np.errstate(divide="ignore"):
            # log(0.0) gives warning otherwise
            return np.log(predictions)
    else:
        raise ValueError(f"Unsupported link function: {link}")


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
        with np.errstate(over="ignore"):
            # scores == 999 gives warning otherwise
            val = np.exp(scores)
        inf_bool = val == np.inf
        with np.errstate(invalid="ignore"):
            # val == +inf gives warning otherwise
            val /= val + 1.0
        val[inf_bool] = 1.0
        val = np.expand_dims(val, axis=-1)
        return np.c_[1.0 - val, val]
    elif link == "mlogit":
        # accept multinominal with 2 classes, even though it's weird
        if scores.shape[-1] <= 1:
            msg = f"scores must have 2 or more elements in the last dimensions, but has {scores.shape[-1]}."
            _log.error(msg)
            raise ValueError(msg)
        with np.errstate(invalid="ignore"):
            # val == +inf or all -inf gives warning otherwise
            val = scores - scores.max(axis=-1, keepdims=True)
        np.exp(val, out=val)
        inf_bool = (
            (scores == np.inf) | np.all(scores == -np.inf, axis=-1, keepdims=True)
        ) & ~np.any(np.isnan(scores), axis=-1, keepdims=True)
        val[np.any(inf_bool, axis=-1)] = 0.0
        val[inf_bool] = 1.0
        val /= val.sum(axis=-1, keepdims=True)
        return val
    elif link == "identity":
        return scores.copy()
    elif link == "log":
        with np.errstate(over="ignore"):
            # scores == 999 gives warning otherwise
            return np.exp(scores)
    else:
        raise ValueError(f"Unsupported link function: {link}")
