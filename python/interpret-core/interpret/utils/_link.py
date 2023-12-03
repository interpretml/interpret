# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

import numpy as np

import logging

_log = logging.getLogger(__name__)

_task_dict = {
    "monoclassification": "classification",
    "logit": "classification",
    "vlogit": "classification",
    "mlogit": "classification",
    "identity": "regression",
    "log": "regression",
}


def identify_task(link):
    try:
        return _task_dict[link]
    except KeyError:
        raise ValueError(f"Unsupported link function: {link}")


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
            val = predictions.squeeze(-1)
            with np.errstate(divide="ignore"):
                # val == 1.0 and log(0.0) gives warning otherwise
                val /= 1.0 - val
                np.log(val, out=val)
            return val
        elif predictions.shape[-1] == 2:
            val = predictions[..., 1]
            tmp = predictions.sum(axis=-1)
            val /= tmp
            np.subtract(1.0, val, out=tmp)
            with np.errstate(divide="ignore"):
                # tmp == 0.0 and log(0.0) gives warning otherwise
                val /= tmp
                np.log(val, out=val)
            return val
        else:
            msg = f"predictions must have 2 elements in the last dimensions, but has {predictions.shape[-1]}."
            _log.error(msg)
            raise ValueError(msg)
    elif link == "vlogit":
        if predictions.shape[-1] <= 1:
            msg = f"predictions must have 2 or more elements in the last dimensions, but has {predictions.shape[-1]}."
            _log.error(msg)
            raise ValueError(msg)

        val = 1.0 - predictions
        with np.errstate(divide="ignore"):
            # val == 0.0 and log(0.0) gives warning otherwise
            np.divide(predictions, val, out=val)
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
            # scores == 999 gives warning otherwise from overflow
            val = np.expand_dims(np.exp(scores), axis=-1)
        inf_bool = np.isposinf(val)
        tmp = val + 1.0
        with np.errstate(invalid="ignore"):
            # val == +inf gives warning otherwise during inf/inf
            val /= tmp
        val[inf_bool] = 1.0
        np.subtract(1.0, val, out=tmp)
        return np.c_[tmp, val]
    elif link == "vlogit":
        with np.errstate(over="ignore"):
            # scores == 999 gives warning otherwise
            val = np.exp(scores)
        inf_bool = np.isposinf(val)
        with np.errstate(invalid="ignore"):
            # val == +inf gives warning otherwise
            val /= val + 1.0
        val[inf_bool] = 1.0
        return val
    elif link == "mlogit":
        # accept multinominal with 2 classes, even though it's weird
        if scores.shape[-1] <= 1:
            msg = f"scores must have 2 or more elements in the last dimensions, but has {scores.shape[-1]}."
            _log.error(msg)
            raise ValueError(msg)

        reduced_float = scores.max(axis=-1, keepdims=True)  # max() preserves NaN
        inf_bool = np.isposinf(scores)
        reduced_bool = np.isneginf(reduced_float)
        inf_bool |= reduced_bool
        np.isnan(reduced_float, out=reduced_bool)
        np.logical_not(reduced_bool, out=reduced_bool)
        inf_bool &= reduced_bool
        with np.errstate(invalid="ignore"):
            # reduced_float == +inf or all -inf gives warning otherwise
            val = scores - reduced_float
        np.exp(val, out=val)
        np.any(inf_bool, axis=-1, keepdims=True, out=reduced_bool)
        val[reduced_bool.squeeze(-1)] = 0.0
        val[inf_bool] = 1.0
        np.sum(val, axis=-1, keepdims=True, out=reduced_float)
        val /= reduced_float
        return val
    elif link == "identity":
        return scores.copy()
    elif link == "log":
        with np.errstate(over="ignore"):
            # scores == 999 gives warning otherwise
            return np.exp(scores)
    else:
        raise ValueError(f"Unsupported link function: {link}")
