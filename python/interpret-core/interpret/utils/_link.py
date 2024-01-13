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

    # For handling classification in the the link_func, the rule is that the last
    # dimension needs to be an array that contains the probabilities for 1 sample.
    # For "logit", if we treated an array of three values [0.75, 0.75, 0.75] as 3
    # samples, then it would be ambiguous when we recieve values like [0.75, 0.25]
    # whether this was meant to be 2 samples or the False/True probabilities for 1
    # sample. We handle monoclassification and multiclass in the same way.

    predictions = np.array(predictions, np.float64, copy=False)
    if link == "monoclassification":
        if predictions.ndim == 0:
            if predictions == 1.0:
                return -np.inf
            elif np.isnan(predictions):
                return np.nan
            else:
                msg = "monoclassification with 1 element must be 1.0 or NaN"
                _log.error(msg)
                raise ValueError(msg)
        scores = np.full(predictions.shape[:-1], -np.inf, np.float64)
        if 1 == predictions.shape[-1]:
            predictions = predictions.squeeze(-1)
            bools = np.isnan(predictions)
            scores[bools] = np.nan
            bools |= predictions == 1.0
            if not bools.all():
                msg = "monoclassification with 1 element must have all 1.0s or NaN"
                _log.error(msg)
                raise ValueError(msg)
        elif 0 != predictions.shape[-1]:
            msg = f"predictions must have 1 element in the last dimensions, but has {predictions.shape[-1]}."
            _log.error(msg)
            raise ValueError(msg)
        return scores
    elif link == "logit":
        if predictions.ndim == 0:
            val = predictions
        elif predictions.shape[-1] == 1:
            val = predictions.squeeze(-1)
        elif predictions.shape[-1] == 2:
            val = predictions[..., 1]
            val /= predictions.sum(axis=-1)
        else:
            msg = f"predictions must have 1 or 2 elements in the last dimensions, but has {predictions.shape[-1]}."
            _log.error(msg)
            raise ValueError(msg)

        with np.errstate(divide="ignore"):
            # val == 1.0 and val == 0.0 gives warning otherwise
            val /= 1.0 - val
            np.log(val, out=val)
        if val.ndim == 0:
            val = val.item()

        return val
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

    scores = np.array(scores, np.float64, copy=False)
    if link == "monoclassification":
        bools = np.isnan(scores)
        preds = np.ones(scores.shape + (1,), np.float64)
        preds[np.expand_dims(bools, axis=-1)] = np.nan

        bools |= scores == -np.inf
        if not bools.all():
            msg = "monoclassification must have all -infs or NaN"
            _log.error(msg)
            raise ValueError(msg)

        return preds
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
