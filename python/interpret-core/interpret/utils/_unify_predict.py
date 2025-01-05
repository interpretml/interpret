# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

import logging

import numpy as np
from sklearn.base import is_classifier, is_regressor

from ._clean_simple import clean_dimensions
from ._native import Native

try:
    import pandas as pd

    _pandas_installed = True
except ImportError:
    _pandas_installed = False

_log = logging.getLogger(__name__)


def determine_classes(model, data, n_samples):
    if n_samples == 0:
        msg = "data cannot have 0 samples"
        _log.error(msg)
        raise ValueError(msg)

    classes = None
    if is_classifier(model):
        classes = model.classes_
        model = model.predict_proba
        preds = clean_dimensions(model(data), "model")
        if n_samples == 1:  # then the sample dimension would have been eliminated
            if preds.ndim != 1:
                msg = "model.predict_proba(data) returned inconsistent number of dimensions"
                _log.error(msg)
                raise ValueError(msg)
            n_classes = preds.shape[0]
        elif preds.shape[0] == 0:
            # we have at least 2 samples, so this means classes was an empty dimension
            n_classes = 0
        elif preds.shape[0] != n_samples:
            msg = "model.predict_proba(data) returned inconsistent number of samples compared to data"
            _log.error(msg)
            raise ValueError(msg)
        elif preds.ndim == 1:
            # we have at least 2 samples, so the one dimension must be for samples, and the other dimension must have been 1 class (mono-classification)
            n_classes = 1
        else:
            n_classes = preds.shape[1]

        if len(classes) == 1:
            # for single-class problems, treat it as binary classification
            # where the second class probability is always 0
            n_classes = 2
            orig_model = model
            original_class = classes[0]

            print(
                f"Warning: Model was trained on single-class data. Model will always predict class {original_class}."
            )

            def mono_classification_model(data):
                preds = orig_model(data)
                if preds.ndim == 1:
                    preds = preds.reshape(-1, 1)
                # add zero probabilities for the synthetic class
                return np.hstack([preds, np.zeros_like(preds)])

            model = mono_classification_model
            # keep original class and add any different value as synthetic class
            synthetic_class = "other" if original_class != "other" else "synthetic"
            classes = np.array([original_class, synthetic_class])

        if n_classes != len(classes):
            msg = "class number mismatch"
            _log.error(msg)
            raise ValueError(msg)
    elif is_regressor(model):
        n_classes = Native.Task_Regression
        model = model.predict
        preds = clean_dimensions(model(data), "model")
        if preds.ndim != 1:
            msg = "model.predict(data) must have only 1 dimension"
            _log.error(msg)
            raise ValueError(msg)
        if preds.shape[0] != n_samples:
            msg = "model.predict(data) returned inconsistent number of samples compared to data"
            _log.error(msg)
            raise ValueError(msg)
    else:
        preds = clean_dimensions(model(data), "model")
        if n_samples == 1:  # then the sample dimension would have been eliminated
            if preds.ndim != 1:
                msg = (
                    "model(data) has an inconsistent number of samples compared to data"
                )
                _log.error(msg)
                raise ValueError(msg)
            if preds.shape[0] != 1:
                # regression is always 1, so it's probabilities, and therefore classification
                n_classes = preds.shape[0]
            else:
                # it could be mono-classification, but that's unlikely, so it's regression
                n_classes = Native.Task_Regression
        elif preds.shape[0] == 0:
            # we have at least 2 samples, so this means classes was an empty dimension
            n_classes = 0
        elif preds.shape[0] != n_samples:
            msg = "model(data) has an inconsistent number of samples compared to data"
            _log.error(msg)
            raise ValueError(msg)
        elif preds.ndim == 1:
            # we have at least 2 samples, so the first dimension must be for samples, and the second held 1 value.
            # it could be mono-classification, but that's unlikely, so it's regression
            n_classes = Native.Task_Regression
        else:
            # we see a non-1 number of items, so it's probabilities, and therefore classification
            n_classes = preds.shape[1]

    # at this point model has been converted to a predict_fn
    return model, n_classes, classes


def unify_predict_fn(predict_fn, X, class_idx):
    if _pandas_installed and isinstance(X, pd.DataFrame):
        # scikit-learn now wants a Pandas dataframe if the model was trained on a Pandas dataframe,
        # so convert it
        names = X.columns
        if class_idx >= 0:
            # classification
            def new_predict_fn(x):
                # TODO: at some point we should also handle column position remapping when the column names match
                X_fill = pd.DataFrame(x, columns=names)
                return predict_fn(X_fill)[:, class_idx]

            return new_predict_fn

        # regression
        def new_predict_fn(x):
            X_fill = pd.DataFrame(x, columns=names)
            return predict_fn(X_fill)

        return new_predict_fn
    if class_idx >= 0:
        # classification
        return lambda x: predict_fn(x)[:, class_idx]
    # regression
    return predict_fn
