# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license


try:
    from sklearn.exceptions import NotFittedError as _NotFittedError
except ImportError:

    class _NotFittedError(ValueError, AttributeError):
        """Stub for sklearn.exceptions.NotFittedError when sklearn is not installed."""

        pass


def _is_classifier(estimator) -> bool:
    """Check if an estimator is a classifier."""

    try:
        tags = estimator.__sklearn_tags__()
    except AttributeError:
        return False

    return tags.estimator_type == "classifier"


def _is_regressor(estimator) -> bool:
    """Check if an estimator is a regressor."""

    try:
        tags = estimator.__sklearn_tags__()
    except AttributeError:
        return False

    return tags.estimator_type == "regressor"
