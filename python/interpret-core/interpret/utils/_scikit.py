# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license


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
