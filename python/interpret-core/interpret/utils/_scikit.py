# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license


try:
    from sklearn.base import (
        BaseEstimator as SKBaseEstimator,
        ClassifierMixin as SKClassifierMixin,
        RegressorMixin as SKRegressorMixin,
        TransformerMixin as SKTransformerMixin,
    )
    from sklearn.exceptions import NotFittedError as SKNotFittedError

except ImportError:

    class SKNotFittedError(ValueError, AttributeError):
        """Stub for sklearn.exceptions.NotFittedError when sklearn is not installed."""

        pass

    class SKTargetTags:
        """Stub for sklearn TargetTags when sklearn is not installed."""

        pass

    class SKClassifierTags:
        """Stub for sklearn ClassifierTags when sklearn is not installed."""

        pass

    class SKRegressorTags:
        """Stub for sklearn RegressorTags when sklearn is not installed."""

        pass

    class SKInputTags:
        """Stub for sklearn InputTags when sklearn is not installed."""

        pass

    class SKTags:
        """Stub for sklearn Tags when sklearn is not installed."""

        def __init__(self):
            self.estimator_type = None
            self.target_tags = SKTargetTags()
            self.classifier_tags = None
            self.regressor_tags = None
            self.input_tags = SKInputTags()

    class SKBaseEstimator:
        """Stub for sklearn.base.BaseEstimator when sklearn is not installed."""

        def __sklearn_tags__(self):
            return SKTags()

    class SKClassifierMixin:
        """Stub for sklearn.base.ClassifierMixin when sklearn is not installed."""

        def __sklearn_tags__(self):
            tags = super().__sklearn_tags__()
            tags.estimator_type = "classifier"
            tags.classifier_tags = SKClassifierTags()
            return tags

    class SKRegressorMixin:
        """Stub for sklearn.base.RegressorMixin when sklearn is not installed."""

        def __sklearn_tags__(self):
            tags = super().__sklearn_tags__()
            tags.estimator_type = "regressor"
            tags.regressor_tags = SKRegressorTags()
            return tags

    class SKTransformerMixin:
        """Stub for sklearn.base.TransformerMixin when sklearn is not installed."""

        def fit_transform(self, X, y=None, **fit_params):
            return self.fit(X, y, **fit_params).transform(X)


def _is_classifier(estimator) -> bool:
    """Check if an estimator is a classifier.

    Uses the documented __sklearn_tags__ API. Returns False for non-estimator
    objects (e.g. None, numpy arrays) instead of raising.
    """

    tags_fn = getattr(estimator, "__sklearn_tags__", None)
    return False if tags_fn is None else tags_fn().estimator_type == "classifier"


def _is_regressor(estimator) -> bool:
    """Check if an estimator is a regressor.

    Uses the documented __sklearn_tags__ API. Returns False for non-estimator
    objects (e.g. None, numpy arrays) instead of raising.
    """

    tags_fn = getattr(estimator, "__sklearn_tags__", None)
    return False if tags_fn is None else tags_fn().estimator_type == "regressor"
