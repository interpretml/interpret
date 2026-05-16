# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license


try:
    from sklearn.base import (
        BaseEstimator as _SKBaseEstimator,
        ClassifierMixin as _SKClassifierMixin,
        RegressorMixin as _SKRegressorMixin,
        TransformerMixin as _SKTransformerMixin,
    )
    from sklearn.exceptions import NotFittedError as _SKNotFittedError

except ImportError:

    class _SKNotFittedError(ValueError, AttributeError):  # type: ignore[no-redef]
        """Stub for sklearn.exceptions.NotFittedError when sklearn is not installed."""

        pass

    class _SKTargetTags:
        """Stub for sklearn TargetTags when sklearn is not installed."""

        pass

    class _SKClassifierTags:
        """Stub for sklearn ClassifierTags when sklearn is not installed."""

        pass

    class _SKRegressorTags:
        """Stub for sklearn RegressorTags when sklearn is not installed."""

        pass

    class _SKInputTags:
        """Stub for sklearn InputTags when sklearn is not installed."""

        pass

    class _SKTags:
        """Stub for sklearn Tags when sklearn is not installed."""

        def __init__(self):
            self.estimator_type = None
            self.target_tags = _SKTargetTags()
            self.classifier_tags = None
            self.regressor_tags = None
            self.input_tags = _SKInputTags()

    class _SKBaseEstimator:  # type: ignore[no-redef]
        """Stub for sklearn.base.BaseEstimator when sklearn is not installed."""

        def __sklearn_tags__(self):
            return _SKTags()

    class _SKClassifierMixin:  # type: ignore[no-redef]
        """Stub for sklearn.base.ClassifierMixin when sklearn is not installed."""

        def __sklearn_tags__(self):
            tags = super().__sklearn_tags__()
            tags.estimator_type = "classifier"
            tags.classifier_tags = _SKClassifierTags()
            return tags

    class _SKRegressorMixin:  # type: ignore[no-redef]
        """Stub for sklearn.base.RegressorMixin when sklearn is not installed."""

        def __sklearn_tags__(self):
            tags = super().__sklearn_tags__()
            tags.estimator_type = "regressor"
            tags.regressor_tags = _SKRegressorTags()
            return tags

    class _SKTransformerMixin:  # type: ignore[no-redef]
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
