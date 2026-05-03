# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy.typing as npt

from ..glassbox._ebm import BaseEBM, EBMClassifierMixin, EBMRegressorMixin


class DPEBMModel(BaseEBM):
    r"""Differentially Private Explainable Boosting Machine model.

    This is the instantiable mid-level class for differentially private EBMs.
    It carries all DP parameters with DP-appropriate defaults.
    Use DPEBMClassifier or DPEBMRegressor for classification or regression tasks.

    Parameters
    ----------
    feature_names : list of str, default=None
        List of feature names.
    feature_types : list of FeatureType, default=None

        List of feature types. For DP-EBMs, feature_types should be fully specified.
        The auto-detector, if used, examines the data and is not included in the privacy budget.
        If auto-detection is used, a privacy warning will be issued.
        FeatureType can be:

            - `'auto'`: Auto-detect (privacy budget is not respected!).
            - `'continuous'`: Use private continuous binning.
            - `[List of str]`: Ordinal categorical where the order has meaning. Eg: ["low", "medium", "high"]. Uses private categorical binning.
            - `'nominal'`: Categorical where the order has no meaning. Eg: country names. Uses private categorical binning.
    max_bins : int, default=32
        Max number of bins per feature.
    exclude : list of tuples of feature indices|names, default=None
        Features to be excluded.
    validation_size : int or float, default=0

        Validation set size. A validation set is needed if outer bags or error bars are desired.

            - Integer (1 <= validation_size): Count of samples to put in the validation sets
            - Percentage (validation_size < 1.0): Percentage of the data to put in the validation sets
            - 0: Outer bags have no utility and error bounds will be eliminated
    outer_bags : int, default=1
        Number of outer bags. Outer bags are used to generate error bounds and help with smoothing the graphs.
    learning_rate : float, default=0.01
        Learning rate for boosting.
    max_rounds : int, default=300
        Total number of boosting rounds with n_terms boosting steps per round.
    max_leaves : int, default=3
        Maximum number of leaves allowed in each tree.
    objective : str, default=None
        The objective to optimize. Restricted to "log_loss" or "rmse" for differentially private EBMs.
    n_jobs : int, default=-2
        Number of jobs to run in parallel. Negative integers are interpreted as following joblib's formula
        (n_cpus + 1 + n_jobs), just like scikit-learn. Eg: -2 means using all threads except 1.
    random_state : int or None, default=None
        Random state. None uses device_random and generates non-repeatable sequences.
        Should be set to 'None' for privacy, but can be set to an integer for testing and repeatability.
    epsilon: float, default=1.0
        Total privacy budget to be spent.
    delta: float, default=1e-5
        Additive component of differential privacy guarantee. Should be smaller than 1/n_training_samples.
    composition: {'gdp', 'classic'}, default='gdp'
        Method of tracking noise aggregation.
    bin_budget_frac: float, default=0.1
        Percentage of total epsilon budget to use for private binning.
    privacy_bounds: Union[np.ndarray, Mapping[Union[int, str], Tuple[float, float]]], default=None
        Specifies known min/max values for each feature.
        If None, DP-EBM shows a warning and uses the data to determine these values.

    """

    _is_differentially_private = True

    noise_scale_binning_: float
    noise_scale_boosting_: float

    def __init__(
        self,
        # Explainer
        feature_names: Sequence[None | str] | None = None,
        feature_types: Sequence[
            None
            | Literal[
                "auto",
                "continuous",
                "quantile",
                "rounded_quantile",
                "uniform",
                "winsorized",
                "nominal",
                "ordinal",
                "ignore",
                "nominal_prevalence",
                "nominal_alphabetical",
            ]
            | int
            | Sequence[str]
            | Sequence[float]
        ]
        | None = None,
        # Preprocessor
        max_bins: int = 32,
        # Stages
        exclude: Sequence[str | int | Sequence[str | int]] | None = None,
        # Ensemble
        validation_size: float | None = 0,
        outer_bags: int = 1,
        # Boosting
        learning_rate: float = 0.01,
        max_rounds: int | None = 300,
        # Trees
        max_leaves: int = 3,
        # Objective
        objective: str | None = None,
        # Overall
        n_jobs: int | None = -2,
        random_state: int | None = None,
        # Differential Privacy
        epsilon: float = 1.0,
        delta: float = 1e-5,
        composition: Literal["gdp", "classic"] = "gdp",
        bin_budget_frac: float = 0.1,
        privacy_bounds: npt.ArrayLike
        | Mapping[int | str, tuple[float, float]]
        | None = None,
    ):
        self.feature_names = feature_names
        self.feature_types = feature_types
        self.max_bins = max_bins
        self.exclude = exclude
        self.validation_size = validation_size
        self.outer_bags = outer_bags
        self.learning_rate = learning_rate
        self.max_rounds = max_rounds
        self.max_leaves = max_leaves
        self.objective = objective
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.epsilon = epsilon
        self.delta = delta
        self.composition = composition
        self.bin_budget_frac = bin_budget_frac
        self.privacy_bounds = privacy_bounds

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = False
        return tags


class DPEBMClassifier(EBMClassifierMixin, DPEBMModel):
    r"""Differentially Private Explainable Boosting Classifier.

    Note that many arguments are defaulted differently than regular EBMs.

    Parameters
    ----------
    feature_names : list of str, default=None
        List of feature names.
    feature_types : list of FeatureType, default=None

        List of feature types. For DP-EBMs, feature_types should be fully specified.
        The auto-detector, if used, examines the data and is not included in the privacy budget.
        If auto-detection is used, a privacy warning will be issued.
        FeatureType can be:

            - `'auto'`: Auto-detect (privacy budget is not respected!).
            - `'continuous'`: Use private continuous binning.
            - `[List of str]`: Ordinal categorical where the order has meaning. Eg: ["low", "medium", "high"]. Uses private categorical binning.
            - `'nominal'`: Categorical where the order has no meaning. Eg: country names. Uses private categorical binning.
    max_bins : int, default=32
        Max number of bins per feature.
    exclude : list of tuples of feature indices|names, default=None
        Features to be excluded.
    validation_size : int or float, default=0

        Validation set size. A validation set is needed if outer bags or error bars are desired.

            - Integer (1 <= validation_size): Count of samples to put in the validation sets
            - Percentage (validation_size < 1.0): Percentage of the data to put in the validation sets
            - 0: Outer bags have no utility and error bounds will be eliminated
    outer_bags : int, default=1
        Number of outer bags. Outer bags are used to generate error bounds and help with smoothing the graphs.
    learning_rate : float, default=0.01
        Learning rate for boosting.
    max_rounds : int, default=300
        Total number of boosting rounds with n_terms boosting steps per round.
    max_leaves : int, default=3
        Maximum number of leaves allowed in each tree.
    n_jobs : int, default=-2
        Number of jobs to run in parallel. Negative integers are interpreted as following joblib's formula
        (n_cpus + 1 + n_jobs), just like scikit-learn. Eg: -2 means using all threads except 1.
    random_state : int or None, default=None
        Random state. None uses device_random and generates non-repeatable sequences.
        Should be set to 'None' for privacy, but can be set to an integer for testing and repeatability.
    epsilon: float, default=1.0
        Total privacy budget to be spent.
    delta: float, default=1e-5
        Additive component of differential privacy guarantee. Should be smaller than 1/n_training_samples.
    composition: {'gdp', 'classic'}, default='gdp'
        Method of tracking noise aggregation.
    bin_budget_frac: float, default=0.1
        Percentage of total epsilon budget to use for private binning.
    privacy_bounds: Union[np.ndarray, Mapping[Union[int, str], Tuple[float, float]]], default=None
        Specifies known min/max values for each feature.
        If None, DP-EBM shows a warning and uses the data to determine these values.

    Attributes
    ----------
    classes\_ : array of bool, int, or unicode with shape ``(2,)``
        The class labels. DPEBMClassifier only supports binary classification, so there are 2 classes.
    n_features_in\_ : int
        Number of features.
    feature_names_in\_ : List of str
        Resolved feature names. Names can come from feature_names, X, or be auto-generated.
    feature_types_in\_ : List of str
        Resolved feature types. Can be: 'continuous', 'nominal', or 'ordinal'.
    bins\_ : List[Union[List[Dict[str, int]], List[array of float with shape ``(n_cuts,)``]]]
        Per-feature list that defines how to bin each feature. Each feature in the list contains
        a list of binning resolutions. The first item in the binning resolution list is for binning
        main effect features. If there are more items in the binning resolution list, they define the
        binning for successive levels of resolutions. The item at index 1, if it exists, defines the
        binning for pairs. The last binning resolution defines the bins for all successive interaction levels.
        If the binning resolution list contains dictionaries, then the feature is either a 'nominal' or
        'ordinal' categorical. If the binning resolution list contains arrays, then the feature is 'continuous'
        and the arrays will contain float cut points that separate continuous values into bins.
    feature_bounds\_ : array of float with shape ``(n_features, 2)``
        min/max bounds for each feature. feature_bounds_[feature_index, 0] is the min value of the feature
        and feature_bounds_[feature_index, 1] is the max value of the feature. Categoricals have min & max
        values of NaN.
    term_features\_ : List of tuples of feature indices
        Additive terms used in the model and their component feature indices.
    term_names\_ : List of str
        List of term names.
    bin_weights\_ : List of array of float with shape ``(n_bins)``
        Per-term list of the total sample weights in each term's bins.
    bagged_scores\_ : List of array of float with shape ``(n_outer_bags, n_bins)``
        Per-term list of the bagged model scores.
    term_scores\_ : List of array of float with shape ``(n_bins)``
        Per-term list of the model scores.
    standard_deviations\_ : List of array of float with shape ``(n_bins)``
        Per-term list of the standard deviations of the bagged model scores.
    link\_ : str
        Link function used to convert the predictions or targets into linear space
        additive scores and vice versa via the inverse link. Possible values include:
        "monoclassification", "custom_binary", "custom_ovr", "custom_multinomial",
        "mlogit", "vlogit", "logit", "probit", "cloglog", "loglog", "cauchit"
    link_param\_ : float
        Float value that can be used by the link function. For classification it is only used by "custom_classification".
    bag_weights\_ : array of float with shape ``(n_outer_bags,)``
        Per-bag record of the total weight within each bag.
    best_iteration\_ : array of int with shape ``(n_stages, n_outer_bags)``
        The number of boosting iterations performed within each stage until either early stopping, or the max_rounds was reached.
        Normally, the count of main effects boosting iterations will be in best_iteration_[0].
    intercept\_ : array of float with shape ``(1,)``
        Intercept of the model.
    bagged_intercept\_ : array of float with shape ``(n_outer_bags,)``
        Bagged intercept of the model.
    noise_scale_binning\_ : float
        The noise scale during binning.
    noise_scale_boosting\_ : float
        The noise scale during boosting.

    """

    def __init__(
        self,
        # Explainer
        feature_names: Sequence[None | str] | None = None,
        feature_types: Sequence[
            None
            | Literal[
                "auto",
                "continuous",
                "quantile",
                "rounded_quantile",
                "uniform",
                "winsorized",
                "nominal",
                "ordinal",
                "ignore",
                "nominal_prevalence",
                "nominal_alphabetical",
            ]
            | int
            | Sequence[str]
            | Sequence[float]
        ]
        | None = None,
        # Preprocessor
        max_bins: int = 32,
        # Stages
        exclude: Sequence[str | int | Sequence[str | int]] | None = None,
        # Ensemble
        validation_size: float | None = 0,
        outer_bags: int = 1,
        # Boosting
        learning_rate: float = 0.01,
        max_rounds: int | None = 300,
        # Trees
        max_leaves: int = 3,
        # Overall
        n_jobs: int | None = -2,
        random_state: int | None = None,
        # Differential Privacy
        epsilon: float = 1.0,
        delta: float = 1e-5,
        composition: Literal["gdp", "classic"] = "gdp",
        bin_budget_frac: float = 0.1,
        privacy_bounds: npt.ArrayLike
        | Mapping[int | str, tuple[float, float]]
        | None = None,
    ):
        super().__init__(
            feature_names=feature_names,
            feature_types=feature_types,
            max_bins=max_bins,
            exclude=exclude,
            validation_size=validation_size,
            outer_bags=outer_bags,
            learning_rate=learning_rate,
            max_rounds=max_rounds,
            max_leaves=max_leaves,
            objective="log_loss",
            n_jobs=n_jobs,
            random_state=random_state,
            epsilon=epsilon,
            delta=delta,
            composition=composition,
            bin_budget_frac=bin_budget_frac,
            privacy_bounds=privacy_bounds,
        )

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        return tags


class DPEBMRegressor(EBMRegressorMixin, DPEBMModel):
    r"""Differentially Private Explainable Boosting Regressor.

    Note that many arguments are defaulted differently than regular EBMs.

    Parameters
    ----------
    feature_names : list of str, default=None
        List of feature names.
    feature_types : list of FeatureType, default=None

        List of feature types. For DP-EBMs, feature_types should be fully specified.
        The auto-detector, if used, examines the data and is not included in the privacy budget.
        If auto-detection is used, a privacy warning will be issued.
        FeatureType can be:

            - `'auto'`: Auto-detect (privacy budget is not respected!).
            - `'continuous'`: Use private continuous binning.
            - `[List of str]`: Ordinal categorical where the order has meaning. Eg: ["low", "medium", "high"]. Uses private categorical binning.
            - `'nominal'`: Categorical where the order has no meaning. Eg: country names. Uses private categorical binning.
    max_bins : int, default=32
        Max number of bins per feature.
    exclude : list of tuples of feature indices|names, default=None
        Features to be excluded.
    validation_size : int or float, default=0

        Validation set size. A validation set is needed if outer bags or error bars are desired.

            - Integer (1 <= validation_size): Count of samples to put in the validation sets
            - Percentage (validation_size < 1.0): Percentage of the data to put in the validation sets
            - 0: Outer bags have no utility and error bounds will be eliminated
    outer_bags : int, default=1
        Number of outer bags. Outer bags are used to generate error bounds and help with smoothing the graphs.
    learning_rate : float, default=0.01
        Learning rate for boosting.
    max_rounds : int, default=300
        Total number of boosting rounds with n_terms boosting steps per round.
    max_leaves : int, default=3
        Maximum number of leaves allowed in each tree.
    n_jobs : int, default=-2
        Number of jobs to run in parallel. Negative integers are interpreted as following joblib's formula
        (n_cpus + 1 + n_jobs), just like scikit-learn. Eg: -2 means using all threads except 1.
    random_state : int or None, default=None
        Random state. None uses device_random and generates non-repeatable sequences.
        Should be set to 'None' for privacy, but can be set to an integer for testing and repeatability.
    epsilon: float, default=1.0
        Total privacy budget to be spent.
    delta: float, default=1e-5
        Additive component of differential privacy guarantee. Should be smaller than 1/n_training_samples.
    composition: {'gdp', 'classic'}, default='gdp'
        Method of tracking noise aggregation.
    bin_budget_frac: float, default=0.1
        Percentage of total epsilon budget to use for private binning.
    privacy_bounds: Union[np.ndarray, Mapping[Union[int, str], Tuple[float, float]]], default=None
        Specifies known min/max values for each feature.
        If None, DP-EBM shows a warning and uses the data to determine these values.
    privacy_target_min: float, default=None
        Known target minimum. 'y' values will be clipped to this min.
        If None, DP-EBM shows a warning and uses the data to determine this value.
    privacy_target_max: float, default=None
        Known target maximum. 'y' values will be clipped to this max.
        If None, DP-EBM shows a warning and uses the data to determine this value.

    Attributes
    ----------
    n_features_in\_ : int
        Number of features.
    feature_names_in\_ : List of str
        Resolved feature names. Names can come from feature_names, X, or be auto-generated.
    feature_types_in\_ : List of str
        Resolved feature types. Can be: 'continuous', 'nominal', or 'ordinal'.
    bins\_ : List[Union[List[Dict[str, int]], List[array of float with shape ``(n_cuts,)``]]]
        Per-feature list that defines how to bin each feature. Each feature in the list contains
        a list of binning resolutions. The first item in the binning resolution list is for binning
        main effect features. If there are more items in the binning resolution list, they define the
        binning for successive levels of resolutions. The item at index 1, if it exists, defines the
        binning for pairs. The last binning resolution defines the bins for all successive interaction levels.
        If the binning resolution list contains dictionaries, then the feature is either a 'nominal' or
        'ordinal' categorical. If the binning resolution list contains arrays, then the feature is 'continuous'
        and the arrays will contain float cut points that separate continuous values into bins.
    feature_bounds\_ : array of float with shape ``(n_features, 2)``
        min/max bounds for each feature. feature_bounds_[feature_index, 0] is the min value of the feature
        and feature_bounds_[feature_index, 1] is the max value of the feature. Categoricals have min & max
        values of NaN.
    term_features\_ : List of tuples of feature indices
        Additive terms used in the model and their component feature indices.
    term_names\_ : List of str
        List of term names.
    bin_weights\_ : List of array of float with shape ``(n_bins)``
        Per-term list of the total sample weights in each term's bins.
    bagged_scores\_ : List of array of float with shape ``(n_outer_bags, n_bins)``
        Per-term list of the bagged model scores.
    term_scores\_ : List of array of float with shape ``(n_bins)``
        Per-term list of the model scores.
    standard_deviations\_ : List of array of float with shape ``(n_bins)``
        Per-term list of the standard deviations of the bagged model scores.
    link\_ : str
        Link function used to convert the predictions or targets into linear space
        additive scores and vice versa via the inverse link. Possible values include:
        "custom_regression", "power", "identity", "log", "inverse", "inverse_square", "sqrt"
    link_param\_ : float
        Float value that can be used by the link function. The primary use is for the power link.
    bag_weights\_ : array of float with shape ``(n_outer_bags,)``
        Per-bag record of the total weight within each bag.
    best_iteration\_ : array of int with shape ``(n_stages, n_outer_bags)``
        The number of boosting iterations performed within each stage until either early stopping, or the max_rounds was reached.
        Normally, the count of main effects boosting iterations will be in best_iteration_[0].
    intercept\_ : float
        Intercept of the model.
    bagged_intercept\_ : array of float with shape ``(n_outer_bags,)``
        Bagged intercept of the model.
    min_target\_ : float
        The minimum value found in 'y', or privacy_target_min if provided.
    max_target\_ : float
        The maximum value found in 'y', or privacy_target_max if provided.
    noise_scale_binning\_ : float
        The noise scale during binning.
    noise_scale_boosting\_ : float
        The noise scale during boosting.

    """

    def __init__(
        self,
        # Explainer
        feature_names: Sequence[None | str] | None = None,
        feature_types: Sequence[
            None
            | Literal[
                "auto",
                "continuous",
                "quantile",
                "rounded_quantile",
                "uniform",
                "winsorized",
                "nominal",
                "ordinal",
                "ignore",
                "nominal_prevalence",
                "nominal_alphabetical",
            ]
            | int
            | Sequence[str]
            | Sequence[float]
        ]
        | None = None,
        # Preprocessor
        max_bins: int = 32,
        # Stages
        exclude: Sequence[str | int | Sequence[str | int]] | None = None,
        # Ensemble
        validation_size: float | None = 0,
        outer_bags: int = 1,
        # Boosting
        learning_rate: float = 0.01,
        max_rounds: int | None = 300,
        # Trees
        max_leaves: int = 3,
        # Overall
        n_jobs: int | None = -2,
        random_state: int | None = None,
        # Differential Privacy
        epsilon: float = 1.0,
        delta: float = 1e-5,
        composition: Literal["gdp", "classic"] = "gdp",
        bin_budget_frac: float = 0.1,
        privacy_bounds: npt.ArrayLike
        | Mapping[int | str, tuple[float, float]]
        | None = None,
        privacy_target_min: float | None = None,
        privacy_target_max: float | None = None,
    ):
        super().__init__(
            feature_names=feature_names,
            feature_types=feature_types,
            max_bins=max_bins,
            exclude=exclude,
            validation_size=validation_size,
            outer_bags=outer_bags,
            learning_rate=learning_rate,
            max_rounds=max_rounds,
            max_leaves=max_leaves,
            objective="rmse",
            n_jobs=n_jobs,
            random_state=random_state,
            epsilon=epsilon,
            delta=delta,
            composition=composition,
            bin_budget_frac=bin_budget_frac,
            privacy_bounds=privacy_bounds,
        )
        self.privacy_target_min = privacy_target_min
        self.privacy_target_max = privacy_target_max

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.regressor_tags.poor_score = True
        return tags


# TODO: Deprecate the old names with a warning
DPExplainableBoostingClassifier = DPEBMClassifier
DPExplainableBoostingRegressor = DPEBMRegressor
