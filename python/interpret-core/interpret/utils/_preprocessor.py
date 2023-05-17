# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

import logging
import math
from itertools import count, repeat, groupby
from warnings import warn

import numpy as np
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
)
from sklearn.utils.validation import check_is_fitted

from ._clean_x import unify_columns, preclean_X, unify_feature_names
from ._clean_simple import clean_dimensions
from ._seed import normalize_initial_seed, increment_seed

from ._native import Native
from ._privacy import (
    validate_eps_delta,
    calc_classic_noise_multi,
    calc_gdp_noise_multi,
    private_numeric_binning,
    private_categorical_binning,
)

_log = logging.getLogger(__name__)
_none_list = [None]


def _cut_continuous(native, X_col, processing, binning, max_bins, min_samples_bin):
    # called under: fit

    if (
        processing != "quantile"
        and processing != "rounded_quantile"
        and processing != "uniform"
        and processing != "winsorized"
        and not isinstance(processing, list)
        and not isinstance(processing, np.ndarray)
    ):
        if isinstance(binning, list) or isinstance(binning, np.ndarray):
            msg = f"illegal binning type {binning}"
            _log.error(msg)
            raise ValueError(msg)
        processing = binning

    if processing == "quantile":
        # one bin for missing, one bin for unknown, and # of cuts is one less again
        cuts = native.cut_quantile(X_col, min_samples_bin, 0, max_bins - 3)
    elif processing == "rounded_quantile":
        # one bin for missing, one bin for unknown, and # of cuts is one less again
        cuts = native.cut_quantile(X_col, min_samples_bin, 1, max_bins - 3)
    elif processing == "uniform":
        # one bin for missing, one bin for unknown, and # of cuts is one less again
        cuts = native.cut_uniform(X_col, max_bins - 3)
    elif processing == "winsorized":
        # one bin for missing, one bin for unknown, and # of cuts is one less again
        cuts = native.cut_winsorized(X_col, max_bins - 3)
    elif isinstance(processing, np.ndarray):
        cuts = processing.astype(dtype=np.float64, copy=False)
    elif isinstance(processing, list):
        cuts = np.array(processing, np.float64)
    else:
        msg = f"illegal binning type {processing}"
        _log.error(msg)
        raise ValueError(msg)

    return cuts


class EBMPreprocessor(BaseEstimator, TransformerMixin):
    """Transformer that preprocesses data to be ready before EBM."""

    def __init__(
        self,
        feature_names=None,
        feature_types=None,
        max_bins=256,
        binning="quantile",
        min_samples_bin=1,
        min_unique_continuous=0,
        random_state=None,
        epsilon=None,
        delta=None,
        composition=None,
        privacy_bounds=None,
    ):
        """Initializes EBM preprocessor.

        Args:
            feature_names: Feature names as list.
            feature_types: Feature types as list, for example "continuous" or "nominal".
            max_bins: Max number of bins to process numeric features.
            binning: Strategy to compute bins: "quantile", "rounded_quantile", "uniform", or "private".
            min_samples_bin: minimum number of samples to put into a quantile or rounded_quantile bin
            min_unique_continuous: number of unique numbers required before a feature is considered continuous
            random_state: Random state.
            epsilon: Privacy budget parameter. Only applicable when binning is "private".
            delta: Privacy budget parameter. Only applicable when binning is "private".
            composition: Method of tracking noise aggregation. Must be one of 'classic' or 'gdp'.
            privacy_bounds: User specified min/max values for numeric features. Only applicable when binning is "private".
        """
        self.feature_names = feature_names
        self.feature_types = feature_types
        self.max_bins = max_bins
        self.binning = binning
        self.min_samples_bin = min_samples_bin
        self.min_unique_continuous = min_unique_continuous
        self.random_state = random_state
        self.epsilon = epsilon
        self.delta = delta
        self.composition = composition
        self.privacy_bounds = privacy_bounds

    def fit(self, X, y=None, sample_weight=None):
        """Fits transformer to provided samples.

        Args:
            X: Numpy array for training samples.
            y: Unused. Only included for scikit-learn compatibility
            sample_weight: Per-sample weights

        Returns:
            Itself.
        """

        n_samples = None
        if y is not None:
            y = clean_dimensions(y, "y")
            if y.ndim != 1:
                raise ValueError("y must be 1 dimensional")
            n_samples = len(y)

        if sample_weight is not None:
            sample_weight = clean_dimensions(sample_weight, "sample_weight")
            if sample_weight.ndim != 1:
                raise ValueError("sample_weight must be 1 dimensional")
            if n_samples is not None and n_samples != len(sample_weight):
                msg = f"y has {n_samples} samples and sample_weight has {len(sample_weight)} samples"
                _log.error(msg)
                raise ValueError(msg)
            n_samples = len(sample_weight)
            sample_weight = sample_weight.astype(np.float64, copy=False)

            # NaN values are guaranteed to be the min if they exist
            min_weight = sample_weight.min()
            if math.isnan(min_weight):
                msg = "illegal NaN sample_weight value"
                _log.error(msg)
                raise ValueError(msg)
            if math.isinf(sample_weight.max()):
                msg = "illegal +infinity sample_weight value"
                _log.error(msg)
                raise ValueError(msg)
            if min_weight < 0:
                msg = "illegal negative sample_weight value"
                _log.error(msg)
                raise ValueError(msg)
            if min_weight == 0:
                # TODO: for now weights of zero are illegal, but in the future accept them
                msg = "illegal sample_weight value of zero"
                _log.error(msg)
                raise ValueError(msg)

        X, n_samples = preclean_X(
            X,
            self.feature_names,
            self.feature_types,
            n_samples,
            "sample_weight" if y is None else "y",
        )

        # TODO: should preprocessors handle 0 samples?
        if n_samples == 0:
            msg = "X has 0 samples"
            _log.error(msg)
            raise ValueError(msg)

        feature_names_in = unify_feature_names(
            X, self.feature_names, self.feature_types
        )
        n_features = len(feature_names_in)

        noise_scale = None  # only applicable for private binning
        if self.binning == "private":
            validate_eps_delta(self.epsilon, self.delta)
            max_weight = 1 if sample_weight is None else np.max(sample_weight)
            if self.composition == "classic":
                noise_scale = calc_classic_noise_multi(
                    total_queries=n_features,
                    target_epsilon=self.epsilon,
                    delta=self.delta,
                    sensitivity=max_weight,
                )
            elif self.composition == "gdp":
                # Alg Line 17"
                noise_scale = (
                    calc_gdp_noise_multi(
                        total_queries=n_features,
                        target_epsilon=self.epsilon,
                        delta=self.delta,
                    )
                    * max_weight
                )
            else:
                raise NotImplementedError(
                    f"Unknown composition method provided: {self.composition}. Please use 'gdp' or 'classic'."
                )

        feature_types_in = _none_list * n_features
        bins = _none_list * n_features
        bin_weights = _none_list * n_features
        feature_bounds = np.full((n_features, 2), np.nan, dtype=np.float64)
        histogram_weights = _none_list * n_features
        missing_val_counts = np.zeros(n_features, dtype=np.int64)
        unique_val_counts = np.zeros(n_features, dtype=np.int64)

        native = Native.get_native_singleton()
        rng = native.create_rng(normalize_initial_seed(self.random_state))
        is_privacy_bounds_warning = False
        is_privacy_types_warning = False
        for feature_idx, (feature_type_in, X_col, categories, bad) in enumerate(
            unify_columns(
                X,
                zip(range(n_features), repeat(None)),
                feature_names_in,
                self.feature_types,
                self.min_unique_continuous,
                False,
            )
        ):
            if n_samples != len(X_col):
                msg = "The columns of X are mismatched in the number of of samples"
                _log.error(msg)
                raise ValueError(msg)

            max_bins = self.max_bins  # TODO: in the future allow this to be per-feature
            if max_bins < 3:
                raise ValueError(
                    f"max_bins was {max_bins}, but must be 3 or higher. One bin for missing, one bin for unknown, and one or more bins for the non-missing values."
                )

            if not X_col.flags.c_contiguous:
                # X_col could be a slice that has a stride.  We need contiguous for caling into C
                X_col = X_col.copy()

            feature_types_in[feature_idx] = feature_type_in
            feature_type_given = (
                None if self.feature_types is None else self.feature_types[feature_idx]
            )
            if categories is None:
                # continuous feature
                if bad is not None:
                    msg = f"Feature {feature_names_in[feature_idx]} is indicated as continuous, but has non-numeric data"
                    _log.error(msg)
                    raise ValueError(msg)

                if self.binning == "private":
                    if np.isnan(X_col).any():
                        msg = "missing values in X not supported for private binning"
                        _log.error(msg)
                        raise ValueError(msg)

                    if feature_type_given != "continuous":
                        is_privacy_types_warning = True

                    min_feature_val = np.nan
                    max_feature_val = np.nan
                    if self.privacy_bounds is not None:
                        if isinstance(self.privacy_bounds, dict):
                            # TODO: check for names/indexes in the dict that are not
                            # in feature_names_in_ or out of bounds, or duplicate
                            # int vs names situations
                            bounds = self.privacy_bounds.get(feature_idx, None)
                            if bounds is None:
                                feature_name = feature_names_in[feature_idx]
                                bounds = self.privacy_bounds.get(feature_name, None)

                            if bounds is not None:
                                min_feature_val = bounds[0]
                                max_feature_val = bounds[1]
                        else:
                            # TODO: do some sanity checking on the shape of privacy_bounds
                            bounds = self.privacy_bounds[feature_idx]
                            min_feature_val = bounds[0]
                            max_feature_val = bounds[1]

                    if math.isnan(min_feature_val):
                        is_privacy_bounds_warning = True
                        min_feature_val = np.nanmin(X_col)

                    if math.isnan(max_feature_val):
                        is_privacy_bounds_warning = True
                        max_feature_val = np.nanmax(X_col)

                    cuts, feature_bin_weights = private_numeric_binning(
                        X_col,
                        sample_weight,
                        noise_scale,
                        max_bins - 1,
                        min_feature_val,
                        max_feature_val,
                        rng,
                    )
                    feature_bin_weights.append(0)
                    feature_bin_weights = np.array(feature_bin_weights, np.float64)
                else:
                    min_feature_val = np.nanmin(X_col)
                    max_feature_val = np.nanmax(X_col)
                    cuts = _cut_continuous(
                        native,
                        X_col,
                        feature_type_given,
                        self.binning,
                        max_bins,
                        self.min_samples_bin,
                    )
                    bin_indexes = native.discretize(X_col, cuts)
                    feature_bin_weights = np.bincount(
                        bin_indexes, weights=sample_weight, minlength=len(cuts) + 3
                    )
                    feature_bin_weights = feature_bin_weights.astype(
                        np.float64, copy=False
                    )

                    n_cuts = native.get_histogram_cut_count(X_col)
                    histogram_cuts = native.cut_uniform(X_col, n_cuts)
                    bin_indexes = native.discretize(X_col, histogram_cuts)
                    feature_histogram_weights = np.bincount(
                        bin_indexes,
                        weights=sample_weight,
                        minlength=len(histogram_cuts) + 3,
                    )
                    feature_histogram_weights = feature_histogram_weights.astype(
                        np.float64, copy=False
                    )

                    histogram_weights[feature_idx] = feature_histogram_weights

                    n_missing = len(X_col)
                    X_col = X_col[~np.isnan(X_col)]
                    n_missing = n_missing - len(X_col)
                    missing_val_counts.itemset(feature_idx, n_missing)
                    unique_val_counts.itemset(feature_idx, len(np.unique(X_col)))

                bins[feature_idx] = cuts
                feature_bounds.itemset((feature_idx, 0), min_feature_val)
                feature_bounds.itemset((feature_idx, 1), max_feature_val)
            else:
                # categorical feature
                if bad is not None:
                    msg = f"Feature {feature_names_in[feature_idx]} has unrecognized ordinal values"
                    _log.error(msg)
                    raise ValueError(msg)

                if self.binning == "private":
                    if np.count_nonzero(X_col) != len(X_col):
                        msg = "missing values in X not supported for private binning"
                        _log.error(msg)
                        raise ValueError(msg)

                    if feature_type_given is None:
                        # if auto-detected then we need to show a privacy warning
                        is_privacy_types_warning = True

                    # TODO: clean up this hack that uses strings of the indexes
                    keep_bins, old_feature_bin_weights = private_categorical_binning(
                        X_col, sample_weight, noise_scale, max_bins - 1, rng
                    )
                    unknown_weight = 0
                    if keep_bins[-1] == "DPOther":
                        unknown_weight = old_feature_bin_weights[-1]
                        keep_bins = keep_bins[:-1]
                        old_feature_bin_weights = old_feature_bin_weights[:-1]

                    keep_bins = keep_bins.astype(np.int64)
                    keep_bins = dict(zip(keep_bins, old_feature_bin_weights))

                    feature_bin_weights = np.empty(len(keep_bins) + 2, np.float64)
                    feature_bin_weights[0] = 0
                    feature_bin_weights[-1] = unknown_weight

                    categories = list(map(tuple, map(reversed, categories.items())))
                    categories.sort()  # groupby requires sorted data

                    new_categories = {}
                    new_idx = 1
                    for idx, category_iter in groupby(categories, lambda x: x[0]):
                        bin_weight = keep_bins.get(idx, None)
                        if bin_weight is not None:
                            feature_bin_weights.itemset(new_idx, bin_weight)
                            for _, category in category_iter:
                                new_categories[category] = new_idx
                            new_idx += 1

                    categories = new_categories
                else:
                    n_unique_indexes = (
                        0 if len(categories) == 0 else max(categories.values())
                    )
                    feature_bin_weights = np.bincount(
                        X_col, weights=sample_weight, minlength=n_unique_indexes + 2
                    )
                    feature_bin_weights = feature_bin_weights.astype(
                        np.float64, copy=False
                    )

                    # for categoricals histograms and bin weights are the same
                    histogram_weights[feature_idx] = feature_bin_weights

                    missing_val_counts.itemset(
                        feature_idx, len(X_col) - np.count_nonzero(X_col)
                    )
                    unique_val_counts.itemset(feature_idx, len(categories))
                bins[feature_idx] = categories
            bin_weights[feature_idx] = feature_bin_weights

        if is_privacy_bounds_warning:
            warn(
                "Possible privacy violation: assuming min/max values per feature are public info. "
                "Pass in privacy_bounds with known public ranges per feature to avoid this warning."
            )
        if is_privacy_types_warning:
            warn(
                "Possible privacy violation: Automatic determination of the feature"
                "types examines the data and is unaccounted for in the privacy budget. "
                "Pass in fully specified feature_types of 'continuous', 'nominal', "
                "'ordinal', or a list of strings to avoid this warning."
            )

        self.feature_names_in_ = feature_names_in
        self.feature_types_in_ = feature_types_in
        self.bins_ = bins
        self.bin_weights_ = bin_weights
        self.feature_bounds_ = feature_bounds
        self.histogram_weights_ = histogram_weights
        self.missing_val_counts_ = missing_val_counts
        self.unique_val_counts_ = unique_val_counts
        self.noise_scale_ = noise_scale
        self.has_fitted_ = True
        return self

    def transform(self, X):
        """Transform on provided samples.

        Args:
            X: Numpy array for samples.

        Returns:
            Transformed numpy array.
        """
        check_is_fitted(self, "has_fitted_")

        X, n_samples = preclean_X(X, self.feature_names_in_, self.feature_types_in_)

        X_binned = np.empty(
            (n_samples, len(self.feature_names_in_)), np.int64, order="F"
        )

        if 0 < n_samples:
            native = Native.get_native_singleton()
            category_iter = (
                category if isinstance(category, dict) else None
                for category in self.bins_
            )
            requests = zip(count(), category_iter)
            cols = unify_columns(
                X, requests, self.feature_names_in_, self.feature_types_in_, None, False
            )
            for feature_idx, bins, (_, X_col, _, _) in zip(count(), self.bins_, cols):
                if n_samples != len(X_col):
                    msg = "The columns of X are mismatched in the number of of samples"
                    _log.error(msg)
                    raise ValueError(msg)

                if not isinstance(bins, dict):
                    # continuous feature

                    if not X_col.flags.c_contiguous:
                        # X_col could be a slice that has a stride.  We need contiguous for caling into C
                        X_col = X_col.copy()

                    X_col = native.discretize(X_col, bins)

                X_binned[:, feature_idx] = X_col

        return X_binned

    def fit_transform(self, X, y=None, sample_weight=None):
        """Fits and Transform on provided samples.

        Args:
            X: Numpy array for samples.
            y: Unused. Only included for scikit-learn compatibility
            sample_weight: Per-sample weights

        Returns:
            Transformed numpy array.
        """

        n_samples = None
        if y is not None:
            y = clean_dimensions(y, "y")
            if y.ndim != 1:
                raise ValueError("y must be 1 dimensional")
            n_samples = len(y)

        if sample_weight is not None:
            sample_weight = clean_dimensions(sample_weight, "sample_weight")
            if sample_weight.ndim != 1:
                raise ValueError("sample_weight must be 1 dimensional")
            if n_samples is not None and n_samples != len(sample_weight):
                msg = f"y has {n_samples} samples and sample_weight has {len(sample_weight)} samples"
                _log.error(msg)
                raise ValueError(msg)
            n_samples = len(sample_weight)
            sample_weight = sample_weight.astype(np.float64, copy=False)

        # materialize any iterators first
        X, _ = preclean_X(X, self.feature_names, self.feature_types, n_samples)
        return self.fit(X, y, sample_weight).transform(X)


def construct_bins(
    X,
    y,
    sample_weight,
    feature_names_given,
    feature_types_given,
    max_bins_leveled,
    binning="quantile",
    min_samples_bin=1,
    min_unique_continuous=0,
    random_state=None,
    epsilon=None,
    delta=None,
    composition=None,
    privacy_bounds=None,
):
    is_mains = True
    for max_bins in max_bins_leveled:
        preprocessor = EBMPreprocessor(
            feature_names_given,
            feature_types_given,
            max_bins,
            binning,
            min_samples_bin,
            min_unique_continuous,
            random_state,
            epsilon,
            delta,
            composition,
            privacy_bounds,
        )

        random_state = increment_seed(random_state)

        preprocessor.fit(X, y, sample_weight)
        if is_mains:
            is_mains = False
            bins = preprocessor.bins_
            for feature_idx in range(len(bins)):
                bins[feature_idx] = [bins[feature_idx]]

            feature_names_in = preprocessor.feature_names_in_
            feature_types_in = preprocessor.feature_types_in_
            bin_weights = preprocessor.bin_weights_
            feature_bounds = preprocessor.feature_bounds_
            histogram_weights = preprocessor.histogram_weights_
            missing_val_counts = preprocessor.missing_val_counts_
            unique_val_counts = preprocessor.unique_val_counts_
            noise_scale = preprocessor.noise_scale_
        else:
            if feature_names_in != preprocessor.feature_names_in_:
                raise RuntimeError("Mismatched feature_names")
            if feature_types_in != preprocessor.feature_types_in_:
                raise RuntimeError("Mismatched feature_types")
            if len(bins) != len(preprocessor.bins_):
                raise RuntimeError("Mismatched bin lengths")

            for bin_levels, feature_bins in zip(bins, preprocessor.bins_):
                bin_levels.append(feature_bins)

    return (
        feature_names_in,
        feature_types_in,
        bins,
        bin_weights,
        feature_bounds,
        histogram_weights,
        missing_val_counts,
        unique_val_counts,
        noise_scale,
    )
