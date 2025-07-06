# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

from itertools import chain, count, repeat

import numpy as np
import pytest
from interpret.utils._clean_simple import clean_dimensions, typify_classification
from interpret.utils._clean_x import preclean_X
from interpret.utils._compressed_dataset import bin_native, bin_native_by_dimension
from interpret.utils._preprocessor import construct_bins
from interpret.utils._shared_dataset import SharedDataset

@pytest.mark.skip(reason="skip this until we have support for missing values")
def test_bin_native():
    X = np.array(
        [["a", 1, np.nan], ["b", 2, 7], ["a", 2, 8], [None, 3, 9]], dtype=np.object_
    )
    feature_names_given = ["f1", 99, "f3"]
    feature_types_given = ["nominal", "nominal", "continuous"]
    y = np.array(["a", 99, 99, "b"])
    sample_weight = np.array([0.5, 1.1, 0.1, 0.5])

    y = clean_dimensions(y, "y")
    assert y.ndim == 1

    y = typify_classification(y)
    classes, y = np.unique(y, return_inverse=True)
    n_classes = len(classes)

    sample_weight = clean_dimensions(sample_weight, "sample_weight")
    assert sample_weight.ndim == 1
    sample_weight = sample_weight.astype(np.float64, copy=False)

    X, n_samples = preclean_X(X, None, None)

    (
        feature_names_in,
        feature_types_in,
        bins,
        bin_weights,
        feature_bounds,
        histogram_weights,
        missing_val_counts,
        unique_val_counts,
        noise_scale_binning,
    ) = construct_bins(
        X, y, sample_weight, feature_names_given, feature_types_given, [256, 5, 3]
    )
    assert feature_names_in is not None
    assert feature_types_in is not None
    assert bins is not None
    assert bin_weights is not None
    assert feature_bounds is not None
    assert histogram_weights is not None
    assert missing_val_counts is not None
    assert unique_val_counts is not None

    feature_idxs_origin = (
        range(len(feature_names_given))
        if feature_types_given is None
        else [
            feature_idx
            for feature_idx, feature_type in zip(count(), feature_types_given)
            if feature_type != "ignore"
        ]
    )
    feature_idxs = []
    bins_iter = []
    for feature_idx, n_dimensions in chain(
        zip(feature_idxs_origin, repeat(1)),
        zip(feature_idxs_origin, repeat(2)),
        zip(feature_idxs_origin, repeat(3)),
    ):
        bin_levels = bins[feature_idx]
        feature_bins = bin_levels[
            -1 if len(bin_levels) < n_dimensions else n_dimensions - 1
        ]
        feature_idxs.append(feature_idx)
        bins_iter.append(feature_bins)

    with SharedDataset() as shared:
        bin_native(
            n_classes,
            feature_idxs,
            bins_iter,
            X,
            y,
            sample_weight,
            feature_names_in,
            feature_types_in,
            shared,
        )
        assert shared.shared_memory is not None
        assert shared.dataset is not None
        assert shared.name is not None

    with SharedDataset() as shared:
        bin_native_by_dimension(
            n_classes, 1, bins, X, y, sample_weight, feature_names_in, feature_types_in,
                shared,
        )
        assert shared.shared_memory is not None
        assert shared.dataset is not None
        assert shared.name is not None
