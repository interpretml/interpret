# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

import numpy as np
from itertools import takewhile
import operator


def synthetic_default(
    classes=["class_0", "class_1"],
    n_samples=10000,
    missing=False,
    objects=True,
    seed=None,
    base_shift=0.0,
    noise_scale=0.25,
    categorical_digits=3,
    clip_low=-2.0,
    clip_high=2.0,
):
    if isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)

    X_orig, names, types = _synthetic_features_default(
        n_samples, missing, objects, rng, categorical_digits, clip_low, clip_high
    )

    X_imp = _normalize_categoricals(X_orig, types, clip_low, clip_high)

    # impute missing values with 0
    missings = np.isnan(X_imp)
    X_imp[missings] = 0.0

    # create some additive terms for our model to find
    # our additions for y below have a bias of about +3.8, so shift the default by
    # -4.0 to get us close to zero and the base shift is for anything away from zero
    base_shift -= 1.0
    y = rng.normal(base_shift, noise_scale, n_samples)
    y += np.sin(3.14 * 2.0 / (clip_high - clip_low) * X_imp[:, 0]) * 0.9375
    y += np.cos(3.14 * 2.0 / (clip_high - clip_low) * X_imp[:, 1]) * 0.9375
    y += np.exp(X_imp[:, 2]) * 0.125
    y += X_imp[:, 3] * 0.375
    y += -X_imp[:, 4] * 0.375
    y += X_imp[:, 5] ** 2 * 0.375
    y += X_imp[:, 6] ** 3 * 0.09375

    # linear addition of low cardinality categorical
    y += X_imp[:, -2] * 0.375

    # linear addition of high cardinality categorical
    y += X_imp[:, -1] * 0.375

    # pairs
    y += X_imp[:, 0] * X_imp[:, 1] * 0.1875
    y += X_imp[:, 0] * X_imp[:, 2] * 0.125

    # 3-way interaction
    y += X_imp[:, 0] * X_imp[:, 1] * X_imp[:, 2] * 0.125

    if classes is not None and classes != 0:
        if type(classes) == int:
            classes = np.arange(classes)
        else:
            classes = np.array(classes)

        n_classes = len(classes)
        if n_classes == 1:
            y = np.full(n_samples, 0, dtype=int)
        else:
            prob = np.exp(y)
            prob = prob / (1.0 + prob)

            y = (rng.uniform(0.0, 1.0, n_samples) < prob).astype(int)

            if 2 < n_classes:
                # multiclass. To keep things simple randomly select classes above 0th
                y = np.where(y == 0, 0, rng.integers(1, n_classes, n_samples))

        y = classes[y]

    return (X_orig, y, names, types)


def _synthetic_features_default(
    n_samples, missing, objects, seed, categorical_digits, clip_low, clip_high
):
    # EBMs are blind to the scale of the feature values since features are binned using
    # quantiles, so we can use whatever scale we want for convenience. We choose the
    # scale of the feature values to make generating the synthetic y easier by roughly
    # keeping the same scale for all features.
    # Each feature is roughly set such that the average of the negative values is -1.0
    # and the average of the positive values is 1.0. This allows us to have a common
    # scale with integers where we have 5 categories from -2 to +2, and we clip at
    # -2.0 and +2.0 to make transformations like exp(x) and x**3 not too extreme.

    if isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)

    clip_low_int = int(np.ceil(clip_low))
    clip_high_int = int(np.floor(clip_high))

    names = []
    types = []
    features = []

    # Feature 0 - Continuous drawn from uniform distribution
    names.append("f0_uniform")
    types.append("continuous")
    features.append(rng.uniform(clip_low, clip_high, n_samples))

    # Feature 1 - Continuous drawn from normal distribution
    names.append("f1_normal")
    types.append("continuous")
    features.append(np.clip(rng.normal(0.0, 1.375, n_samples), clip_low, clip_high))

    # Feature 2 - Continuous time between events with avg time between events of 1.4375
    names.append("f2_exponential")
    types.append("continuous")
    features.append(
        np.clip(
            rng.exponential(scale=1.4375, size=n_samples) - 1.75, clip_low, clip_high
        )
    )

    # Feature 3 - Integer number of events in an interval, with average rate 1.75
    names.append("f3_poisson")
    types.append("continuous")
    features.append(
        np.clip(rng.poisson(lam=1.75, size=n_samples) - 2, clip_low_int, clip_high_int)
    )

    # Feature 4 - Interaction between feature 3 and feature 4
    names.append("f4_interaction")
    types.append("continuous")
    features.append(np.clip(features[2] * features[3], clip_low, clip_high))

    # Feature 5 - Positive correlation with feature 0 and negative with 1
    names.append("f5_multicollinearity")
    types.append("continuous")
    features.append(
        np.clip(
            0.75 * features[0] - 0.625 * features[1] + rng.normal(0.0, 1.0, n_samples),
            clip_low,
            clip_high,
        )
    )

    # Feature 6 - Correlation with feature 2 in center region
    names.append("f6_partial_correlation")
    types.append("continuous")
    clip_quarter = (clip_high - clip_low) * 0.25
    features.append(
        np.clip(
            np.where(
                (clip_low + clip_quarter < features[2])
                & (features[2] < clip_high - clip_quarter),
                1.5,
                0.0,
            )
            * features[2]
            + rng.normal(0.0, 1.0, n_samples),
            clip_low,
            clip_high,
        )
    )

    # Feature 7 - Useless feature
    names.append("f7_useless")
    types.append("continuous")
    features.append(rng.normal(1000.0, 1000.0, n_samples))

    # Feature 8 - Categorical feature with low cardinality
    names.append("f8_low_cardinality")
    types.append("nominal")
    n_categories = 9
    col = _make_categorical_float(rng, n_samples, n_categories, categorical_digits)
    if objects:
        col = _make_categorical_str(col, "l", categorical_digits)
    features.append(col)

    # Feature 9 - Categorical feature with high cardinality
    names.append("f9_high_cardinality")
    types.append("nominal")
    n_categories = 50
    col = _make_categorical_float(rng, n_samples, n_categories, categorical_digits)
    if objects:
        col = _make_categorical_str(col, "h", categorical_digits)
    features.append(col)

    # Convert list of features to a 2D numpy array and transpose
    X = np.array(features, dtype=object if objects else float).T

    if missing:
        # make 10% of feature data missing
        mask = rng.choice([False, True], X.shape, p=[0.9, 0.1])
        X[mask] = np.nan

    return (X, names, types)


def _make_categorical_float(rng, n_samples, n_categories, categorical_digits):
    n_modulo = 10**categorical_digits
    n_categories = min(n_categories, n_modulo - 1)
    mapping = rng.permutation(n_categories)
    mapping += n_modulo + 1
    mapping = mapping.astype(str)
    mapping = np.char.add(mapping, ".")
    vals = rng.choice(n_categories, n_samples)
    mapping = mapping[vals]

    # reserve 0 for an alternative missing representation
    vals += 1
    vals = vals.astype(str)
    vals = np.char.zfill(vals, categorical_digits)

    return np.char.add(mapping, vals).astype(float)


def _make_categorical_str(col, prefix, categorical_digits):
    cat_mod = 10**categorical_digits

    order = np.floor(col)

    col -= order
    col *= cat_mod
    np.round(col, out=col)
    col = col.astype(int).astype(str)
    col = np.char.zfill(col, categorical_digits)

    order = order.astype(int)
    order -= cat_mod
    order = order.astype(str)
    order = np.char.zfill(order, categorical_digits)
    order = np.char.translate(order, str.maketrans("0123456789", "abcdefghij"))
    order = np.char.add(prefix, np.char.add(order, "_"))

    return np.char.add(order, col)


def _normalize_float_categorical(col, clip_low, clip_high):
    missings = np.isnan(col)
    i_nonnan = sum(takewhile(operator.truth, missings))
    if i_nonnan != len(col):
        categorical_digits = len(str(int(np.floor(col[i_nonnan])))) - 1
        cat_mod = 10**categorical_digits

        col[missings] = 0.0
        col -= np.floor(col)
        col *= cat_mod
        np.round(col, out=col)
        col += -1.0
        col *= (clip_high - clip_low) / col.max()
        col += clip_low

        col[missings] = np.nan
    return col


def _normalize_string_categorical(col, clip_low, clip_high):
    missings = col != col
    if not missings.all():
        col[missings] = "_0"
        col = col.astype(str)
        col = np.char.rpartition(col, "_")[:, 2]
        col = col.astype(int).astype(float)
        col += -1.0
        col *= (clip_high - clip_low) / col.max()
        col += clip_low

        col[missings] = np.nan

    return col


def _normalize_categoricals(X, types, clip_low, clip_high):
    if X.dtype == object:
        features = []
        for i in range(X.shape[1]):
            col = X[:, i]
            col[col == np.array(None)] = np.nan
            if str in set(map(type, col)):
                col = _normalize_string_categorical(col, clip_low, clip_high)
            features.append(col)
        X = np.array(features, float).T
    else:
        X = X.copy()
        if types is not None:
            for i in range(X.shape[1]):
                if types[i] == "nominal":
                    X[:, i] = _normalize_float_categorical(X[:, i], clip_low, clip_high)
    return X


def _check_synthetic_dataset(
    X, y, names=None, types=None, clip_low=-2.0, clip_high=2.0
):
    n_display = 20

    X_imp = _normalize_categoricals(X, types, clip_low, clip_high)

    # impute missing values with 0
    missings = np.isnan(X_imp)
    X_imp[missings] = 0.0

    for i in range(X.shape[1]):
        print("--------------------")
        if names is not None:
            print(names[i])

        col = X_imp[:, i]
        n_zeros = len(col) - np.count_nonzero(col)
        col[col == 0.0] = np.tile(
            [2.2250738585072014e-308, -2.2250738585072014e-308], n_zeros // 2 + 1
        )[:n_zeros]

        negatives = col < 0.0
        negatives = str(np.average(col[negatives])) if negatives.any() else "NONE"
        print("neg_avg: " + negatives)

        positives = col > 0.0
        positives = str(np.average(col[positives])) if positives.any() else "NONE"
        print("pos_avg: " + positives)

        col = X[:, i]
        if X.dtype == object and str in set(map(type, col)):
            print("\n".join([str(x) for x in col[:n_display]]))
        elif types is not None and types[i] == "nominal":
            categorical_digits = len(str(int(np.floor(np.nanmax(col))))) - 1
            print("\n".join([f"{x:.{categorical_digits}f}" for x in col[:n_display]]))
        else:
            print(
                "\n".join([f"{x:.5f}".rstrip("0").rstrip(".") for x in col[:n_display]])
            )

    print("--------------------")
    print("y")
    if y.dtype == np.float64:
        negatives = y < 0.0
        negatives = str(np.average(y[negatives])) if negatives.any() else "NONE"
        print("neg_avg: " + negatives)

        positives = y > 0.0
        positives = str(np.average(y[positives])) if positives.any() else "NONE"
        print("pos_avg: " + positives)
        print("\n".join([f"{x:.5f}".rstrip("0").rstrip(".") for x in y[:n_display]]))
    else:
        print("\n".join([str(x) for x in y[:n_display]]))
