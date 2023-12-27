# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

import numpy as np
from itertools import takewhile
import operator


def _make_categorical_float(rng, n_samples, n_categories, cat_digits):
    n_modulo = 10**cat_digits
    n_categories = min(n_categories, n_modulo - 1)
    vals = rng.choice(n_categories, n_samples)
    mapping = n_modulo + 1 + rng.permutation(n_categories)
    mapping = mapping[vals]
    mapping = mapping.astype(str)
    mapping = np.char.add(mapping, ".")
    # reserve 0 for an alternative missing representation
    vals = (vals + 1).astype(str)
    vals = np.char.zfill(vals, cat_digits)
    vals = np.char.add(mapping, vals)
    vals = vals.astype(float)
    return vals


def _make_categorical_str(col, prefix, cat_digits):
    cat_mod = 10**cat_digits

    order = np.floor(col)

    col -= order
    col *= cat_mod
    col = np.char.zfill(np.round(col).astype(int).astype(str), cat_digits)

    order = order.astype(int)
    order -= cat_mod
    order = np.char.zfill(order.astype(str), cat_digits)
    order = np.char.translate(order, str.maketrans("0123456789", "abcdefghij"))
    order = np.char.add(prefix, np.char.add(order, "_"))

    return np.char.add(order, col)


def _synthetic_features(
    rng, n_samples, missing, objects, cat_digits, min_clip, max_clip
):
    # EBMs are blind to the scale of the feature values since features are binned using
    # quantiles, so we can use whatever scale we want for convenience. We choose the
    # scale of the feature values to make generating the synthetic y easier by roughly
    # keeping the same scale for all features.
    # Each feature is roughly set such that the average of the negative values is -1.0
    # and the average of the positive values is 1.0. This allows us to have a common
    # scale with integers where we have 5 categories from -2 to +2, and we clip at
    # -2.0 and +2.0 to make transformations like exp(x) and x**3 not too extreme.

    min_clip_int = int(np.ceil(min_clip))
    max_clip_int = int(np.floor(max_clip))

    names = []
    types = []
    features = []

    # Feature 0 - Continuous drawn from uniform distribution
    names.append("f0_uniform")
    types.append("continuous")
    features.append(rng.uniform(min_clip, max_clip, n_samples))

    # Feature 1 - Continuous drawn from normal distribution
    names.append("f1_normal")
    types.append("continuous")
    features.append(np.clip(rng.normal(0.0, 1.375, n_samples), min_clip, max_clip))

    # Feature 2 - Continuous time between events with avg time between events of 1.4375
    names.append("f2_exponential")
    types.append("continuous")
    features.append(
        np.clip(
            rng.exponential(scale=1.4375, size=n_samples) - 1.75, min_clip, max_clip
        )
    )

    # Feature 3 - Integers with lumpy distribution
    names.append("f3_ints")
    types.append("continuous")
    features.append(
        rng.choice(max_clip_int - min_clip_int + 1, n_samples)
        - (max_clip_int - min_clip_int) // 2
    )

    # Feature 4 - Integer number of events in an interval, with average rate 1.75
    names.append("f4_poisson")
    types.append("continuous")
    features.append(
        np.clip(rng.poisson(lam=1.75, size=n_samples) - 2, min_clip_int, max_clip_int)
    )

    # Feature 5 - Positive correlation with feature 0 and negative with 1
    names.append("f5_multicol")
    types.append("continuous")
    features.append(
        np.clip(
            0.75 * features[0] - 0.625 * features[1] + rng.normal(0.0, 1.0, n_samples),
            min_clip,
            max_clip,
        )
    )

    # Feature 6 - Correlation with feature 2 in center region
    names.append("f6_partial")
    types.append("continuous")
    clip_quarter = (max_clip - min_clip) * 0.25
    features.append(
        np.clip(
            np.where(
                (min_clip + clip_quarter < features[2])
                & (features[2] < max_clip - clip_quarter),
                1.5,
                0.0,
            )
            * features[2]
            + rng.normal(0.0, 1.0, n_samples),
            min_clip,
            max_clip,
        )
    )

    # Feature 7 - Interaction between feature 3 and feature 4
    names.append("f7_interact")
    types.append("continuous")
    features.append(np.clip(features[3] * features[4], min_clip_int, max_clip_int))

    # Feature 8 - Categorical feature with low cardinality
    names.append("f8_low")
    types.append("nominal")
    n_categories = 9
    col = _make_categorical_float(rng, n_samples, n_categories, cat_digits)
    if objects:
        col = _make_categorical_str(col, "l", cat_digits)
    features.append(col)

    # Feature 9 - Categorical feature with high cardinality
    names.append("f9_high")
    types.append("nominal")
    n_categories = 50
    col = _make_categorical_float(rng, n_samples, n_categories, cat_digits)
    if objects:
        col = _make_categorical_str(col, "h", cat_digits)
    features.append(col)

    # Convert list of features to a 2D numpy array and transpose
    X = np.array(features, dtype=object if objects else float).T

    if missing:
        # make 10% of feature data missing
        mask = rng.choice([False, True], X.shape, p=[0.9, 0.1])
        X[mask] = np.nan

    return (X, names, types)


def _normalize_string_categorical(col, min_clip, max_clip):
    missings = col != col
    if not missings.all():
        col[missings] = "_0"
        col = col.astype(str)
        col = np.char.rpartition(col, "_")[:, 2]
        col = col.astype(int)

        col = col - 1
        multiple = (max_clip - min_clip) / float(col.max())
        col = col.astype(float)
        col *= multiple
        col += min_clip

        col[missings] = np.nan

    return col


def _normalize_float_categorical(col, min_clip, max_clip):
    missings = np.isnan(col)
    i_nonnan = sum(takewhile(operator.truth, missings))
    if i_nonnan != len(col):
        cat_digits = len(str(int(np.floor(col[i_nonnan])))) - 1
        cat_mod = 10**cat_digits

        col[missings] = 0.0
        col -= np.floor(col)
        col *= cat_mod
        col = np.round(col).astype(int)

        col = col - 1
        multiple = (max_clip - min_clip) / float(col.max())
        col = col.astype(float)
        col *= multiple
        col += min_clip

        col[missings] = np.nan
    return col


def _normalize_categoricals(X, types, min_clip, max_clip):
    if X.dtype == object:
        features = []
        for i in range(X.shape[1]):
            col = X[:, i]
            col[col == np.array(None)] = np.nan
            if str in set(map(type, col)):
                col = _normalize_string_categorical(col, min_clip, max_clip)
            features.append(col)
        X = np.array(features, float).T
    else:
        X = X.copy()
        if types is not None:
            for i in range(X.shape[1]):
                if types[i] == "nominal":
                    X[:, i] = _normalize_float_categorical(X[:, i], min_clip, max_clip)
    return X


def make_synthetic(
    classes=["class_0", "class_1"],
    n_samples=1000,
    missing=False,
    objects=True,
    seed=1,
    intercept_shift=-4.0,  # avg y response is positive so shift to slighly negative
    noise_scale=1.0,
    cat_digits=4,
    min_clip=-2.0,
    max_clip=2.0,
):
    rng = np.random.default_rng(seed)

    X, names, types = _synthetic_features(
        rng, n_samples, missing, objects, cat_digits, min_clip, max_clip
    )

    X_imp = _normalize_categoricals(X, types, min_clip, max_clip)

    # impute missing values with 0
    missings = np.isnan(X_imp)
    X_imp[missings] = 0.0

    # create some additive terms for our model to find
    y = rng.normal(intercept_shift, noise_scale, n_samples)
    y += X_imp[:, 0] ** 2
    y += X_imp[:, 1] ** 3
    y += np.exp(X_imp[:, 2])
    y += X_imp[:, 3]  # integers (use linear response)
    y += -X_imp[:, 4]  # integers (use linear response)
    y += np.sin(3.14159 * 2.0 / (max_clip - min_clip) * X_imp[:, 5]) * 5.0
    y += np.cos(3.14159 * 2.0 / (max_clip - min_clip) * X_imp[:, 6]) * 5.0
    y += X_imp[:, 7]  # integer interaction feature

    # linear addition of low cardinality categorical
    y += X_imp[:, -2]

    # linear addition of high cardinality categorical
    y += X_imp[:, -1]

    # pairs
    y += X_imp[:, 0] * X_imp[:, 1]
    y += X_imp[:, 1] * X_imp[:, 2]

    # 3-way interaction
    y += X_imp[:, 0] * X_imp[:, 1] * X_imp[:, 2]

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

    return (X, y, names, types)


def _check_synthetic_dataset(X, y, names=None, types=None, min_clip=-2.0, max_clip=2.0):
    n_display = 20

    X_imp = _normalize_categoricals(X, types, min_clip, max_clip)

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
            cat_digits = len(str(int(np.floor(np.nanmax(col))))) - 1
            print("\n".join([f"{x:.{cat_digits}f}" for x in col[:n_display]]))
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
