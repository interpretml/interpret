# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license
# Author: Paul Koch <code@koch.ninja>

import numpy as np
from itertools import takewhile
import operator


def make_synthetic(
    classes=["class_0", "class_1"],
    n_samples=1000,
    missing=False,
    objects=True,
    seed=None,
    base_shift=0.0,
    noise_scale=0.25,
    categories=[9, 46],  # 46 is the max categories before the UI hides them
    categorical_floor=[0.2, 0.01],
    categorical_digits=3,
    clip_low=-2.0,
    clip_high=2.0,
):
    rng = np.random.default_rng(seed)

    X_orig, names, types = _make_synthetic_features(
        n_samples,
        missing,
        objects,
        rng,
        categories,
        categorical_floor,
        categorical_digits,
        clip_low,
        clip_high,
    )

    X_imp = _normalize_categoricals(X_orig, types, clip_low, clip_high)

    # Impute missing values with 0
    missings = np.isnan(X_imp, order="F")
    X_imp[missings] = 0.0

    # Create some additive term mains for our model to find
    y = rng.normal(base_shift, noise_scale, n_samples)
    y += np.cos(3.14159 * 4.0 / (clip_high - clip_low) * X_imp[:, 0]) * 0.9
    y += np.sin(3.14159 * 2.0 / (clip_high - clip_low) * X_imp[:, 1]) * 0.9
    y += X_imp[:, 2] ** 2 * 0.4
    y += X_imp[:, 3] * 0.4  # feature 3 contains poisson distributed integers
    y += np.where(((X_imp[:, 4] - clip_low) * 1.9999).astype(int) % 2, +0.7, -0.7)
    y += (np.modf(X_imp[:, 5] - clip_low)[0] - 0.5) * 1.5  # sawtooth wave
    y += np.exp(X_imp[:, 6]) * 0.15
    # Feature 7 is unused in the generation function
    y += X_imp[:, -2] * 0.4  # low cardinality categorical
    y += X_imp[:, -1] * 0.4  # high cardinality categorical

    # pair interactions
    xor_val = (X_imp[:, 3].astype(int) - int(np.floor(clip_low))) % 2
    y += X_imp[:, 0] * np.where(xor_val, +0.2, -0.2)
    y += X_imp[:, 1] * X_imp[:, 2] * 0.1

    # 3-way interaction
    y += X_imp[:, 0] * X_imp[:, 1] * X_imp[:, 2] * 0.02

    if classes is not None and classes != 0:
        if isinstance(classes, int):
            classes = np.arange(classes)
        else:
            classes = np.array(classes)

        n_classes = len(classes)
        if n_classes == 1:
            y = np.full(n_samples, classes[0])
        else:
            prob = np.exp(y)
            prob /= 1.0 + prob

            y = (rng.uniform(0.0, 1.0, n_samples) < prob).astype(int)

            if 2 < n_classes:
                # multiclass. To keep things simple, randomly select classes above
                # the 0th class with equal probability between the classes above 0.
                y = np.where(y, rng.integers(1, n_classes, n_samples), 0)

            y = classes[y]

    return (X_orig, y, names, types)


def _make_synthetic_features(
    n_samples,
    missing,
    objects,
    seed,
    categories,
    categorical_floor,
    categorical_digits,
    clip_low,
    clip_high,
):
    # EBMs are insensitive to the scale of the feature values since features are binned
    # using quantiles, so we have the flexibility to use any scale that suits our
    # needs. We have selected a scale for the feature values that makes generating
    # synthetic y values easier by keeping the same scale for all features in the
    # range between -2.0 and 2.0. This also helps keep transformations such as exp(x)
    # and x * x from becoming excessively large.
    # We've also adjusted each feature so the average negative value is about -1.0
    # and the average positive value is about +1.0. This establishes a uniform scale
    # with integers, accommodating five integers ranging from -2 to +2.

    if isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)

    clip_low_int = int(np.ceil(clip_low))
    clip_high_int = int(np.floor(clip_high))

    names = []
    types = []
    features = []

    # Feature 0 - Continuous drawn from a uniform distribution
    names.append("feature_0")
    types.append("continuous")
    features.append(rng.uniform(clip_low, clip_high, n_samples))

    # Feature 1 - Continuous drawn from a normal distribution
    names.append("feature_1")
    types.append("continuous")
    features.append(np.clip(rng.normal(0.0, 1.5, n_samples), clip_low, clip_high))

    # Feature 2 - Continuous time between events with avg time between events of 2.0
    names.append("feature_2")
    types.append("continuous")
    features.append(
        np.clip(rng.exponential(scale=2.0, size=n_samples) - 2.0, clip_low, clip_high)
    )

    # Feature 3 - Integer number of events in an interval, with average rate 2.0
    names.append("feature_3_integers")
    types.append("continuous")
    features.append(
        np.clip(rng.poisson(lam=2.0, size=n_samples) - 2, clip_low_int, clip_high_int)
    )

    # Feature 4 - Positive correlation with feature 0 and negative with 1
    names.append("feature_4")
    types.append("continuous")
    features.append(
        np.clip(
            0.7 * features[0] - 0.6 * features[1] + rng.normal(0.0, 0.7, n_samples),
            clip_low,
            clip_high,
        )
    )

    # Feature 5 - Correlation with feature 2 when feature 2 is in the center region
    names.append("feature_5")
    types.append("continuous")
    clip_quarter = (clip_high - clip_low) / 4.0
    features.append(
        np.clip(
            np.where(
                (clip_low + clip_quarter < features[2])
                & (features[2] < clip_high - clip_quarter),
                1.5,
                0.0,
            )
            * features[2]
            + rng.normal(0.0, 1.25, n_samples),
            clip_low,
            clip_high,
        )
    )

    # Feature 6 - Interaction between feature 2 and feature 3
    names.append("feature_6")
    types.append("continuous")
    features.append(
        np.clip(
            features[2] * features[3] + rng.normal(0.0, 0.5, n_samples),
            clip_low,
            clip_high,
        )
    )

    # Feature 7 - Unused feature
    names.append("feature_7_unused")
    types.append("continuous")
    features.append(rng.uniform(10.0, 100.0, n_samples))

    # Feature 8 - Categorical with low cardinality
    names.append("feature_8_low_cardinality")
    types.append("nominal")
    col = _make_categorical_float(
        rng, n_samples, categories[0], categorical_floor[0], categorical_digits
    )
    if objects:
        col = _make_categorical_str(col, "l", categorical_digits)
    features.append(col)

    # Feature 9 - Categorical with high cardinality
    names.append("feature_9_high_cardinality")
    types.append("nominal")
    col = _make_categorical_float(
        rng, n_samples, categories[-1], categorical_floor[-1], categorical_digits
    )
    if objects:
        col = _make_categorical_str(col, "h", categorical_digits)
    features.append(col)

    # Convert list of features to a 2D numpy array and transpose
    X = np.array(features, dtype=object if objects else float).T

    if missing is True:
        missing = 0.1  # by default make 10% of feature data missing
    elif missing is False or missing is None:
        missing = 0.0
    elif isinstance(missing, int):
        missing = float(missing)
    elif not isinstance(missing, float):
        raise ValueError(f"missing must be bool or float, but is {type(missing)}")

    if missing < 0.0:
        raise ValueError(f"missing cannot be negative")
    elif 1.0 < missing:
        raise ValueError(f"missing cannot be more than 1.0")
    elif missing != 0.0:
        mask = rng.choice(
            [False, True], tuple(reversed(X.shape)), p=[1.0 - missing, missing]
        ).T
        X[mask] = None if objects else np.nan

    return (X, names, types)


def _make_categorical_float(
    rng, n_samples, n_categories, categorical_floor, categorical_digits
):
    n_modulo = 10**categorical_digits
    n_categories = min(n_categories, n_modulo - 1)
    mapping = rng.permutation(n_categories)
    mapping += n_modulo + 1
    mapping = mapping.astype(str)
    mapping = np.char.add(mapping, ".")

    # reserve 0 for an alternative missing representation
    categories = np.arange(1, n_categories + 1).astype(str)
    categories = np.char.zfill(categories, categorical_digits)

    mapping = np.char.add(mapping, categories)
    mapping = mapping.astype(float)

    probs = rng.permutation(n_categories).astype(float)
    probs *= 1.0 / float(n_categories)
    probs += categorical_floor
    probs *= 1.0 / probs.sum()

    vals = rng.choice(n_categories, n_samples, p=probs)
    return mapping[vals]


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
    missings = col != col  # this is a check for nan that works with non-floats
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
