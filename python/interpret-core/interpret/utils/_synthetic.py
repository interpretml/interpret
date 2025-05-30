# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license
# Author: Paul Koch <code@koch.ninja>

import numpy as np
from ._misc import safe_isinstance

try:
    import pandas as pd

    _pandas_installed = True
except ImportError:
    _pandas_installed = False


def make_synthetic(
    classes=("class_0", "class_1"),
    n_samples=1000,
    missing=False,
    seed=None,
    output_type="object",
    noise_scale=0.25,
    base_shift=0.0,
    higher_class_probs=None,
    impute_missing=0.0,
    disable=None,
    categories=(9, 46),  # 46 is the max categories before the UI hides them
    categorical_floor=(0.2, 0.01),
    categorical_digits=3,
    clip_low=-2.0,
    clip_high=2.0,
):
    if disable is None:
        disable = []
    rng = np.random.default_rng(seed)

    X_orig, names, types = _make_synthetic_features(
        n_samples,
        missing,
        rng,
        output_type,
        categories,
        categorical_floor,
        categorical_digits,
        clip_low,
        clip_high,
    )

    X_imp = _normalize_categoricals(X_orig, types, clip_low, clip_high)

    # Impute missing values with 0
    missings = np.isnan(X_imp, order="F")
    X_imp[missings] = impute_missing

    # Create some additive term mains for our model to find
    y = rng.normal(base_shift, noise_scale, n_samples)
    if all(d not in disable for d in ["cos", "mains"]):
        y += np.cos(3.14159 * 4.0 / (clip_high - clip_low) * X_imp[:, 0]) * 0.9
    if all(d not in disable for d in ["sin", "mains"]):
        y += np.sin(3.14159 * 2.0 / (clip_high - clip_low) * X_imp[:, 1]) * 0.9
    if all(d not in disable for d in ["parabola", "mains"]):
        y += X_imp[:, 2] ** 2 * 0.4
    if all(d not in disable for d in ["linear_int", "mains"]):
        y += X_imp[:, 3] * 0.4  # feature 3 contains poisson distributed integers
    if all(d not in disable for d in ["square_wave", "mains"]):
        y += np.where(((X_imp[:, 4] - clip_low) * 1.9999).astype(int) % 2, +0.6, -0.6)
    if all(d not in disable for d in ["sawtooth_wave", "mains"]):
        y += (np.modf(X_imp[:, 5] - clip_low)[0] - 0.5) * 1.4
    if all(d not in disable for d in ["exp", "mains"]):
        y += np.exp(X_imp[:, 6]) * 0.15
    # Feature 7 is unused in the generation function
    if all(d not in disable for d in ["low_cardinality", "nominals", "mains"]):
        y += X_imp[:, -2] * 0.4
    if all(d not in disable for d in ["high_cardinality", "nominals", "mains"]):
        y += X_imp[:, -1] * 0.4

    # pair interactions
    if all(d not in disable for d in ["xor", "pairs"]):
        xor_val = (X_imp[:, 3].astype(np.int64) - int(np.floor(clip_low))) % 2
        y += X_imp[:, 0] * np.where(xor_val, +0.3, -0.3)
    if all(d not in disable for d in ["multiply_continuous", "pairs"]):
        y += X_imp[:, 1] * X_imp[:, 2] * 0.1
    if all(d not in disable for d in ["multiply_nominal", "pairs"]):
        y += X_imp[:, 3] * X_imp[:, -2] * 0.2

    # 3-way interaction between float, int, and nominal
    if all(d not in disable for d in ["triples"]):
        y += X_imp[:, 2] * X_imp[:, 3] * X_imp[:, -2] * 0.02

    if classes is not None and classes != 0:
        classes = np.arange(classes) if isinstance(classes, int) else np.array(classes)

        n_classes = len(classes)
        if n_classes == 1:
            y = np.full(n_samples, classes[0])
        else:
            prob = np.exp(y)
            prob /= 1.0 + prob

            y = (rng.uniform(0.0, 1.0, n_samples) < prob).astype(np.int64)

            if n_classes > 2:
                # multiclass. To keep things simple, randomly select classes above
                # the 0th class with equal probability between the classes above 0.
                y = np.where(
                    y, rng.choice(n_classes - 1, n_samples, p=higher_class_probs) + 1, 0
                )

            y = classes[y]

    return (X_orig, y, names, types)


def _make_synthetic_features(
    n_samples,
    missing,
    seed,
    output_type,
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

    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

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
    if output_type in ["object", "pandas", "str"]:
        col = _make_categorical_str(col, "l", categorical_digits)
    features.append(col)

    # Feature 9 - Categorical with high cardinality
    names.append("feature_9_high_cardinality")
    types.append("nominal")
    col = _make_categorical_float(
        rng, n_samples, categories[-1], categorical_floor[-1], categorical_digits
    )
    if output_type in ["object", "pandas", "str"]:
        col = _make_categorical_str(col, "h", categorical_digits)
    features.append(col)

    if missing is True:
        missing = 0.1  # by default make 10% of feature data missing
    elif missing is False or missing is None:
        missing = 0.0
    elif isinstance(missing, int):
        missing = float(missing)
    elif not isinstance(missing, float):
        msg = f"missing must be bool or float, but is {type(missing)}"
        raise ValueError(msg)

    if missing < 0.0:
        msg = "missing cannot be negative"
        raise ValueError(msg)
    if missing > 1.0:
        msg = "missing cannot be more than 1.0"
        raise ValueError(msg)
    if missing == 0.0:
        mask = None
    else:
        mask = rng.choice(
            [False, True], (len(features), n_samples), p=[1.0 - missing, missing]
        )

    # Convert list of features to a 2D numpy array and transpose
    if output_type == "object":
        X = np.array(features, np.object_)
        if mask is not None:
            X[mask] = None
        X = X.T
    elif output_type == "float":
        X = np.array(features, np.float64)
        if mask is not None:
            X[mask] = np.nan
        X = X.T
    elif output_type == "str":
        if mask is not None:
            X = np.ma.array(features, np.str_, mask=mask)
        else:
            X = np.array(features, np.str_)
        X = X.T
    elif output_type == "pandas":
        if not _pandas_installed:
            msg = "pandas was requested, but is not installed."
            raise ValueError(msg)

        for i in range(len(features)):
            col = features[i]
            if types[i] == "nominal":
                dtype = "category"
            elif np.issubdtype(col.dtype, np.integer) and mask is not None:
                dtype = "Int64"
            else:
                dtype = col.dtype
            col = pd.Series(col, dtype=dtype, name=names[i])
            if mask is not None:
                col[mask[i]] = np.nan
            features[i] = col

        X = pd.concat(features, axis=1)
    elif output_type == "csc_matrix":
        try:
            from scipy.sparse import csc_matrix
        except ImportError:
            raise ImportError(
                'Please install the scipy package using `pip install scipy` in order to call make_synthetic with output_type set to "scipy"!'
            )
        X = np.array(features, np.float64)
        if mask is not None:
            X[mask] = np.nan
        X = csc_matrix(X.T)
    elif output_type == "csc_array":
        try:
            from scipy.sparse import csc_array
        except ImportError:
            raise ImportError(
                'Please install the scipy package using `pip install scipy` in order to call make_synthetic with output_type set to "scipy"!'
            )
        X = np.array(features, np.float64)
        if mask is not None:
            X[mask] = np.nan
        X = csc_array(X.T)
    else:
        msg = f"unknown output_type={output_type}"
        raise ValueError(msg)

    return (X, names, types)


def _make_categorical_float(
    rng, n_samples, n_categories, categorical_floor, categorical_digits
):
    n_modulo = 10**categorical_digits
    n_categories = min(n_categories, n_modulo - 1)
    mapping = rng.permutation(n_categories)
    mapping += n_modulo + 1
    mapping = mapping.astype(np.str_)
    mapping = np.char.add(mapping, ".")

    # reserve 0 for an alternative missing representation
    categories = np.arange(1, n_categories + 1).astype(np.str_)
    categories = np.char.zfill(categories, categorical_digits)

    mapping = np.char.add(mapping, categories)
    mapping = mapping.astype(np.float64)

    probs = rng.permutation(n_categories).astype(np.float64)
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
    col = col.astype(np.int64).astype(np.str_)
    col = np.char.zfill(col, categorical_digits)

    order = order.astype(np.int64)
    order -= cat_mod
    order = order.astype(np.str_)
    order = np.char.zfill(order, categorical_digits)
    order = np.char.translate(order, str.maketrans("0123456789", "abcdefghij"))
    order = np.char.add(prefix, np.char.add(order, "_"))

    return np.char.add(order, col)


def _normalize_float_categorical(col, clip_low, clip_high):
    categorical_digits = len(str(int(np.floor(col[0])))) - 1
    cat_mod = 10**categorical_digits

    col -= np.floor(col)
    col *= cat_mod
    np.round(col, out=col)
    col += -1.0
    col *= (clip_high - clip_low) / col.max()
    col += clip_low

    return col


def _normalize_string_categorical(col, clip_low, clip_high):
    col = np.asarray(col, np.str_)
    col = np.char.rpartition(col, "_")[:, 2]
    col = col.astype(np.int64).astype(np.float64)
    col += -1.0
    col *= (clip_high - clip_low) / col.max()
    col += clip_low

    return col


def _normalize_categoricals(X, types, clip_low, clip_high):
    features = []
    for i in range(X.shape[1]):
        if _pandas_installed and isinstance(X, pd.DataFrame):
            col = X.iloc[:, i]
        elif safe_isinstance(X, "scipy.sparse.spmatrix") or safe_isinstance(
            X, "scipy.sparse.sparray"
        ):
            col = X[:, [i]].toarray().ravel()
        else:
            col = X[:, i]

        if _pandas_installed:
            nonmissings = pd.notna(col)
        else:
            # this is a check for nan that works with non-floats
            nonmissings = col == col
            if isinstance(col.dtype, np.dtype) and np.issubdtype(col.dtype, np.object_):
                nonmissings &= col != np.array(None)

        if isinstance(col, np.ma.masked_array) and col.mask is not np.ma.nomask:
            nonmissings &= ~col.mask

        col = col[nonmissings]
        floats = np.full(len(nonmissings), np.nan, np.float64)
        if len(col) != 0:
            if types is not None and types[i] == "nominal":
                if isinstance(col.dtype, np.dtype) and np.issubdtype(
                    col.dtype, np.floating
                ):
                    col = _normalize_float_categorical(col, clip_low, clip_high)
                else:
                    col = _normalize_string_categorical(col, clip_low, clip_high)
            else:
                try:
                    col = col.astype(np.float64, copy=False)
                except ValueError:
                    col = _normalize_string_categorical(col, clip_low, clip_high)

            np.place(floats, nonmissings, col)

        features.append(floats)
    return np.array(features, np.float64).T


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
        if X.dtype == np.object_ and str in set(map(type, col)):
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
    if np.issubdtype(y.dtype, np.floating):
        negatives = y < 0.0
        negatives = str(np.average(y[negatives])) if negatives.any() else "NONE"
        print("neg_avg: " + negatives)

        positives = y > 0.0
        positives = str(np.average(y[positives])) if positives.any() else "NONE"
        print("pos_avg: " + positives)
        print("\n".join([f"{x:.5f}".rstrip("0").rstrip(".") for x in y[:n_display]]))
    else:
        print("\n".join([str(x) for x in y[:n_display]]))
