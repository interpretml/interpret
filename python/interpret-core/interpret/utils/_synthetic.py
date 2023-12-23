# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

import numpy as np


def _make_categorical_float(rng, n_samples, n_categories, cat_digits):
    n_modulo = 10**cat_digits
    n_categories = min(n_categories, n_modulo - 1)
    vals = rng.choice(n_categories, n_samples)
    mapping = rng.permutation(n_categories) + 1
    mapping = mapping[vals] + n_modulo
    mapping = mapping.astype(str)
    mapping = np.char.add(mapping, ".")
    # reserve 0 for an alternative missing representation
    vals = (vals + 1).astype(str)
    vals = np.char.zfill(vals, cat_digits)
    vals = np.char.add(mapping, vals)
    vals = vals.astype(float)
    return vals


def _make_categorical_str(col, prefix, cat_mod):
    order = np.floor(col).astype(int) % cat_mod
    col = np.round(col * cat_mod).astype(int) % cat_mod
    col = np.array(
        [prefix + str(o) + "_" + str(n) for o, n in zip(order, col)], dtype=object
    )
    return col


def _synthetic_features(rng, n_samples, missing, objects, cat_digits):
    # each feature is roughly set such that the average of the negative values is -2.5
    # and the average of the positive values is 2.5. This allows us to have a common scale
    # with integers where we have 9 categories from -4 to +4

    cat_mod = 10**cat_digits

    names = []
    types = []
    features = []

    # Feature 0 - Continuous drawn from uniform distribution
    names.append("f0_uniform")
    types.append("continuous")
    features.append(rng.uniform(-5.0, 5.0, n_samples))

    # Feature 1 - Continuous drawn from normal distribution
    names.append("f1_normal")
    types.append("continuous")
    features.append(rng.normal(0.125, 3.0, n_samples))

    # Feature 2 - Continuous time between events with avg time between events of 2.5
    names.append("f2_exponential")
    types.append("continuous")
    features.append(rng.exponential(scale=2.5, size=n_samples) - 4.0)  # shifted

    # Feature 3 - Integers with lumpy distribution
    names.append("f3_ints")
    types.append("continuous")
    features.append(rng.choice(9, n_samples) - 4)

    # Feature 4 - Integer number of events in an interval, with average rate 9
    names.append("f4_poisson")
    types.append("continuous")
    features.append(rng.poisson(lam=9, size=n_samples) - 9)

    # Feature 5 - Positive correlation with feature 0 and negative with 1
    names.append("f5_multicol")
    types.append("continuous")
    features.append(
        0.75 * features[0] - 0.625 * features[1] + rng.uniform(-3.5, 3.5, n_samples)
    )

    # Feature 6 - Correlation with feature 2 when feature 2 negative
    names.append("f6_partial")
    types.append("continuous")
    features.append(
        np.where(features[2] < 0.0, 0.75, 0.0) * features[2]
        + rng.uniform(-3.0, 6.0, n_samples)
    )

    # Feature 7 - Interaction between feature 3 and feature 4
    names.append("f7_interact")
    types.append("continuous")
    features.append(features[3] * features[4] * 0.375)

    # Feature 8 - Categorical feature with high cardinality
    names.append("f8_high")
    types.append("nominal")
    n_categories = int(n_samples / 4)
    col = _make_categorical_float(rng, n_samples, n_categories, cat_digits)
    if objects:
        col = _make_categorical_str(col, "h", cat_mod)
    features.append(col)

    # Feature 9 - Categorical feature with low cardinality
    names.append("f9_low")
    types.append("nominal")
    n_categories = 9
    col = _make_categorical_float(rng, n_samples, n_categories, cat_digits)
    if objects:
        col = _make_categorical_str(col, "l", cat_mod)
    features.append(col)

    # Convert list of features to a 2D numpy array of dtype=object and transpose
    X = np.array(features, dtype=object if objects else float).T

    if missing:
        # make 10% of feature data missing
        mask = rng.choice([False, True], X.shape, p=[0.9, 0.1])
        X[mask] = np.nan

    return (X, names, types)


ideal_cat_min = -5.0
ideal_cat_max = 5.0


def _normalize_string_categorical(col):
    missings = np.logical_or(col == np.array(None), col != col)
    if not missings.all():
        col[missings] = "m_0"
        col = np.array([float(x.split("_")[1]) for x in col], float)

        col -= col[~missings].min()
        col_max = col[~missings].max()
        col *= (ideal_cat_max - ideal_cat_min) / col_max
        col += ideal_cat_min

    col[missings] = np.nan
    return col


def _normalize_float_categorical(col):
    missings = np.isnan(col)
    if not missings.all():
        cat_digits = len(str(int(np.floor(np.nanmax(col))))) - 1
        cat_mod = 10**cat_digits

        col[missings] = 0.0
        col = (np.round(col * cat_mod).astype(int) % cat_mod).astype(float)

        col -= col[~missings].min()
        col_max = col[~missings].max()
        col *= (ideal_cat_max - ideal_cat_min) / col_max
        col += ideal_cat_min

        col[missings] = np.nan
    return col


def _normalize_categoricals(X, types):
    X = X.copy()
    if X.dtype == object:
        # if we have str categoricals, convert to float and normalize their range
        for i in range(X.shape[1]):
            col = X[:, i]
            if str in set(map(type, col)):
                X[:, i] = _normalize_string_categorical(col)
        missings = np.logical_or(X == np.array(None), X != X)
        X[missings] = np.nan  # change any None(s) to np.nan
        X = X.astype(float)
    else:
        # if we have float categoricals, normalize their range
        for i in range(X.shape[1]):
            if types is not None and types[i] == "nominal":
                X[:, i] = _normalize_float_categorical(X[:, i])
    return X


def make_synthetic(
    class_probs=[0.375, 0.25, 0.375],
    n_samples=1000,
    missing=False,
    objects=True,
    seed=1,
    noise_scale=1.0,
    cat_digits=4,
):
    rng = np.random.default_rng(seed)

    X, names, types = _synthetic_features(rng, n_samples, missing, objects, cat_digits)

    X_imp = _normalize_categoricals(X, types)

    # impute missing values with 0
    missings = np.isnan(X_imp)
    X_imp[missings] = 0.0

    # create some additive terms for our model to find
    y = rng.normal(-0.125, noise_scale, n_samples)
    y += np.exp(X_imp[:, 0] / 10.0)
    y += (X_imp[:, 1] / 10.0) ** 2
    y += (X_imp[:, 2] / 10.0) ** 3
    y += X_imp[:, 3] / 10.0 + np.sin(X_imp[:, 3])
    y += X_imp[:, 4] / 10.0 + np.sin(X_imp[:, 4])
    y += X_imp[:, 5] / 10.0 + np.sin(X_imp[:, 5])
    y += X_imp[:, 6] / 10.0 + np.sin(X_imp[:, 6])
    y += X_imp[:, 7] / 10.0 + np.sin(X_imp[:, 7])

    # 3-way interaction
    y += X_imp[:, 0] * X_imp[:, 1] * X_imp[:, 2] / 500.0

    # pairs
    y += X_imp[:, 0] * X_imp[:, 3] / 50.0
    y += X_imp[:, 3] * X_imp[:, 4] / 50.0

    # linear addition of high cardinality categorical
    y += X_imp[:, -2] / 10.0

    # linear addition of low cardinality categorical
    y += X_imp[:, -1] / 10.0

    if class_probs is not None:
        # if class_probs is non-None then it is classification

        # it would be better to treat y as logits and generate classes
        # but this is not meant for benchmarking, just testing and illustration
        # and it is easier to get multiclass this way with perscribed
        # numbers of classes

        class_probs = np.array(class_probs, float)
        class_probs /= class_probs.sum()  # normalize to prob of 1.0 just in case
        cumulative_probs = np.cumsum(class_probs)

        y_sorted = np.sort(y, kind="stable")
        split_indices = (cumulative_probs * n_samples).astype(int)[:-1]
        split_points = y_sorted[split_indices]
        y = np.digitize(y, split_points)

    return (X, y, names, types)


def _check_synthetic_dataset(X, y, names=None, types=None):
    X_imp = _normalize_categoricals(X, types)

    # impute missing values with 0
    missings = np.isnan(X_imp)
    X_imp[missings] = 0.0

    for i in range(X.shape[1]):
        print("--------------------")
        if names is not None:
            print(names[i])

        col = X_imp[:, i]

        negatives = col < 0.0
        negatives = str(np.average(col[negatives])) if negatives.any() else "NONE"
        print("neg_avg: " + negatives)

        positives = col > 0.0
        positives = str(np.average(col[positives])) if positives.any() else "NONE"
        print("pos_avg: " + positives)

        if X.dtype == object and str in set(map(type, X[:, i])):
            print("\n".join([str(x) for x in X[:20, i]]))
        else:
            print("\n".join([f"{x:.4f}" for x in X[:20, i]]))

    print("--------------------")
    print("y")
    if y.dtype == np.float64:
        negatives = y < 0.0
        negatives = str(np.average(y[negatives])) if negatives.any() else "NONE"
        print("neg_avg: " + negatives)

        positives = y > 0.0
        positives = str(np.average(y[positives])) if positives.any() else "NONE"
        print("pos_avg: " + positives)
        print("\n".join([f"{x:.4f}" for x in y[:20]]))
    else:
        print("\n".join([str(x) for x in y[:20]]))
