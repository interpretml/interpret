# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

import numpy as np
import math

def _make_categorical_float(n_samples, n_categories, cat_digits):
    n_modulo = 10 ** cat_digits
    n_categories = min(n_categories, n_modulo - 1)
    vals = np.random.choice(n_categories, n_samples)
    mapping = np.random.permutation(n_categories) + 1
    mapping = mapping[vals] + n_modulo
    mapping = mapping.astype(str)
    mapping = np.char.add(mapping, '.')
    vals = vals.astype(str)
    vals = np.char.zfill(vals, cat_digits)
    vals = np.char.add(mapping, vals)
    return vals
    
def _make_categorical_str(col, prefix):
    n_samples = len(col)
    col = col.astype(float)
    cat_digits = len(str(int(math.floor(np.nanmax(col))))) - 1
    cat_mod = 10 ** cat_digits

    missings = col != col

    col[missings] = cat_mod - 1
    order = col.astype(float).astype(int) % cat_mod
    col = (col.astype(float) * cat_mod).astype(int) % cat_mod
    col = np.array([prefix + str(o) + "_" + str(n) for o, n in zip(order, col)], dtype=object)
    col[missings] = np.nan
    return col

def make_synthetic(class_probs=[0.375, 0.25, 0.375], n_samples=400, missing=False, objects=False, seed=1, cat_digits=4):
    # each feature is roughly set such that the average of the negative values is -2.5
    # and the average of the positive values is 2.5. This allows us to have a common scale
    # with integers where we have 9 categories from -4 to +4

    cat_mod = 10 ** cat_digits

    np.random.seed(seed)
    names = []
    types = []
    features = []

    # Feature 0 - Continuous drawn from uniform distribution
    names.append("f0_uniform")
    types.append("continuous")
    features.append(np.random.uniform(-5.0, 5.0, n_samples))

    # Feature 1 - Continuous drawn from normal distribution
    names.append("f1_normal")
    types.append("continuous")
    features.append(np.random.normal(0.125, 3.0, n_samples))

    # Feature 2 - Continuous time between events with rate 1/10 per unit time
    names.append("f2_exp")
    types.append("continuous")
    features.append(np.random.exponential(scale=2.5, size=n_samples) - 4.0)

    # Feature 3 - Integers with lumpy distribution
    names.append("f3_int")
    types.append("continuous")
    features.append(np.random.choice(9, n_samples) - 4)

    # Feature 4 - Integer number of events in a fixed interval, with average rate 9
    names.append("f4_poisson")
    types.append("continuous")
    features.append(np.random.poisson(lam=9, size=n_samples) - 9)

    # Feature 5 - Positive correlation with feature 0 and negative with 1
    names.append("f5_multicol")
    types.append("continuous")
    features.append(0.75 * features[0] - 0.625 * features[1] + np.random.uniform(-3.5, 3.5, n_samples))

    # Feature 6 - Correlation with feature 2 when feature 2 negative
    names.append("f6_partial")
    types.append("continuous")
    features.append(np.where(features[2] < 0, 0.75, 0) * features[2] + np.random.uniform(-3.0, 6.0, n_samples))

    # Feature 7 - Interaction between feature 3 and feature 4
    names.append("f7_interact")
    types.append("continuous")
    features.append(features[3] * features[4] * 0.375)

    # Feature 8 - Categorical feature with high cardinality
    names.append("f8_high")
    types.append("nominal")
    n_categories = int(n_samples / 4)
    col = _make_categorical_float(n_samples, n_categories, cat_digits)
    if objects:
        col = _make_categorical_str(col, "h")
    features.append(col)

    # Feature 9 - Categorical feature with low cardinality
    names.append("f9_low")
    types.append("nominal")
    n_categories = 9
    col = _make_categorical_float(n_samples, n_categories, cat_digits)
    if objects:
        col = _make_categorical_str(col, "l")
    features.append(col)


    # Convert list of features to a 2D numpy array of dtype=object and transpose
    X = np.array(features, dtype=object if objects else float).T

    if missing:
        # make 10% of feature data missing
        mask = np.random.choice([False, True], size=X.shape, p=[0.9, 0.1])
        X[mask] = np.nan


    y = np.random.normal(-0.125, 1.0, n_samples)
    y += np.exp(features[0] / 10.0)
    y += (features[1] / 10.0) ** 2
    y += (features[2] / 10.0) ** 3
    y += features[3] / 10.0 + np.sin(features[3])
    y += features[4] / 10.0 + np.sin(features[4])
    y += features[5] / 10.0 + np.sin(features[5])
    y += features[6] / 10.0 + np.sin(features[6])
    y += features[7] / 10.0 + np.sin(features[7])

    # low cardinality is 0-9, so center around zero
    col = features[8]
    if objects:
        vals = np.array([float(x.split('_')[1]) for x in col])
    else:
        vals = (col.astype(float) * cat_mod).astype(int) % cat_mod
    vals = vals.astype(float)
    vals = vals / vals.max() - 0.5
    y += vals

    # high cardinality has range larger than 10, so normalize
    col = features[8]
    if objects:
        vals = np.array([float(x.split('_')[1]) for x in col])
    else:
        vals = (col.astype(float) * cat_mod).astype(int) % cat_mod
    vals = vals.astype(float)
    vals = vals / vals.max() - 0.5
    y += vals

    # 3-way interaction
    y += features[0] * features[1] * features[2] * 0.125

    # pairs
    y += features[0] * features[3] * 0.25
    y += features[3] * features[4] * 0.25


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

def _check_dataset(X, y, names=None, types=None):
    for i in range(X.shape[1]):
        print("--------------------")
        if names is not None:
            print(names[i])

        is_str = False
        col = X[:, i].copy()
        if types is not None and types[i] == "nominal":
            missings = col != col
            if col.dtype == object:
                is_str = True
                col[missings] = "x_-1"
                col = np.array([float(x.split('_')[1]) for x in col])
            else:
                cat_digits = len(str(int(math.floor(np.nanmax(col))))) - 1
                cat_mod = 10 ** cat_digits

                col[missings] = cat_mod - 1
                col = (col.astype(float) * cat_mod).astype(int) % cat_mod
                col[missings] = -1
        else:
            col[col != col] = 0
        print("neg_avg: " + str(np.average(col[col < 0])))
        print("pos_avg: " + str(np.average(col[col > 0])))
        if is_str:
            print('\n'.join([str(x) for x in X[:20, i]]))
        else:
            print('\n'.join([f'{x:.4f}' for x in X[:20, i]]))

    print("--------------------")
    print("y")
    if y.dtype == np.float64:
        print("neg_avg: " + str(np.average(y[y < 0])))
        print("pos_avg: " + str(np.average(y[y > 0])))
    print('\n'.join([f'{x:.4f}' for x in y[:20]]))
