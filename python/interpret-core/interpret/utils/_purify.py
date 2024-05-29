# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license
# Author: Paul Koch <code@koch.ninja>

# For more details, please refer to the paper:
# https://arxiv.org/abs/1911.04974

import numpy as np

from ._native import Native


def _UNUSED_purify_single_level(scores, weights):
    shape = scores.shape

    n_tensor = 1
    for n_bins in shape:
        n_tensor *= n_bins

    surface_dim_inc = np.empty((len(shape), len(shape)), int)
    surface_dim_reset = np.empty((len(shape), len(shape)), int)

    surface_indexes = np.empty(len(shape), int)
    n_equations = 0
    multiply = 1
    for exclude_idx in range(len(shape) - 1, -1, -1):
        count = n_tensor // shape[exclude_idx]
        surface_indexes[exclude_idx] = n_equations
        n_equations += count

        n_neg_bins = -shape[exclude_idx]
        for i in range(len(shape) - 1, exclude_idx, -1):
            val = multiply // shape[i]
            surface_dim_inc[exclude_idx, i] = val
            surface_dim_reset[exclude_idx, i] = val * n_neg_bins
        surface_dim_inc[exclude_idx, exclude_idx] = 0
        surface_dim_reset[exclude_idx, exclude_idx] = 0
        for i in range(exclude_idx - 1, -1, -1):
            surface_dim_inc[exclude_idx, i] = multiply
            surface_dim_reset[exclude_idx, i] = multiply * n_neg_bins
        multiply *= shape[exclude_idx]

    b = np.empty(n_equations, float)
    coefficients = np.zeros(n_equations * n_equations, float)

    flat_scores = scores.flatten()
    flat_weights = weights.flatten()

    cur_inc = np.full(len(shape) - 1, 1, int)
    exclude_idx = len(shape) - 1
    equation_idx = 0
    tensor_index = [0] * len(shape)
    tensor_idx = 0
    tensor_inc = 1
    while True:
        n_start = equation_idx * n_equations
        save_index = surface_indexes[exclude_idx] + n_start
        reduced_indexes = np.delete(surface_indexes, exclude_idx)
        reduced_indexes += n_start
        local_idx = tensor_idx
        total = 0.0
        bval = 0.0
        for _ in range(shape[exclude_idx]):
            weight = flat_weights[local_idx]
            total += weight
            bval += weight * flat_scores[local_idx]
            coefficients[reduced_indexes] = weight
            reduced_indexes += cur_inc
            local_idx += tensor_inc
        b[equation_idx] = bval
        coefficients[save_index] = total

        equation_idx += 1
        dim_idx = len(shape) - 1
        multiply = 1
        while True:
            if dim_idx != exclude_idx:
                surface_indexes += surface_dim_inc[dim_idx]

                tensor_idx += multiply

                bin_idx = tensor_index[dim_idx] + 1
                tensor_index[dim_idx] = bin_idx

                if bin_idx != shape[dim_idx]:
                    break
                tensor_index[dim_idx] = 0
                tensor_idx -= multiply * shape[dim_idx]

                surface_indexes += surface_dim_reset[dim_idx]
            multiply *= shape[dim_idx]

            dim_idx -= 1
            if dim_idx < 0:
                tensor_inc *= shape[exclude_idx]
                exclude_idx -= 1
                if exclude_idx < 0:
                    coefficients = coefficients.reshape((n_equations, n_equations))

                    # Solve the system of equations. There are many possible methods for this..

                    # FAILS: LU decomposition, error "Singular matrix"
                    # solution = np.linalg.solve(coefficients, b)

                    # FAILS: cholesky and LU decomposition, error "Matrix is not positive definite"
                    # coefficients = np.linalg.cholesky(coefficients)
                    # solution = np.linalg.solve(coefficients, b)

                    # FAILS (sometimes): QR Decomposition, error "Singular matrix"
                    # 6.877648830413818 seconds
                    # Q, R = np.linalg.qr(coefficients)
                    # solution = np.linalg.solve(R, np.dot(Q.T, b))

                    # WORKS: SVD (Singular Value Decomposition). Very pure results.
                    # 33.93425178527832 seconds
                    # U, s, V = np.linalg.svd(coefficients)
                    # c = np.dot(U.T, b)
                    # w = np.linalg.solve(np.diag(s), c[:len(s)])
                    # solution = np.dot(V.T, w)

                    # WORKS: Biconjugate Gradient Method. Iterative and approximate and result has impurities.
                    # 0.08552408218383789 seconds
                    # from scipy.sparse.linalg import bicg
                    # solution, _ = bicg(coefficients, b)

                    # WORKS: Biconjugate Gradient Stabilized Method. Iterative and approximate and result has impurities.
                    # 0.042778968811035156 seconds
                    # from scipy.sparse.linalg import bicgstab
                    # solution, _ = bicgstab(coefficients, b)

                    # WORKS: Conjugate Gradient iteration. Iterative and approximate and result has impurities.
                    # 0.04851675033569336 seconds
                    from scipy.sparse.linalg import cg

                    solution, _ = cg(coefficients, b, atol=1e-10)

                    # WORKS: Generalized Minimal Residual. Iterative and approximate and result has impurities.
                    # 0.045647621154785156 seconds
                    # from scipy.sparse.linalg import gmres
                    # solution, _ = gmres(coefficients, b, x0=np.zeros_like(b))

                    # WORKS: lstsq. Very pure results.
                    # 26.812599897384644 seconds
                    # solution, _, _, _ = np.linalg.lstsq(coefficients, b)

                    # WORKS: Solve using pseudoinverse of a matrix. Somewhat inaccurate for large tensors
                    # 33.44229531288147 seconds
                    # solution = np.dot(np.linalg.pinv(coefficients), b)

                    # Also possible: Successive Over-Relaxation, Jacobi or Gauss-Seidel Iterative Methods

                    impurities = []
                    base_idx = 0
                    for exclude_idx in range(len(shape) - 1, -1, -1):
                        count = n_tensor // shape[exclude_idx]
                        impurity = solution[base_idx : base_idx + count]
                        base_idx += count

                        impure_shape = list(shape)
                        impure_shape[exclude_idx] = 1
                        scores -= impurity.reshape(tuple(impure_shape))

                        del impure_shape[exclude_idx]
                        impurity = impurity.reshape(tuple(impure_shape))
                        impurities.append(impurity)

                    # using systems of equations is not exact because we either
                    # use an approximate iterative method or an exact method with
                    # many steps that introduces floating point noise. We can
                    # extract the intercept more accurately though, so do that.
                    residual_intercept = np.average(scores, weights=weights)
                    scores -= residual_intercept

                    return impurities, residual_intercept

                cur_inc = np.delete(surface_dim_inc[exclude_idx], exclude_idx)
                break


def _measure_impurity(scores, weights):
    if scores.ndim != weights.ndim:
        if scores.ndim != weights.ndim + 1:
            raise Exception(
                "scores and weights do not match in terms of dimensionality."
            )
        # multiclass means the scores have the class scores in the last dimension
        return sum(
            _measure_impurity(scores[..., i], weights) for i in range(scores.shape[-1])
        )

    shape = scores.shape
    exclude_idx = len(shape) - 1
    tensor_index = [0] * len(shape)
    total_system = 0.0
    while True:
        total_equation = 0.0
        for bin_idx in range(shape[exclude_idx]):
            tensor_index[exclude_idx] = bin_idx
            tupple_index = tuple(tensor_index)
            total_equation += weights[tupple_index] * scores[tupple_index]
        tensor_index[exclude_idx] = 0
        total_system += abs(total_equation)

        dim_idx = len(shape) - 1
        while True:
            if dim_idx != exclude_idx:
                bin_idx = tensor_index[dim_idx] + 1
                tensor_index[dim_idx] = bin_idx
                if bin_idx != shape[dim_idx]:
                    break
                tensor_index[dim_idx] = 0
            if dim_idx == 0:
                exclude_idx -= 1
                break
            dim_idx -= 1
        if exclude_idx < 0:
            break
    return total_system


def _purify_downstream(scores, weights):
    native = Native.get_native_singleton()

    scores = scores.copy()
    n_dim = scores.ndim
    impurities = []
    n_possible = (1 << n_dim) - 1
    prev_level = [None] * n_possible
    prev_level[0] = [scores, weights]
    next_level = [None] * n_possible
    intercept = 0.0
    for n_dimensions in range(n_dim, 1, -1):
        for dims in range(n_possible):
            items = prev_level[dims]
            if items is None:
                continue
            level_scores, level_weights = items
            prev_level[dims] = None

            level_impurities, level_intercept = native.purify(
                level_scores, level_weights
            )
            intercept += level_intercept
            if dims != 0:
                # do not insert the original score tensor into the impurities
                key = tuple(
                    n_dim - 1 - i
                    for i in range(n_dim - 1, -1, -1)
                    if ((1 << i) & dims) == 0
                )
                impurities.append((key, level_scores))

            impure_idx = 0
            for dim_idx in range(n_dim):
                if (1 << dim_idx) & dims != 0:
                    continue
                new_dims = dims | (1 << dim_idx)
                items = next_level[new_dims]
                if items is not None:
                    items[0] += level_impurities[impure_idx]
                else:
                    next_level[new_dims] = [
                        level_impurities[impure_idx],
                        level_weights.sum(axis=n_dimensions - 1 - impure_idx),
                    ]
                impure_idx += 1
        temp = next_level
        next_level = prev_level
        prev_level = temp

    for dims in range(n_possible):
        items = prev_level[dims]
        if items is None:
            continue
        level_scores, level_weights = items

        mean = np.average(level_scores, weights=level_weights)
        intercept += mean
        level_scores -= mean
        if dims != 0:
            key = tuple(
                n_dim - 1 - i
                for i in range(n_dim - 1, -1, -1)
                if ((1 << i) & dims) == 0
            )
            impurities.append((key, level_scores))

    return scores, impurities, intercept


# TODO: Apply purification to EBMs either (based on a boolean option that we can expose publicly):
#    1) After all boosting is complete.  We can either throw away the lower dimensional contributions,
#       or move the score contributions to the lower dimensional terms based on benchmarking results.
#    2) During boosting, where we would throw away the impure components so that
#       the algorithm would not overfit the lower dimensional components.
#       - This would be especially important when we boost mains and interactions together at
#         the same time because we don't want the model to force feed some mains that just happen
#         to be included in an interaction.
def purify(scores, weights):
    if scores.ndim != weights.ndim:
        if scores.ndim != weights.ndim + 1:
            raise Exception(
                "scores and weights do not match in terms of dimensionality."
            )
        # multiclass means the scores have the class scores in the last dimension

        new_dims = None
        new_tensor = []
        new_intercept = []
        for class_idx in range(scores.shape[-1]):
            tensor, impurities, intercept = _purify_downstream(
                scores[..., class_idx], weights
            )
            new_tensor.append(tensor)
            new_intercept.append(intercept)
            if new_dims is None:
                new_dims = [dims for dims, _ in impurities]
                new_impurities = [[] for _ in impurities]
            for i in range(len(impurities)):
                new_impurities[i].append(impurities[i][1])

        impurities = [
            (key, np.stack(vals, axis=-1, dtype=float))
            for key, vals in zip(new_dims, new_impurities)
        ]
        new_tensor = np.stack(new_tensor, axis=-1, dtype=float)
        new_intercept = np.array(new_intercept, float)
        return new_tensor, impurities, new_intercept
    tensor, impurities, intercept = _purify_downstream(scores, weights)
    return tensor, impurities, np.array([intercept], float)