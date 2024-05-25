# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license
# Author: Paul Koch <code@koch.ninja>

# For more details, please refer to the paper:
# https://arxiv.org/abs/1911.04974

import numpy as np


def _determine_impurities(scores, weights):
    shape = scores.shape
    n_equations = 0
    n_tensor = 1
    for dim_idx in range(len(shape)):
        multiply = 1
        n_tensor *= shape[dim_idx]
        for exclude_idx in range(len(shape)):
            if dim_idx != exclude_idx:
                multiply *= shape[exclude_idx]
        n_equations += multiply
    b = np.zeros((n_equations,), float)
    coefficients = np.zeros((n_equations, n_equations), float)
    lookups = np.empty(shape + (len(shape),), int)
    exclude_idx = len(shape) - 1
    equation_idx = 0
    tensor_index = [0] * len(shape)
    while True:
        for bin_idx in range(shape[exclude_idx]):
            tensor_index[exclude_idx] = bin_idx
            lookups[tuple(tensor_index + [exclude_idx])] = equation_idx
        tensor_index[exclude_idx] = 0
        equation_idx += 1
        if equation_idx == n_equations:
            break
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
    exclude_idx = len(shape) - 1
    equation_idx = 0
    tensor_index = [0] * len(shape)
    while True:
        for bin_idx in range(shape[exclude_idx]):
            tensor_index[exclude_idx] = bin_idx
            tupple_index = tuple(tensor_index)
            weight = weights[tupple_index]
            b[equation_idx] += weight * scores[tupple_index]
            for coeff_idx in lookups[tupple_index]:
                coefficients[equation_idx, coeff_idx] += weight
        tensor_index[exclude_idx] = 0
        equation_idx += 1
        if equation_idx == n_equations:
            break
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

    # Solve the system of equations. There are many possible methods for this..

    # FAILS: LU decomposition, error "Singular matrix"
    # solution = np.linalg.solve(coefficients, b)

    # FAILS: cholesky and LU decomposition, error "Matrix is not positive definite"
    # coefficients = np.linalg.cholesky(coefficients)
    # solution = np.linalg.solve(coefficients, b)

    # WORKS?: (Sometimes has "Singular matrix" error ?): QR Decomposition
    # 6.877648830413818 seconds
    # Q, R = np.linalg.qr(coefficients)
    # solution = np.linalg.solve(R, np.dot(Q.T, b))

    # WORKS: (but has floating point noise): SVD (Singular Value Decomposition)
    # 33.93425178527832 seconds
    # U, s, V = np.linalg.svd(coefficients)
    # c = np.dot(U.T, b)
    # w = np.linalg.solve(np.diag(s), c[:len(s)])
    # solution = np.dot(V.T, w)

    # WORKS: Biconjugate Gradient Method. Iterative and approximate but seems to return purest solution.
    # 0.08552408218383789 seconds
    # from scipy.sparse.linalg import bicg
    # solution, _ = bicg(coefficients, b)

    # WORKS (very noisy): Biconjugate Gradient Stabilized Method. Seems to have noise in the 1e-6 range
    # 0.042778968811035156 seconds
    # from scipy.sparse.linalg import bicgstab
    # solution, _ = bicgstab(coefficients, b)

    # WORKS: Conjugate Gradient iteration. Iterative and approximate but seems to return purest solution.
    # 0.04851675033569336 seconds
    from scipy.sparse.linalg import cg

    solution, _ = cg(coefficients, b)

    # WORKS: Generalized Minimal Residual. Iterative and approximate but seems to return purest solution.
    # 0.045647621154785156 seconds
    # from scipy.sparse.linalg import gmres
    # solution, _ = gmres(coefficients, b, x0=np.zeros_like(b))

    # WORKS (but has floating point noise): lstsq
    # 26.812599897384644 seconds
    # solution, _, _, _ = np.linalg.lstsq(coefficients, b)

    # WORKS (but has floating point noise): solve using pseudoinverse of a matrix
    # 33.44229531288147 seconds
    # solution = np.dot(np.linalg.pinv(coefficients), b)

    # Also possible: Successive Over-Relaxation, Jacobi or Gauss-Seidel Iterative Methods

    impurities = []
    base_idx = 0
    for exclude_idx in range(len(shape) - 1, -1, -1):
        count = n_tensor // shape[exclude_idx]
        impure_shape = list(shape)
        del impure_shape[exclude_idx]
        impure_shape = tuple(impure_shape)
        impurities.append(solution[base_idx : base_idx + count].reshape(impure_shape))
        base_idx += count
    return impurities


def _remove_impurities(scores, impurities):
    for i in range(scores.ndim):
        new_shape = list(scores.shape)
        new_shape[scores.ndim - 1 - i] = 1
        scores -= impurities[i].reshape(tuple(new_shape))


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


def _purify_single(scores, weights):
    scores = scores.copy()
    n_dim = scores.ndim
    impurities = []
    prev_level = [(0, [scores, weights])]
    for n_dimensions in range(n_dim, 1, -1):
        next_level = {}
        for dims, (level_scores, level_weights) in prev_level:
            level_impurities = _determine_impurities(level_scores, level_weights)
            _remove_impurities(level_scores, level_impurities)
            if n_dimensions != n_dim:
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
                if new_dims in next_level:
                    next_level[new_dims][0] += level_impurities[impure_idx]
                else:
                    next_level[new_dims] = [
                        level_impurities[impure_idx],
                        level_weights.sum(axis=n_dimensions - 1 - impure_idx),
                    ]
                impure_idx += 1
        prev_level = sorted(next_level.items())

    intercept = 0.0
    for dims, (level_scores, level_weights) in prev_level:
        mean = np.average(level_scores, weights=level_weights)
        intercept += mean
        level_scores -= mean
        key = tuple(
            n_dim - 1 - i for i in range(n_dim - 1, -1, -1) if ((1 << i) & dims) == 0
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
            tensor, impurities, intercept = _purify_single(
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
    tensor, impurities, intercept = _purify_single(scores, weights)
    return tensor, impurities, np.array([intercept], float)
