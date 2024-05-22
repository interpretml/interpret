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
        m = 1
        n_tensor *= shape[dim_idx]
        for exclude_idx in range(len(shape)):
            if dim_idx != exclude_idx:
                m *= shape[exclude_idx]
        n_equations += m
    b = np.zeros((n_equations,), float)
    coefficients = np.zeros((n_equations, n_equations), float)
    lookups = np.empty(shape + (len(shape),), int)
    exclude_idx = len(shape) - 1
    idx = 0
    tensor_index = np.zeros(len(shape), int)
    while True:
        for bin_idx in range(shape[exclude_idx]):
            tensor_index[exclude_idx] = bin_idx
            lookups[tuple(tensor_index) + (exclude_idx,)] = idx
        tensor_index[exclude_idx] = 0
        idx += 1
        if idx == n_equations:
            break
        dim_idx = len(shape) - 1
        while True:
            if dim_idx != exclude_idx:
                tensor_index[dim_idx] += 1
                if tensor_index[dim_idx] != shape[dim_idx]:
                    break
                tensor_index[dim_idx] = 0
            if dim_idx == 0:
                exclude_idx -= 1
                break
            dim_idx -= 1
    exclude_idx = len(shape) - 1
    idx = 0
    tensor_index = np.zeros(len(shape), int)
    while True:
        for bin_idx in range(shape[exclude_idx]):
            tensor_index[exclude_idx] = bin_idx
            for coeff_idx in lookups[tuple(tensor_index)]:
                coefficients[idx, coeff_idx] += weights[tuple(tensor_index)]
            b[idx] += weights[tuple(tensor_index)] * scores[tuple(tensor_index)]
        tensor_index[exclude_idx] = 0
        idx += 1
        if idx == n_equations:
            break
        dim_idx = len(shape) - 1
        while True:
            if dim_idx != exclude_idx:
                tensor_index[dim_idx] += 1
                if tensor_index[dim_idx] != shape[dim_idx]:
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
        impure_shape = tuple(
            n_bins for dim_idx, n_bins in enumerate(shape) if dim_idx != exclude_idx
        )
        impurities.append(solution[base_idx : base_idx + count].reshape(impure_shape))
        base_idx += count
    return impurities


def _remove_impurities(scores, impurities):
    shape = scores.shape
    for i in range(len(shape)):
        new_shape = list(shape)
        new_shape[len(shape) - 1 - i] = 1
        new_shape = tuple(new_shape)
        scores -= impurities[i].reshape(new_shape)
