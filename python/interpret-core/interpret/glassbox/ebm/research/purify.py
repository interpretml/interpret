""" Extra module for EBM. Code here can work with EBMs, but is not integrated into the main classes.
"""

# Purify Methods

# Reduce a numpy matrix such that each row and column has mean 0.
# Any offset is moved to the main effects. This recreates the
# Functional ANOVA decomposition.

# For more details, please refer to the paper:
# https://arxiv.org/abs/1911.04974

import numpy as np
import logging

log = logging.getLogger(__name__)


def purify_row(mat, marg, densities, i):
    # Purify such that row i has mean 0.
    try:
        _mean = np.average(mat[i, :], weights=densities[i, :])
        marg[i] += _mean
        mat[i, :] -= _mean
    except ZeroDivisionError:  # pragma: no cover
        pass
    return mat, marg


def purify_col(mat, marg, densities, j):
    # Purify such that column j has mean 0.
    try:
        _mean = np.average(mat[:, j], weights=densities[:, j])
        marg[j] += _mean
        mat[:, j] -= _mean
    except ZeroDivisionError:  # pragma: no cover
        pass
    return mat, marg


def purify_once(mat, densities=None, m1=None, m2=None, randomize=False):
    # Purify ecah row and column of the matrix mat.
    if densities is None:
        densities = np.ones_like(mat)
    # m1 along first axis
    if m1 is None:
        m1 = np.zeros((mat.shape[0], 1))

    # m2 along second axis
    if m2 is None:
        m2 = np.zeros((mat.shape[1], 1))

    if randomize:  # randomize each row/col selection rather than in-order
        nonzero_rows = set(list(range(mat.shape[0])))
        nonzero_cols = set(list(range(mat.shape[1])))
        for iteration in range(mat.shape[0] + mat.shape[1]):
            if np.random.binomial(1, 0.5) and len(nonzero_rows) > 0:
                i = np.random.choice(list(nonzero_rows))  # todo: maybe slow
                nonzero_rows.remove(i)
                mat, m1 = purify_row(mat, m1, densities, i)
            elif len(nonzero_cols) > 0:
                j = np.random.choice(list(nonzero_cols))
                nonzero_cols.remove(j)
                mat, m2 = purify_col(mat, m2, densities, j)

    # Fix each row mean
    for i in range(mat.shape[0]):
        mat, m1 = purify_row(mat, m1, densities, i)
    # Fix each col mean
    for j in range(mat.shape[1]):
        mat, m2 = purify_col(mat, m2, densities, j)

    return np.squeeze(m1), np.squeeze(m2), mat


def calc_row_means(mat, densities):
    means = []
    for i in range(mat.shape[0]):
        try:
            means.append(np.average(mat[i, :], weights=densities[i, :]))
        except ZeroDivisionError:  # pragma: no cover
            means.append(0)
    return means


def calc_col_means(mat, densities):
    means = []
    for j in range(mat.shape[1]):
        try:
            means.append(np.average(mat[:, j], weights=densities[:, j]))
        except ZeroDivisionError:  # pragma: no cover
            means.append(0)
    return means


def purify(mat, densities=None, verbose=False, tol=1e-6, randomize=False):
    # Move the means of the rows and columns into the marginal effects,
    # respecting sample density.
    # If randomize is True, then the order of row and columns is randomly
    # selected; otherwise they are processed in order.

    if densities is None:  # Use a uniform density
        densities = np.ones_like(mat)
    i = 1
    m1, m2, mat = purify_once(mat, densities)
    row_means = calc_row_means(mat, densities)
    col_means = calc_col_means(mat, densities)
    max_row = np.max(np.abs(row_means))
    max_col = np.max(np.abs(col_means))
    while max_row > tol or max_col > tol:
        i += 1
        if verbose:  # pragma: no cover
            log.info(i, max_row, max_col)
        m1, m2, mat = purify_once(mat, densities, m1, m2, randomize)
        row_means = calc_row_means(mat, densities)
        col_means = calc_col_means(mat, densities)
        max_row = np.max(np.abs(row_means))
        max_col = np.max(np.abs(col_means))
    # Center m1 and m2
    intercept = 0.0
    intercept += np.average(m1, weights=np.sum(densities, axis=1))
    m1 -= np.average(m1, weights=np.sum(densities, axis=1))
    intercept += np.average(m2, weights=np.sum(densities, axis=0))
    m2 -= np.average(m2, weights=np.sum(densities, axis=0))
    return intercept, m1, m2, mat, i
