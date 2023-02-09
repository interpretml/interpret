# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

from ._native import Native

import logging
_log = logging.getLogger(__name__)

def validate_eps_delta(eps, delta):
    if eps is None or eps <= 0 or delta is None or delta <= 0:
        raise ValueError(f"Epsilon: '{eps}' and delta: '{delta}' must be set to positive numbers")

def calc_classic_noise_multi(total_queries, target_epsilon, delta, sensitivity):
    variance = (8*total_queries*sensitivity**2 * np.log(np.exp(1) + target_epsilon / delta)) / target_epsilon ** 2
    return np.sqrt(variance)

# General calculations, largely borrowed from tensorflow/privacy and presented in https://arxiv.org/abs/1911.11607
def delta_eps_mu(eps, mu):
    ''' Code adapted from: https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/analysis/gdp_accountant.py#L44
    '''
    return norm.cdf(-eps/mu + mu/2) - np.exp(eps) * norm.cdf(-eps/mu - mu/2)

def calc_gdp_noise_multi(total_queries, target_epsilon, delta):
    ''' GDP analysis following Algorithm 2 in: https://arxiv.org/abs/2106.09680. 
    '''
    def f(mu, eps, delta):
        return delta_eps_mu(eps, mu) - delta

    final_mu = brentq(lambda x: f(x, target_epsilon, delta), 1e-5, 1000)
    sigma = np.sqrt(total_queries) / final_mu
    return sigma

def private_numeric_binning(X_col, sample_weight, noise_scale, max_bins, min_feature_val, max_feature_val, rng=None):
    native = Native.get_native_singleton()
    uniform_weights, uniform_edges = np.histogram(X_col, bins=max_bins*2, range=(min_feature_val, max_feature_val), weights=sample_weight)
    noisy_weights = uniform_weights + native.generate_gaussian_random(rng=rng, stddev=noise_scale, count=uniform_weights.shape[0])
        
    # Postprocess to ensure realistic bin values (min=0)
    noisy_weights = np.clip(noisy_weights, 0, None)

    # TODO PK: check with Harsha, but we can probably alternate the taking of nibbles from both ends
    # so that the larger leftover bin tends to be in the center rather than on the right.

    # Greedily collapse bins until they meet or exceed target_weight threshold
    sample_weight_total = len(X_col) if sample_weight is None else np.sum(sample_weight)
    target_weight = sample_weight_total / max_bins
    bin_weights, bin_cuts = [0], [uniform_edges[0]]
    curr_weight = 0
    for index, right_edge in enumerate(uniform_edges[1:]):
        curr_weight += noisy_weights[index]
        if curr_weight >= target_weight:
            bin_cuts.append(right_edge)
            bin_weights.append(curr_weight)
            curr_weight = 0

    if len(bin_weights) == 1:
        # since we're adding unbounded random noise, it's possible that the total weight is less than the
        # threshold required for a single bin.  It could in theory even be negative.
        # clip to the target_weight.  If we had more than the target weight we'd have a bin

        bin_weights.append(target_weight)
        bin_cuts = np.empty(0, dtype=np.float64)
    else:
        # Ignore min/max value as part of cut definition
        bin_cuts = np.array(bin_cuts, dtype=np.float64)[1:-1]

        # All leftover datapoints get collapsed into final bin
        bin_weights[-1] += curr_weight

    return bin_cuts, bin_weights

def private_categorical_binning(X_col, sample_weight, noise_scale, max_bins, rng=None):
    native = Native.get_native_singleton()
    # Initialize estimate
    X_col = X_col.astype('U')
    uniq_vals, uniq_idxs = np.unique(X_col, return_inverse=True)
    weights = np.bincount(uniq_idxs, weights=sample_weight, minlength=len(uniq_vals))

    weights = weights + native.generate_gaussian_random(rng=rng, stddev=noise_scale, count=weights.shape[0])

    # Postprocess to ensure realistic bin values (min=0)
    weights = np.clip(weights, 0, None)

    # Collapse bins until target_weight is achieved.
    sample_weight_total = len(X_col) if sample_weight is None else np.sum(sample_weight)
    target_weight = sample_weight_total / max_bins
    small_bins = np.where(weights < target_weight)[0]
    if len(small_bins) > 0:
        other_weight = np.sum(weights[small_bins])
        mask = np.ones(weights.shape, dtype=bool)
        mask[small_bins] = False

        # Collapse all small bins into "DPOther"
        uniq_vals = np.append(uniq_vals[mask], "DPOther")
        weights = np.append(weights[mask], other_weight)

        if other_weight < target_weight:
            if len(weights) == 1:
                # since we're adding unbounded random noise, it's possible that the total weight is less than the
                # threshold required for a single bin.  It could in theory even be negative.
                # clip to the target_weight
                weights[0] = target_weight
            else:
                # If "DPOther" bin is too small, absorb 1 more bin (guaranteed above threshold)
                collapse_bin = np.argmin(weights[:-1])
                mask = np.ones(weights.shape, dtype=bool)
                mask[collapse_bin] = False

                # Pack data into the final "DPOther" bin
                weights[-1] += weights[collapse_bin]

                # Delete absorbed bin
                uniq_vals = uniq_vals[mask]
                weights = weights[mask]

    return uniq_vals, weights


