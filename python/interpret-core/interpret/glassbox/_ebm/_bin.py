# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

import logging

import numpy as np
from itertools import count, tee, repeat
import operator

from ...utils._clean_x import unify_columns
from ...utils._native import Native

_log = logging.getLogger(__name__)


_none_list = [None]
_none_ndarray = np.array(None)


def eval_terms(X, n_samples, feature_names_in, feature_types_in, bins, term_features):
    # called under: predict

    # prior to calling this function, call deduplicate_bins which will eliminate extra work in this function

    # this generator function returns data in whatever order it thinks is most efficient.  Normally for
    # mains it returns them in order, but pairs will be returned as their data completes and they can
    # be mixed in with mains.  So, if we request data for [(0), (1), (2), (3), (4), (1, 3)] the return sequence
    # could be [(0), (1), (2), (3), (1, 3), (4)].  More complicated pair/triples return even more randomized ordering.
    # For additive models the results can be processed in any order, so this imposes no penalities on us.

    _log.info("eval_terms")

    request_feature_idxs = []
    request_categories = []

    waiting = {}
    # the first len(term_feature_idxs) items hold the binned data that we get back as it arrives
    for requirements, term_feature_idxs in zip(
        map(
            operator.add,
            map(_none_list.__mul__, map(len, term_features)),
            map(list, zip(count())),
        ),
        term_features,
    ):
        for feature_idx in term_feature_idxs:
            bin_levels = bins[feature_idx]
            feature_bins = bin_levels[min(len(bin_levels), len(term_feature_idxs)) - 1]
            if isinstance(feature_bins, dict):
                # categorical feature
                key = (feature_idx, id(feature_bins))
            else:
                # continuous feature
                feature_bins = None
                key = feature_idx
            waiting_list = waiting.get(key)
            if waiting_list is None:
                request_feature_idxs.append(feature_idx)
                request_categories.append(feature_bins)
                waiting[key] = [requirements]
            else:
                waiting_list.append(requirements)

    native = Native.get_native_singleton()

    for column_feature_idx, (_, X_col, column_categories, bad) in zip(
        request_feature_idxs,
        unify_columns(
            X,
            request_feature_idxs,
            request_categories,
            feature_names_in,
            feature_types_in,
            None,
            True,
        ),
    ):
        if n_samples != len(X_col):
            msg = "The columns of X are mismatched in the number of of samples"
            _log.error(msg)
            raise ValueError(msg)

        if column_categories is None:
            # continuous feature

            if bad is not None:
                # TODO: we could pass out a bool array instead of objects for this function only
                bad = bad != _none_ndarray

            if not X_col.flags.c_contiguous:
                # we requrested this feature, so at some point we're going to call discretize,
                # which requires contiguous memory
                X_col = X_col.copy()

            bin_levels = bins[column_feature_idx]
            max_level = len(bin_levels)
            binning_completed = _none_list * max_level
            for requirements in waiting[column_feature_idx]:
                if len(requirements) != 0:
                    term_idx = requirements[-1]
                    feature_idxs = term_features[term_idx]
                    is_done = True
                    for dimension_idx, term_feature_idx in enumerate(feature_idxs):
                        if term_feature_idx == column_feature_idx:
                            level_idx = min(max_level, len(feature_idxs)) - 1
                            bin_indexes = binning_completed[level_idx]
                            if bin_indexes is None:
                                cuts = bin_levels[level_idx]
                                bin_indexes = native.discretize(X_col, cuts)
                                if bad is not None:
                                    bin_indexes[bad] = -1
                                binning_completed[level_idx] = bin_indexes
                            requirements[dimension_idx] = bin_indexes
                        elif requirements[dimension_idx] is None:
                            is_done = False

                    if is_done:
                        yield term_idx, requirements[:-1]
                        # clear references so that the garbage collector can free them
                        requirements.clear()
        else:
            # categorical feature

            if bad is not None:
                # TODO: we could pass out a single bool (not an array) if these aren't continuous convertible
                pass  # TODO: improve this handling

            for requirements in waiting[(column_feature_idx, id(column_categories))]:
                if len(requirements) != 0:
                    term_idx = requirements[-1]
                    feature_idxs = term_features[term_idx]
                    is_done = True
                    for dimension_idx, term_feature_idx in enumerate(feature_idxs):
                        if term_feature_idx == column_feature_idx:
                            # "term_categories is column_categories" since any term in the waiting_list must have
                            # one of it's elements match this (feature_idx, categories) index, and all items in this
                            # term need to have the same categories since they came from the same bin_level
                            requirements[dimension_idx] = X_col
                        elif requirements[dimension_idx] is None:
                            is_done = False

                    if is_done:
                        yield term_idx, requirements[:-1]
                        # clear references so that the garbage collector can free them
                        requirements.clear()


def ebm_predict_scores(
    X,
    n_samples,
    feature_names_in,
    feature_types_in,
    bins,
    intercept,
    term_scores,
    term_features,
    init_score=None,
):
    sample_scores = (
        np.full(
            n_samples
            if isinstance(intercept, float) or len(intercept) == 1
            else (n_samples, len(intercept)),
            intercept,
            dtype=np.float64,
        )
        if init_score is None
        else init_score + intercept
    )

    if n_samples > 0:
        term_idxs, binned = tee(
            eval_terms(
                X, n_samples, feature_names_in, feature_types_in, bins, term_features
            ),
            2,
        )

        # sum is used to iterate outside the interpreter. The result is not used.
        sum(
            map(
                operator.is_,
                map(
                    sample_scores.__iadd__,
                    map(
                        operator.getitem,
                        map(
                            term_scores.__getitem__,
                            map(operator.itemgetter(0), term_idxs),
                        ),
                        map(tuple, map(operator.itemgetter(1), binned)),
                    ),
                ),
                repeat(None),
            )
        )

    return sample_scores


def ebm_eval_terms(
    X,
    n_samples,
    n_scores,
    feature_names_in,
    feature_types_in,
    bins,
    term_scores,
    term_features,
):
    explanations = np.empty(
        (n_samples, len(term_features))
        if n_scores == 1
        else (n_samples, len(term_features), n_scores),
        dtype=np.float64,
    )

    if n_samples > 0:
        term_idxs1, term_idxs2, binned = tee(
            eval_terms(
                X, n_samples, feature_names_in, feature_types_in, bins, term_features
            ),
            3,
        )

        # sum is used to iterate outside the interpreter. The result is not used.
        sum(
            map(
                operator.truth,
                map(
                    explanations.__setitem__,
                    zip(
                        repeat(slice(None)),
                        map(operator.itemgetter(0), term_idxs1),
                    ),
                    map(
                        operator.getitem,
                        map(
                            term_scores.__getitem__,
                            map(operator.itemgetter(0), term_idxs2),
                        ),
                        map(tuple, map(operator.itemgetter(1), binned)),
                    ),
                ),
            )
        )

    return explanations


def make_bin_weights(
    X, n_samples, sample_weight, feature_names_in, feature_types_in, bins, term_features
):
    bin_weights = _none_list * len(term_features)
    for term_idx, bin_indexes in eval_terms(
        X, n_samples, feature_names_in, feature_types_in, bins, term_features
    ):
        feature_idxs = term_features[term_idx]
        multiple = 1
        dimensions = []
        for dimension_idx in range(len(feature_idxs) - 1, -1, -1):
            feature_idx = feature_idxs[dimension_idx]
            bin_levels = bins[feature_idx]
            feature_bins = bin_levels[min(len(bin_levels), len(feature_idxs)) - 1]
            if isinstance(feature_bins, dict):
                # categorical feature
                n_bins = 2 if len(feature_bins) == 0 else max(feature_bins.values()) + 2
            else:
                # continuous feature
                n_bins = len(feature_bins) + 3

            dimensions.append(n_bins)
            dim_data = bin_indexes[dimension_idx]
            dim_data = np.where(dim_data < 0, n_bins - 1, dim_data)
            if multiple == 1:
                flat_indexes = dim_data
            else:
                flat_indexes += dim_data * multiple
            multiple *= n_bins
        dimensions = tuple(reversed(dimensions))

        if sample_weight is None:
            term_bin_weights = np.bincount(flat_indexes, minlength=multiple)
        else:
            term_bin_weights = np.bincount(
                flat_indexes, weights=sample_weight, minlength=multiple
            )
        term_bin_weights = term_bin_weights.astype(np.float64, copy=False)
        term_bin_weights = term_bin_weights.reshape(dimensions)
        bin_weights[term_idx] = term_bin_weights

    return bin_weights
