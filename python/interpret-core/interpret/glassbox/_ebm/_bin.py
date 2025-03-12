# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

import logging

import numpy as np
from itertools import count, tee, repeat, chain, compress
from operator import mul, itemgetter, getitem, truth, add, attrgetter, is_not, not_, is_

from ...utils._clean_x import unify_columns
from ...utils._native import Native

_log = logging.getLogger(__name__)


_none_list = [None]
_make_non_list = _none_list.__mul__
_none_ndarray = np.array(None)
_repeat_none = repeat(None)
_repeat_dict = repeat(dict)
_make_new_lists = map(list, repeat(tuple()))
_sub_one = (-1).__add__
_eq_getter = attrgetter("__eq__")
_flags_getter = attrgetter("flags")
_continuous_getter = attrgetter("c_contiguous")
_itemgetter0 = itemgetter(0)
_itemgetter1 = itemgetter(1)
_itemgetter2 = itemgetter(2)
_itemgetter3 = itemgetter(3)
_slice_remove_last = slice(None, -1)


def eval_terms(X, n_samples, feature_names_in, feature_types_in, bins, term_features):
    # called under: predict

    # prior to calling this function, call deduplicate_bins which will eliminate extra work in this function

    # this generator function returns data in whatever order it thinks is most efficient.  Normally for
    # mains it returns them in order, but pairs will be returned as their data completes and they can
    # be mixed in with mains.  So, if we request data for [(0), (1), (2), (3), (4), (1, 3)] the return sequence
    # could be [(0), (1), (2), (3), (1, 3), (4)].  More complicated pair/triples return even more randomized ordering.
    # For additive models the results can be processed in any order, so this imposes no penalities on us.

    _log.info("eval_terms")

    all_requirements = list(
        chain.from_iterable(
            map(
                mul,
                zip(
                    map(
                        add,
                        map(_make_non_list, map(len, term_features)),
                        map(list, zip(count())),
                    )
                ),
                map(len, term_features),
            )
        ),
    )

    all_bin_levels1, all_bin_levels2 = tee(
        map(bins.__getitem__, chain.from_iterable(term_features)), 2
    )

    feature_bins1, feature_bins2 = tee(
        map(
            getitem,
            all_bin_levels1,
            map(
                min,
                zip(
                    map((-1).__add__, map(len, all_bin_levels2)),
                    map((-2).__add__, map(len, all_requirements)),
                ),
            ),
        ),
        2,
    )

    all_feature_bins = list(
        map(
            getitem,
            zip(_repeat_none, feature_bins1),
            map(isinstance, feature_bins2, _repeat_dict),
        )
    )

    requests = dict(
        zip(
            zip(chain.from_iterable(term_features), map(id, all_feature_bins)),
            zip(count(), all_feature_bins),
        )
    )

    # Order requests by (feature_idx, term order) for implementation independence.
    # Since term_features is sorted by # dimensions, this also orders by # dimensions.
    requests = sorted(
        zip(
            map(_itemgetter0, requests.keys()),
            map(_itemgetter0, requests.values()),
            map(_itemgetter1, requests.values()),
        )
    )

    request_feature_idxs = list(map(_itemgetter0, requests))

    keys1, keys2 = tee(
        zip(chain.from_iterable(term_features), map(id, all_feature_bins)), 2
    )

    waiting = {}
    # sum is used to iterate outside the interpreter. The result is not used.
    sum(
        map(
            truth,
            map(
                waiting.__setitem__,
                keys1,
                map(
                    add,
                    map(waiting.get, keys2, _make_new_lists),
                    map(list, zip(all_requirements)),
                ),
            ),
        )
    )

    native = Native.get_native_singleton()

    col1, col2, col3, col4, col5 = tee(
        unify_columns(
            X,
            request_feature_idxs,
            map(_itemgetter2, requests),
            feature_names_in,
            feature_types_in,
            None,
            True,
        ),
        5,
    )

    for (
        column_feature_idx,
        column_feature_idx_eq,
        bin_levels,
        max_level,
        binning_completed,
        all_requirements,
        is_mismatch,
        is_bad,
        is_non_contiguous,
        (_, X_col, column_categories, bad),
    ) in zip(
        request_feature_idxs,
        map(_eq_getter, request_feature_idxs),
        map(bins.__getitem__, request_feature_idxs),
        map(len, map(bins.__getitem__, request_feature_idxs)),
        map(_make_non_list, map(len, map(bins.__getitem__, request_feature_idxs))),
        map(
            waiting.__getitem__,
            zip(request_feature_idxs, map(id, map(_itemgetter2, col1))),
        ),
        map(n_samples.__ne__, map(len, map(_itemgetter1, col2))),
        map(is_not, map(_itemgetter3, col3), _repeat_none),
        map(
            not_,
            map(
                _continuous_getter,
                map(_flags_getter, map(_itemgetter1, col4)),
            ),
        ),
        col5,
    ):
        if is_mismatch:
            msg = "The columns of X are mismatched in the number of of samples"
            _log.error(msg)
            raise ValueError(msg)

        if column_categories:
            # categorical feature

            # if is_bad:
            #     # TODO: we could pass out a single bool (not an array) if these aren't continuous convertible
            #     pass  # TODO: improve this handling

            for requirements in all_requirements:
                term_idx = requirements[-1]

                # if it was illegal to have duplicate features in a term we could do:
                # requirements[term_features[term_idx].index(column_feature_idx)] = X_col
                sum(
                    map(
                        truth,
                        map(
                            requirements.__setitem__,
                            compress(
                                count(),
                                map(column_feature_idx_eq, term_features[term_idx]),
                            ),
                            repeat(X_col),
                        ),
                    )
                )

                if all(map(is_not, requirements, _repeat_none)):
                    yield term_idx, requirements[_slice_remove_last]
                    # clear references so that the garbage collector can free them
                    requirements.clear()
        else:
            # continuous feature

            if is_bad:
                # TODO: we could pass out a bool array instead of objects for this function only
                bad = bad != _none_ndarray

            if is_non_contiguous:
                # we requrested this feature, so at some point we're going to call discretize,
                # which requires contiguous memory
                X_col = X_col.copy()

            for requirements in all_requirements:
                term_idx = requirements[-1]
                feature_idxs = term_features[term_idx]
                level_idx = _sub_one(min(max_level, len(feature_idxs)))
                bin_indexes = binning_completed[level_idx]
                if bin_indexes is None:
                    bin_indexes = native.discretize(X_col, bin_levels[level_idx])
                    if is_bad:
                        bin_indexes[bad] = -1
                    binning_completed[level_idx] = bin_indexes

                # if it was illegal to have duplicate features in a term we could do:
                # requirements[feature_idxs.index(column_feature_idx)] = bin_indexes
                sum(
                    map(
                        truth,
                        map(
                            requirements.__setitem__,
                            compress(count(), map(column_feature_idx_eq, feature_idxs)),
                            repeat(bin_indexes),
                        ),
                    )
                )

                if all(map(is_not, requirements, _repeat_none)):
                    yield term_idx, requirements[_slice_remove_last]
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
                is_,
                map(
                    sample_scores.__iadd__,
                    map(
                        getitem,
                        map(
                            term_scores.__getitem__,
                            map(_itemgetter0, term_idxs),
                        ),
                        map(tuple, map(_itemgetter1, binned)),
                    ),
                ),
                _repeat_none,
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
                truth,
                map(
                    explanations.__setitem__,
                    zip(
                        repeat(slice(None)),
                        map(_itemgetter0, term_idxs1),
                    ),
                    map(
                        getitem,
                        map(
                            term_scores.__getitem__,
                            map(_itemgetter0, term_idxs2),
                        ),
                        map(tuple, map(_itemgetter1, binned)),
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
