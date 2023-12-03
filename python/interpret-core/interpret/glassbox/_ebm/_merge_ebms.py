# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

from math import isnan
from ._utils import (
    remove_unused_higher_bins,
    order_terms,
    generate_term_names,
    process_terms,
    convert_categorical_to_continuous,
    deduplicate_bins,
)
from ...utils._native import Native

import numpy as np
import warnings
from itertools import count, chain

import logging

_log = logging.getLogger(__name__)


def _harmonize_tensor(
    new_feature_idxs,
    new_bounds,
    new_bins,
    old_feature_idxs,
    old_bounds,
    old_bins,
    old_mapping,
    old_tensor,
    bin_evidence_weight,
):
    # TODO: don't pass in new_bound and old_bounds.  We use the bounds to proportion
    # weights at the tail ends of the graphs, but the problem with that is that
    # you can have outliers that'll stretch the weight very thin.  If you have an
    # old_min of -10000000 but the lowest old cut is at 0.  If you have a new cut
    # at -100, it'll put very very close to 0 weight in the region from -100 to 0.
    # Instead of using the min/max to proportionate, we should start from the
    # lowest new_bins cut and then find all the other models that have that exact
    # same lowest cut (averaging their results). There must be at least 1 model with that cut.  After we
    # find that other model, we can use the weights in existing bin_weights to
    # proportionate the regions from the new lowest bin cut to the old lowest bin cut.
    # Do the same for the highest bin cut.  One issue is that the model(s) that have
    # the exact lowest bin cut are unlikely to share the lowest old cut, so we
    # proportionate the bin in the other model to the other model's next cut that is
    # greater than the old model's lowest cut.
    # eg:  new:      |    |            |   |    |
    #      old:                        |        |
    #   other1:      |    |   proprotion   |
    #   other2:      |        proportion        |
    # One wrinkle is that for pairs, we'll be using the pair cuts and we need to
    # one-dimensionalize any existing pair weights onto their respective 1D axies
    # before proportionating them.  Annother issue is that we might not even have
    # another term_feature that uses some particular feature that we use in our model
    # so we don't have any weights.  We can solve that issue by dropping any feature's
    # bins for terms that we have no information for.  After we do this we'll have
    # guaranteed that we only have new bin cuts for feature axies that we have inside
    # the bin level that we're handling!

    old_feature_idxs = list(old_feature_idxs)

    axes = []
    for feature_idx in new_feature_idxs:
        old_idx = old_feature_idxs.index(feature_idx)
        old_feature_idxs[old_idx] = -1  # in case we have duplicate feature idxs
        axes.append(old_idx)

    if bin_evidence_weight is not None:
        bin_evidence_weight = bin_evidence_weight.transpose(tuple(axes))

    n_multiclasses = 1
    if len(axes) != old_tensor.ndim:
        # multiclass. The last dimension always stays put
        axes.append(len(axes))
        n_multiclasses = old_tensor.shape[-1]

    old_tensor = old_tensor.transpose(tuple(axes))

    mapping = []
    lookups = []
    percentages = []
    for feature_idx in new_feature_idxs:
        old_bin_levels = old_bins[feature_idx]
        old_feature_bins = old_bin_levels[
            min(len(old_bin_levels), len(old_feature_idxs)) - 1
        ]

        mapping_levels = old_mapping[feature_idx]
        old_feature_mapping = mapping_levels[
            min(len(mapping_levels), len(old_feature_idxs)) - 1
        ]
        if old_feature_mapping is None:
            old_feature_mapping = list(
                (x,)
                for x in range(
                    len(old_feature_bins)
                    + (2 if isinstance(old_feature_bins, dict) else 3)
                )
            )
        mapping.append(old_feature_mapping)

        new_bin_levels = new_bins[feature_idx]
        new_feature_bins = new_bin_levels[
            min(len(new_bin_levels), len(new_feature_idxs)) - 1
        ]

        if isinstance(new_feature_bins, dict):
            # categorical feature

            old_reversed = dict()
            for category, bin_idx in old_feature_bins.items():
                category_list = old_reversed.get(bin_idx)
                if category_list is None:
                    old_reversed[bin_idx] = [category]
                else:
                    category_list.append(category)

            new_reversed = dict()
            for category, bin_idx in new_feature_bins.items():
                category_list = new_reversed.get(bin_idx)
                if category_list is None:
                    new_reversed[bin_idx] = [category]
                else:
                    category_list.append(category)
            new_reversed = sorted(new_reversed.items())

            lookup = [0]
            percentage = [1.0]
            for _, new_categories in new_reversed:
                # if there are two items in new_categories then they should both resolve
                # to the same index in old_feature_bins otherwise they would have been
                # split into two categories
                old_bin_idx = old_feature_bins.get(new_categories[0], -1)
                if 0 <= old_bin_idx:
                    percentage.append(
                        len(new_categories) / len(old_reversed[old_bin_idx])
                    )
                else:
                    # map to the unknown bin for scores, but take no percentage of the weight
                    percentage.append(0.0)
                lookup.append(old_bin_idx)
            percentage.append(1.0)
            lookup.append(-1)
        else:
            # continuous feature

            lookup = list(
                np.searchsorted(old_feature_bins, new_feature_bins, side="left") + 1
            )
            lookup.append(len(old_feature_bins) + 1)

            percentage = [1.0]
            for new_idx_minus_one, old_idx in enumerate(lookup):
                if new_idx_minus_one == 0:
                    new_low = new_bounds[feature_idx, 0]
                    # TODO: if nan OR out of bounds from the cuts, estimate it.
                    # If -inf or +inf, change it to min/max for float
                else:
                    new_low = new_feature_bins[new_idx_minus_one - 1]

                if len(new_feature_bins) <= new_idx_minus_one:
                    new_high = new_bounds[feature_idx, 1]
                    # TODO: if nan OR out of bounds from the cuts, estimate it.
                    # If -inf or +inf, change it to min/max for float
                else:
                    new_high = new_feature_bins[new_idx_minus_one]

                if old_idx == 1:
                    old_low = old_bounds[feature_idx, 0]
                    # TODO: if nan OR out of bounds from the cuts, estimate it.
                    # If -inf or +inf, change it to min/max for float
                else:
                    old_low = old_feature_bins[old_idx - 2]

                if len(old_feature_bins) < old_idx:
                    old_high = old_bounds[feature_idx, 1]
                    # TODO: if nan OR out of bounds from the cuts, estimate it.
                    # If -inf or +inf, change it to min/max for float
                else:
                    old_high = old_feature_bins[old_idx - 1]

                if old_high <= new_low or new_high <= old_low:
                    # if there are bins in the area above where the old data extended, then
                    # we'll have zero contribution in the old data where these new bins are
                    # located
                    percentage.append(0.0)
                else:
                    if new_low < old_low:
                        # this can't happen except at the lowest bin where the new min can be
                        # lower than the old min.  In that case we know the old data
                        # had zero contribution between the new min to the old min.
                        new_low = old_low

                    if old_high < new_high:
                        # this can't happen except at the lowest bin where the new max can be
                        # higher than the old max.  In that case we know the old data
                        # had zero contribution between the new max to the old max.
                        new_high = old_high

                    percentage.append((new_high - new_low) / (old_high - old_low))

            percentage.append(1.0)
            lookup.insert(0, 0)
            lookup.append(-1)

        lookups.append(lookup)
        percentages.append(percentage)

    new_shape = tuple(len(lookup) for lookup in lookups)
    n_cells = np.prod(new_shape)

    if 1 < n_multiclasses:
        # for multiclass we need to add another dimension for the scores of each class
        new_shape += (n_multiclasses,)

    lookups.reverse()
    percentages.reverse()
    mapping.reverse()

    # now we need to inflate it
    intermediate_shape = (n_cells, n_multiclasses) if 1 < n_multiclasses else n_cells
    new_tensor = np.empty(intermediate_shape, np.float64)
    for cell_idx in range(n_cells):
        remainder = cell_idx
        old_reversed_bin_idxs = []
        frac = 1.0
        for lookup, percentage in zip(lookups, percentages):
            n_bins = len(lookup)
            new_bin_idx = remainder % n_bins
            remainder //= n_bins
            old_reversed_bin_idxs.append(lookup[new_bin_idx])
            frac *= percentage[new_bin_idx]

        cell_map = [
            map_bins[bin_idx]
            for map_bins, bin_idx in zip(mapping, old_reversed_bin_idxs)
        ]
        n_cells2 = np.prod([len(x) for x in cell_map])
        val = np.zeros(n_multiclasses, np.float64)
        total_weight = 0.0
        for cell2_idx in range(n_cells2):
            remainder2 = cell2_idx
            old_reversed_bin2_idxs = []
            for lookup2 in cell_map:
                n_bins2 = len(lookup2)
                new_bin2_idx = remainder2 % n_bins2
                remainder2 //= n_bins2
                old_reversed_bin2_idxs.append(lookup2[new_bin2_idx])
            update = old_tensor[tuple(reversed(old_reversed_bin2_idxs))]
            if n_cells2 == 1:
                # if there's just one cell, which is typical, don't
                # incur the floating point loss in precision
                val = update
            else:
                if bin_evidence_weight is not None:
                    evidence_weight = bin_evidence_weight[
                        tuple(reversed(old_reversed_bin2_idxs))
                    ]
                    update *= evidence_weight
                    total_weight += evidence_weight
                val += update
        if bin_evidence_weight is None:
            # we're doing a bin weight and NOT a score tensor
            val *= frac
        elif total_weight != 0.0:
            # we're doing scores and we need to take a weighted average
            # but if the total_weight is zero then val should be zero and
            # our update should still be zero, which it already is
            val = val / total_weight
        new_tensor[cell_idx] = val
    new_tensor = new_tensor.reshape(new_shape)
    return new_tensor


def merge_ebms(models):
    """Merges EBM models trained on similar datasets that have the same set of features.

    Args:
        models: List of EBM models to be merged.

    Returns:
        An EBM model with averaged mean and standard deviation of input models.
    """

    if len(models) == 0:  # pragma: no cover
        raise Exception("0 models to merge.")

    model_types = list(set(map(type, models)))
    if len(model_types) == 2:
        type_names = [model_type.__name__ for model_type in model_types]
        if (
            "ExplainableBoostingClassifier" in type_names
            and "DPExplainableBoostingClassifier" in type_names
        ):
            ebm_type = model_types[type_names.index("ExplainableBoostingClassifier")]
            is_classification = True
            is_dp = False
        elif (
            "ExplainableBoostingRegressor" in type_names
            and "DPExplainableBoostingRegressor" in type_names
        ):
            ebm_type = model_types[type_names.index("ExplainableBoostingRegressor")]
            is_classification = False
            is_dp = False
        else:
            raise Exception("Inconsistent model types attempting to be merged.")
    elif len(model_types) == 1:
        ebm_type = model_types[0]
        if ebm_type.__name__ == "ExplainableBoostingClassifier":
            is_classification = True
            is_dp = False
        elif ebm_type.__name__ == "DPExplainableBoostingClassifier":
            is_classification = True
            is_dp = True
        elif ebm_type.__name__ == "ExplainableBoostingRegressor":
            is_classification = False
            is_dp = False
        elif ebm_type.__name__ == "DPExplainableBoostingRegressor":
            is_classification = False
            is_dp = True
        else:
            raise Exception(
                f"Invalid EBM model type {ebm_type.__name__} attempting to be merged."
            )
    else:
        raise Exception("Inconsistent model types being merged.")

    # TODO: create the ExplainableBoostingClassifier etc, type directly
    # by name instead of using __new__ from ebm_type
    ebm = ebm_type.__new__(ebm_type)

    if any(
        not getattr(model, "has_fitted_", False) for model in models
    ):  # pragma: no cover
        raise Exception("All models must be fitted.")
    ebm.has_fitted_ = True

    link = models[0].link_
    if any(model.link_ != link for model in models):
        raise Exception("Models with different link functions cannot be merged")
    ebm.link_ = link

    link_param = models[0].link_param_
    if isnan(link_param):
        if not all(isnan(model.link_param_) for model in models):
            raise Exception("Models with different link param values cannot be merged")
    else:
        if any(model.link_param_ != link_param for model in models):
            raise Exception("Models with different link param values cannot be merged")
    ebm.link_param_ = link_param

    # self.bins_ is the only feature based attribute that we absolutely require
    n_features = len(models[0].bins_)

    for model in models:
        if n_features != len(model.bins_):  # pragma: no cover
            raise Exception("Inconsistent numbers of features in the models.")

        feature_names_in = getattr(model, "feature_names_in_", None)
        if feature_names_in is not None:
            if n_features != len(feature_names_in):  # pragma: no cover
                raise Exception("Inconsistent numbers of features in the models.")

        feature_types_in = getattr(model, "feature_types_in_", None)
        if feature_types_in is not None:
            if n_features != len(feature_types_in):  # pragma: no cover
                raise Exception("Inconsistent numbers of features in the models.")

        feature_bounds = getattr(model, "feature_bounds_", None)
        if feature_bounds is not None:
            if n_features != feature_bounds.shape[0]:  # pragma: no cover
                raise Exception("Inconsistent numbers of features in the models.")

        histogram_weights = getattr(model, "histogram_weights_", None)
        if histogram_weights is not None:
            if n_features != len(histogram_weights):  # pragma: no cover
                raise Exception("Inconsistent numbers of features in the models.")

        unique_val_counts = getattr(model, "unique_val_counts_", None)
        if unique_val_counts is not None:
            if n_features != len(unique_val_counts):  # pragma: no cover
                raise Exception("Inconsistent numbers of features in the models.")

    old_bounds = []
    old_mapping = []
    old_bins = []
    for model in models:
        if any(len(set(map(type, bin_levels))) != 1 for bin_levels in model.bins_):
            raise Exception("Inconsistent bin types within a model.")

        feature_bounds = getattr(model, "feature_bounds_", None)
        if feature_bounds is None:
            old_bounds.append(None)
        else:
            old_bounds.append(feature_bounds.copy())

        old_mapping.append([[] for _ in range(n_features)])
        old_bins.append([[] for _ in range(n_features)])

    # TODO: every time we merge models we fragment the bins more and more and this is undesirable
    # especially for pairs.  When we build models, we store the feature bin cuts for pairs even
    # if we have no pairs that use that paritcular feature as a pair.  We can eliminate these useless
    # pair feature cuts before merging the bins and that'll give us less resulting cuts.  Having less
    # cuts reduces the number of estimates that we need to make and reduces the complexity of the
    # tensors, so it's good to have this reduction.

    new_feature_types = []
    new_bins = []
    for feature_idx in range(n_features):
        bin_types = set(type(model.bins_[feature_idx][0]) for model in models)

        if len(bin_types) == 1 and next(iter(bin_types)) is dict:
            # categorical
            new_feature_type = None
            for model in models:
                feature_types_in = getattr(model, "feature_types_in_", None)
                if feature_types_in is not None:
                    feature_type = feature_types_in[feature_idx]
                    if feature_type == "nominal":
                        new_feature_type = "nominal"
                    elif feature_type == "ordinal" and new_feature_type is None:
                        new_feature_type = "ordinal"
            if new_feature_type is None:
                new_feature_type = "nominal"
        else:
            # continuous
            if any(bin_type not in {dict, np.ndarray} for bin_type in bin_types):
                raise Exception("Invalid bin type.")
            new_feature_type = "continuous"
        new_feature_types.append(new_feature_type)

        level_end = max(len(model.bins_[feature_idx]) for model in models)
        new_leveled_bins = []
        for level_idx in range(level_end):
            model_bins = []
            for model_idx, model in enumerate(models):
                bin_levels = model.bins_[feature_idx]
                bin_level = bin_levels[min(level_idx, len(bin_levels) - 1)]
                model_bins.append(bin_level)

                old_mapping[model_idx][feature_idx].append(None)
                old_bins[model_idx][feature_idx].append(bin_level)

            if len(bin_types) == 1 and next(iter(bin_types)) is dict:
                # categorical
                merged_keys = sorted(
                    set(chain.from_iterable(bin.keys() for bin in model_bins))
                )
                # TODO: for now we just support alphabetical ordering in merged models, but
                # we could do all sort of special processing like trying to figure out if the original
                # ordering was by prevalence or alphabetical and then attempting to preserve that
                # order and also handling merged categories (where two categories map to a single score)
                # We should first try to progress in order along each set of keys and see if we can
                # establish the perfect order which might work if there are isolated missing categories
                # and if we can't get a unique guaranteed sorted order that way then examime all the
                # different known sort order and figure out if any of the possible orderings match
                merged_bins = dict(zip(merged_keys, count(1)))
            else:
                # continuous

                if 1 != len(bin_types):
                    # We have both categorical and continuous.  We can't convert continuous
                    # to categorical since we lack the original labels, but we can convert
                    # categoricals to continuous.  If the feature flavors are similar, which
                    # needs to be the case for model merging, one of the models only found
                    # float64 in their data, so there shouldn't be a lot of non-float values
                    # in the other models.

                    for model_idx, bins_in_model in enumerate(model_bins):
                        if isinstance(bins_in_model, dict):
                            (
                                converted_bins,
                                mapping,
                                converted_min,
                                converted_max,
                            ) = convert_categorical_to_continuous(bins_in_model)
                            model_bins[model_idx] = converted_bins

                            old_min = old_bounds[model_idx][feature_idx][0]
                            if isnan(old_min) or converted_min < old_min:
                                old_bounds[model_idx][feature_idx][0] = converted_min

                            old_max = old_bounds[model_idx][feature_idx][1]
                            if isnan(old_max) or old_max < converted_max:
                                old_bounds[model_idx][feature_idx][1] = converted_max

                            old_bins[model_idx][feature_idx][level_idx] = converted_bins
                            old_mapping[model_idx][feature_idx][level_idx] = mapping

                merged_bins = np.array(
                    sorted(set(chain.from_iterable(model_bins))), np.float64
                )
            new_leveled_bins.append(merged_bins)
        new_bins.append(new_leveled_bins)
    ebm.feature_types_in_ = new_feature_types
    deduplicate_bins(new_bins)
    ebm.bins_ = new_bins

    feature_names_merged = [None] * n_features
    for model in models:
        feature_names_in = getattr(model, "feature_names_in_", None)
        if feature_names_in is not None:
            for feature_idx, feature_name in enumerate(feature_names_in):
                if feature_name is not None:
                    feature_name_merged = feature_names_merged[feature_idx]
                    if feature_name_merged is None:
                        feature_names_merged[feature_idx] = feature_name
                    elif feature_name != feature_name_merged:
                        raise Exception(
                            "All models should have the same feature names."
                        )
    if any(feature_name is not None for feature_name in feature_names_merged):
        ebm.feature_names_in_ = feature_names_merged

    min_feature_vals = [bounds[:, 0] for bounds in old_bounds if bounds is not None]
    max_feature_vals = [bounds[:, 1] for bounds in old_bounds if bounds is not None]
    if 0 < len(min_feature_vals):  # max_feature_vals has the same len
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            min_feature_vals = np.nanmin(min_feature_vals, axis=0)
            max_feature_vals = np.nanmax(max_feature_vals, axis=0)
            if any(not isnan(val) for val in min_feature_vals) or any(
                not isnan(val) for val in max_feature_vals
            ):
                ebm.feature_bounds_ = np.array(
                    list(zip(min_feature_vals, max_feature_vals)), np.float64
                )

    if not is_dp:
        if all(
            hasattr(model, "histogram_weights_") and hasattr(model, "feature_bounds_")
            for model in models
        ):
            if hasattr(ebm, "feature_bounds_"):
                # TODO: estimate the histogram bin counts by taking the min of the mins and the max of the maxes
                # and re-apportioning the counts based on the distributions of the previous histograms.  Proprotion
                # them to the floor of their counts and then assign any remaining integers based on how much
                # they reduce the RMSE of the integer counts from the ideal floating point counts.
                pass

    if is_classification:
        ebm.classes_ = models[0].classes_.copy()
        if any(not np.array_equal(ebm.classes_, model.classes_) for model in models):
            # pragma: no cover
            raise Exception("The target classes should be identical.")

        n_classes = len(ebm.classes_)
    else:
        if any(hasattr(model, "min_target_") for model in models):
            ebm.min_target_ = min(
                model.min_target_ for model in models if hasattr(model, "min_target_")
            )
        if any(hasattr(model, "max_target_") for model in models):
            ebm.max_target_ = max(
                model.max_target_ for model in models if hasattr(model, "max_target_")
            )
        n_classes = Native.Task_Regression
    n_scores = Native.get_count_scores_c(n_classes)

    bag_weights = []
    model_weights = []
    bagged_intercept = []
    for model in models:
        avg_weight = np.average([tensor.sum() for tensor in model.bin_weights_])
        model_weights.append(avg_weight)

        n_outer_bags = -1
        if hasattr(model, "bagged_scores_"):
            if 0 < len(model.bagged_scores_):
                n_outer_bags = len(model.bagged_scores_[0])

        model_bag_weights = getattr(model, "bag_weights_", None)
        if model_bag_weights is None:
            # this model wasn't the result of a merge, so get the total weight for the model
            # every term in a model should have the same weight, but perhaps the user edited
            # the model weights and they don't agree.  We handle these by taking the average
            model_bag_weights = [avg_weight] * n_outer_bags
        elif len(model_bag_weights) != n_outer_bags:
            raise Exception(
                "self.bagged_weights_ should have the same length as n_outer_bags."
            )

        model_bag_intercept = getattr(model, "bagged_intercept_", None)
        if model_bag_intercept is None:
            if n_scores == 1:
                model_bag_intercept = np.zeros(n_outer_bags, np.float64)
            else:
                model_bag_intercept = np.zeros((n_outer_bags, n_scores), np.float64)

        bagged_intercept.extend(model_bag_intercept)
        bag_weights.extend(model_bag_weights)
    ebm.bag_weights_ = bag_weights
    ebm.bagged_intercept_ = np.array(bagged_intercept, np.float64)

    fg_dicts = []
    all_fg = set()
    for model in models:
        fg_sorted = [
            tuple(sorted(feature_idxs)) for feature_idxs in model.term_features_
        ]
        fg_dicts.append(dict(zip(fg_sorted, count())))
        all_fg.update(fg_sorted)

    sorted_fgs = order_terms(list(all_fg))

    # TODO: in the future we might at this point try and figure out the most
    #       common feature ordering within the terms.  Take the mode first
    #       and amonst the orderings that tie, choose the one that's best sorted by
    #       feature indexes
    ebm.term_features_ = sorted_fgs

    ebm.bin_weights_ = []
    ebm.bagged_scores_ = []
    for sorted_fg in sorted_fgs:
        # since interactions are often automatically generated, we'll often always have
        # interaction mismatches where an interaction will be in one model, but not the other.
        # We need to estimate the bin_weight_ tensors that would have existed in this case.
        # We'll use the interaction terms that we do have in other models to estimate the
        # distribution in the essense of the data, which should be roughly consistent or you
        # shouldn't be attempting to merge the models in the first place.  We'll then scale
        # the percentage distribution by the total weight of the model that we're fillin in the
        # details for.

        # TODO: this algorithm has some problems.  The estimated tensor that we get by taking the
        # model weight and distributing it by a per-cell percentage measure means that we get
        # inconsistent weight distibutions along the axis.  We can take our resulting weight tensor
        # and sum the columns/rows to get the weights on each individual feature axis.  Our model
        # however comes with a known set of weights on each feature, and the result of our operation
        # will not match the existing distribution in almost all cases.  I think there might be
        # some algorithm where we start with the per-feature weights and use the distribution hints
        # from the other models to inform where we place our exact weights that we know about in our
        # model from each axis.  The problem is that the sums in both axies need to agree, and each
        # change we make influences both.  I'm not sure we can even guarantee that there is an answer
        # and if there was one I'm not sure how we'd go about generating it.  I'm going to leave
        # this problem for YOU: a future person who is smarter than me and has more time to solve this.
        # One hint: I think a possible place to start would be an iterative algorithm that's similar
        # to purification where you randomly select a row/column and try to get closer at each step
        # to the rigth answer.  Good luck!
        #
        # Oh, there's also another deeper problem.. let's say you had a crazy 5 way interaction in the
        # model eg: (0,1,2,3,4) and you had 2 and 3 way interactions that either overlap or not.
        # Eg: (0,1), and either (1,2,3) or (2,3,4).  The ideal solution would take the 5 way interaction
        # and look for all the possible combinations of interactions for further information it could
        # use and then it would make something that is consistent across all of these disparate sources
        # of information.  Hopefully, the user hasn't edited the model in a way that creates no solution.

        bin_weight_percentages = []
        for model_idx, model, fg_dict, model_weight in zip(
            count(), models, fg_dicts, model_weights
        ):
            term_idx = fg_dict.get(sorted_fg)
            if term_idx is not None:
                fixed_tensor = _harmonize_tensor(
                    sorted_fg,
                    ebm.feature_bounds_,
                    ebm.bins_,
                    model.term_features_[term_idx],
                    old_bounds[model_idx],
                    old_bins[model_idx],
                    old_mapping[model_idx],
                    model.bin_weights_[term_idx],
                    None,
                )
                bin_weight_percentages.append(fixed_tensor * model_weight)

        # use this when we don't have a term in a model as a reasonable
        # set of guesses for the distribution of the weight of the model
        bin_weight_percentages = np.sum(bin_weight_percentages, axis=0)
        bin_weight_percentages = bin_weight_percentages / bin_weight_percentages.sum()

        additive_shape = bin_weight_percentages.shape
        if 0 <= n_classes and n_classes != 2:
            additive_shape = tuple(list(additive_shape) + [n_classes])

        new_bin_weights = []
        new_bagged_scores = []
        for model_idx, model, fg_dict, model_weight in zip(
            count(), models, fg_dicts, model_weights
        ):
            n_outer_bags = -1
            if hasattr(model, "bagged_scores_"):
                if 0 < len(model.bagged_scores_):
                    n_outer_bags = len(model.bagged_scores_[0])

            term_idx = fg_dict.get(sorted_fg)
            if term_idx is None:
                new_bin_weights.append(model_weight * bin_weight_percentages)
                new_bagged_scores.extend(
                    n_outer_bags * [np.zeros(additive_shape, np.float64)]
                )
            else:
                harmonized_bin_weights = _harmonize_tensor(
                    sorted_fg,
                    ebm.feature_bounds_,
                    ebm.bins_,
                    model.term_features_[term_idx],
                    old_bounds[model_idx],
                    old_bins[model_idx],
                    old_mapping[model_idx],
                    model.bin_weights_[term_idx],
                    None,
                )
                new_bin_weights.append(harmonized_bin_weights)
                for bag_idx in range(n_outer_bags):
                    harmonized_bagged_scores = _harmonize_tensor(
                        sorted_fg,
                        ebm.feature_bounds_,
                        ebm.bins_,
                        model.term_features_[term_idx],
                        old_bounds[model_idx],
                        old_bins[model_idx],
                        old_mapping[model_idx],
                        model.bagged_scores_[term_idx][bag_idx],
                        model.bin_weights_[
                            term_idx
                        ],  # we use these to weigh distribution of scores for mulple bins
                    )
                    new_bagged_scores.append(harmonized_bagged_scores)
        ebm.bin_weights_.append(np.sum(new_bin_weights, axis=0))
        ebm.bagged_scores_.append(np.array(new_bagged_scores, np.float64))

    (
        ebm.intercept_,
        ebm.term_scores_,
        ebm.standard_deviations_,
    ) = process_terms(
        ebm.bagged_intercept_, ebm.bagged_scores_, ebm.bin_weights_, ebm.bag_weights_
    )
    if n_classes < 0:
        # scikit-learn uses a float for regression, and a numpy array with 1 element for binary classification
        ebm.intercept_ = float(ebm.intercept_[0])

    # TODO: we might be able to do these operations earlier
    remove_unused_higher_bins(ebm.term_features_, ebm.bins_)
    # removing the higher order terms might allow us to eliminate some extra bins now that couldn't before
    deduplicate_bins(ebm.bins_)

    # dependent attributes (can be re-derrived after serialization)
    ebm.n_features_in_ = len(ebm.bins_)  # scikit-learn specified name
    ebm.term_names_ = generate_term_names(ebm.feature_names_in_, ebm.term_features_)

    return ebm
