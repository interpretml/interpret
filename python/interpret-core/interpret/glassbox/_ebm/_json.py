# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

from warnings import warn
from ...utils._link import identify_task
from math import isnan
import numpy as np
from itertools import groupby


def jsonify_lists(vals):
    if len(vals) != 0:
        if type(vals[0]) is float:
            for idx, val in enumerate(vals):
                # JSON doesn't have NaN, or infinities, but javaScript
                # uses dynamic typing so we can use strings instead.
                if isnan(val):
                    vals[idx] = "nan"  # standardize as lower case for all characters
                elif val == np.inf:
                    vals[idx] = "+inf"  # use a '+' to allow searching for just +inf
                elif val == -np.inf:
                    vals[idx] = "-inf"
        else:
            for nested in vals:
                jsonify_lists(nested)
    return vals  # we modify in place, but return it just for easy access


def jsonify_item(val):
    # JSON doesn't have NaN, or infinities, but javaScript
    # uses dynamic typing so we can use strings instead.
    if isnan(val):
        val = "nan"  # standardize as lower case for all characters
    elif val == np.inf:
        val = "+inf"  # use a '+' to allow searching for just +inf
    elif val == -np.inf:
        val = "-inf"
    return val


def _to_json_inner(ebm, detail="all"):
    """Converts the inner model to a JSONable representation.

    Args:
        detail: 'minimal', 'interpretable', 'mergeable', 'all'

    Returns:
        JSONable object
    """

    if detail == "minimal":
        level = 0
    elif detail == "interpretable":
        level = 1
    elif detail == "mergeable":
        level = 2
    elif detail == "all":
        level = 3
    else:
        msg = f"Unrecognized to_json detail: {detail}"
        _log.error(msg)
        raise ValueError(msg)

    j = {}

    # future-proof support for multi-output models
    outputs = []
    output = {}
    task = identify_task(ebm.link_)
    output["task"] = task
    if task == "classification":
        output["classes"] = ebm.classes_.tolist()
    elif task == "regression":
        if 3 <= level:
            min_target = getattr(ebm, "min_target_", None)
            if min_target is not None and not isnan(min_target):
                output["min_target"] = jsonify_item(min_target)
            max_target = getattr(ebm, "max_target_", None)
            if max_target is not None and not isnan(max_target):
                output["max_target"] = jsonify_item(max_target)
    else:
        raise ValueError(f"Unsupported link function: {ebm.link_}")

    output["link"] = ebm.link_
    output["link_param"] = jsonify_item(ebm.link_param_)

    outputs.append(output)
    j["outputs"] = outputs

    if type(ebm.intercept_) is float:
        # scikit-learn requires that we have a single float value as our intercept for compatibility with
        # RegressorMixin, but in other scenarios where we want to support things like multi-output it would be
        # easier if the regression intercept were handled identically to classification, so put it in an array
        # for our JSON format to harmonize the cross-language representation
        j["intercept"] = [jsonify_item(ebm.intercept_)]
    else:
        j["intercept"] = jsonify_lists(ebm.intercept_.tolist())

    bagged_intercept = getattr(ebm, "bagged_intercept_", None)
    if bagged_intercept is not None:
        j["bagged_intercept"] = jsonify_lists(bagged_intercept.tolist())

    if 3 <= level:
        noise_scale_binning = getattr(ebm, "noise_scale_binning_", None)
        if noise_scale_binning is not None:
            j["noise_scale_binning"] = jsonify_item(noise_scale_binning)
        noise_scale_boosting = getattr(ebm, "noise_scale_boosting_", None)
        if noise_scale_boosting is not None:
            j["noise_scale_boosting"] = jsonify_item(noise_scale_boosting)
    if 2 <= level:
        bag_weights = getattr(ebm, "bag_weights_", None)
        if bag_weights is not None:
            j["bag_weights"] = jsonify_lists(bag_weights.tolist())
    if 3 <= level:
        breakpoint_iteration = getattr(ebm, "breakpoint_iteration_", None)
        if breakpoint_iteration is not None:
            j["breakpoint_iteration"] = breakpoint_iteration.tolist()

    if 3 <= level:
        j["implementation"] = "python"
        params = {}

        # TODO: we need to clean up and validate our input parameters before putting them into JSON
        # if we were pass a numpy array instead of a list or a numpy type these would fail
        # for now we can just require that anything numpy as input is illegal

        if hasattr(ebm, "feature_names"):
            params["feature_names"] = ebm.feature_names

        if hasattr(ebm, "feature_types"):
            params["feature_types"] = ebm.feature_types

        if hasattr(ebm, "max_bins"):
            params["max_bins"] = ebm.max_bins

        if hasattr(ebm, "max_interaction_bins"):
            params["max_interaction_bins"] = ebm.max_interaction_bins

        if hasattr(ebm, "interactions"):
            params["interactions"] = ebm.interactions

        if hasattr(ebm, "exclude"):
            params["exclude"] = ebm.exclude

        if hasattr(ebm, "validation_size"):
            params["validation_size"] = ebm.validation_size

        if hasattr(ebm, "outer_bags"):
            params["outer_bags"] = ebm.outer_bags

        if hasattr(ebm, "inner_bags"):
            params["inner_bags"] = ebm.inner_bags

        if hasattr(ebm, "learning_rate"):
            params["learning_rate"] = ebm.learning_rate

        if hasattr(ebm, "greediness"):
            params["greediness"] = ebm.greediness

        if hasattr(ebm, "smoothing_rounds"):
            params["smoothing_rounds"] = ebm.smoothing_rounds

        if hasattr(ebm, "max_rounds"):
            params["max_rounds"] = ebm.max_rounds

        if hasattr(ebm, "early_stopping_rounds"):
            params["early_stopping_rounds"] = ebm.early_stopping_rounds

        if hasattr(ebm, "early_stopping_tolerance"):
            params["early_stopping_tolerance"] = ebm.early_stopping_tolerance

        if hasattr(ebm, "min_samples_leaf"):
            params["min_samples_leaf"] = ebm.min_samples_leaf

        if hasattr(ebm, "max_leaves"):
            params["max_leaves"] = ebm.max_leaves

        if hasattr(ebm, "objective"):
            params["objective"] = ebm.objective

        if hasattr(ebm, "n_jobs"):
            params["n_jobs"] = ebm.n_jobs

        if hasattr(ebm, "random_state"):
            params["random_state"] = ebm.random_state

        if hasattr(ebm, "epsilon"):
            params["epsilon"] = ebm.epsilon

        if hasattr(ebm, "delta"):
            params["delta"] = ebm.delta

        if hasattr(ebm, "composition"):
            params["composition"] = ebm.composition

        if hasattr(ebm, "bin_budget_frac"):
            params["bin_budget_frac"] = ebm.bin_budget_frac

        if hasattr(ebm, "privacy_bounds"):
            params["privacy_bounds"] = ebm.privacy_bounds

        if hasattr(ebm, "privacy_target_min"):
            params["privacy_target_min"] = ebm.privacy_target_min

        if hasattr(ebm, "privacy_target_max"):
            params["privacy_target_max"] = ebm.privacy_target_max

        j["implementation_params"] = params

    unique_val_counts = getattr(ebm, "unique_val_counts_", None)
    feature_bounds = getattr(ebm, "feature_bounds_", None)
    histogram_weights = getattr(ebm, "histogram_weights_", None)

    features = []
    for i in range(len(ebm.bins_)):
        feature = {}

        feature["name"] = ebm.feature_names_in_[i]
        feature["type"] = ebm.feature_types_in_[i]

        if 1 <= level:
            if unique_val_counts is not None:
                feature["num_unique_vals"] = int(unique_val_counts[i])

        if isinstance(ebm.bins_[i][0], dict):
            categories = []
            for bins in ebm.bins_[i]:
                leveled_categories = []
                feature_categories = list(map(tuple, map(reversed, bins.items())))
                feature_categories.sort()  # groupby requires sorted data
                for _, category_iter in groupby(feature_categories, lambda x: x[0]):
                    category_group = [category for _, category in category_iter]
                    if len(category_group) == 1:
                        leveled_categories.append(category_group[0])
                    else:
                        leveled_categories.append(category_group)
                categories.append(leveled_categories)
            feature["categories"] = categories
        else:
            cuts = []
            for bins in ebm.bins_[i]:
                cuts.append(bins.tolist())
            feature["cuts"] = cuts
            if 1 <= level:
                if feature_bounds is not None:
                    feature_min = feature_bounds[i, 0]
                    if not isnan(feature_min):
                        feature["min"] = jsonify_item(feature_min)
                    feature_max = feature_bounds[i, 1]
                    if not isnan(feature_max):
                        feature["max"] = jsonify_item(feature_max)
                if histogram_weights is not None:
                    feature_histogram_weights = histogram_weights[i]
                    if feature_histogram_weights is not None:
                        feature[
                            "histogram_weights"
                        ] = feature_histogram_weights.tolist()

        features.append(feature)
    j["features"] = features

    standard_deviations_all = getattr(ebm, "standard_deviations_", None)
    bagged_scores_all = getattr(ebm, "bagged_scores_", None)

    terms = []
    for term_idx in range(len(ebm.term_features_)):
        term = {}
        # we already used "features", so use "term_features" to avoid confusion
        term["term_features"] = [
            ebm.feature_names_in_[feature_idx]
            for feature_idx in ebm.term_features_[term_idx]
        ]
        term["scores"] = jsonify_lists(ebm.term_scores_[term_idx].tolist())
        if 1 <= level:
            if standard_deviations_all is not None:
                standard_deviations = standard_deviations_all[term_idx]
                if standard_deviations is not None:
                    term["standard_deviations"] = jsonify_lists(
                        standard_deviations.tolist()
                    )
        if 2 <= level:
            if bagged_scores_all is not None:
                bagged_scores = bagged_scores_all[term_idx]
                if bagged_scores is not None:
                    term["bagged_scores"] = jsonify_lists(bagged_scores.tolist())
        if 1 <= level:
            term["bin_weights"] = jsonify_lists(ebm.bin_weights_[term_idx].tolist())

        terms.append(term)
    j["terms"] = terms

    return j


def to_jsonable(ebm, detail="all"):
    """Converts the model to a JSONable representation.

    Args:
        detail: 'minimal', 'interpretable', 'mergeable', 'all'

    Returns:
        JSONable object
    """

    warn(
        "JSON formats are in beta. The JSON format may change in a future version without compatibility between releases."
    )

    # NOTES: When recording edits to the EBM within a single file, we should:
    #        1) Have the final EBM section first.  This allows people to diff two models and the diffs for
    #           the current model (the most important information) will be at the top. If people are comparing a
    #           non-edited model to an edited model then they will be comparing the non-edited model to the
    #           current model, which is what we want. When people open the file they'll see the current model,
    #           which will confuse people less.
    #        2) Have the initial model LAST.  This will help separate the final and inital model spacially.
    #           Someone examining the models won't accidentlly stray as easily from the current model into the
    #           initial model while examining them. This also helps prevent the diffing tool from getting
    #           confused and diffing parts of the final model with parts of the initial model if there are
    #           substantial changes. Two final models that have the same initial model should then have a large
    #           unmodified section at the bottom, which the diffing tool should easily identify and keep
    #           together as one block since diffing tools look for longest unmodified sections of text
    #        3) The edits in the MIDDLE, starting from the LAST edit to the FIRST edit chronologically.
    #           If two models are derrived from the same initial model, then they will share a common initial
    #           block of text at the bottom of the file. If the two models share a few edits, then the shared edits
    #           will be at the bottom and will therefore form a larger block of unmodified text along with the
    #           initial model.  Since diff tools look for longest unmodified blocks, this will gobble up the initial
    #           model and the initial edits together first, and thus leave the final models for comparison with
    #           eachother. All edits should have a bi-directional nature so someone could start
    #           from the final model and work backwards to the initial model, or vice versa. The overall file
    #           can then be viewed as a reverse chronological ordering from the final model back to its
    #           original/initial model.
    # - A non-edited EBM file should be saved with just the single JSON for the model and not an initial and
    #   final model.  The only section should be marked with the tag "ebm" so that tools that read in EBMs
    #   Are compatible with both editied and non-edited files.  The tools will always look for the "ebm"
    #   section, which will be in both non-edited EBMs and edited EBMs at the top.
    # - The file would look like this for an edited EBMs:
    #   {
    #     "version": "1.0"
    #     "ebm": { FINAL_EBM_JSON }
    #     "edits": [
    #       { NEWEST_EDIT_JSON },
    #       { MID_EDITs_JSON },
    #       { OLDEST_EDIT_JSON }
    #     ]
    #     "initial_ebm": { INITIAL_EBM_JSON }
    #   }
    # - The file would look like this for an unedited EBMs:
    #   {
    #     "version": "1.0"
    #     "ebm": { EBM_JSON }
    #   }
    # - In python, we could contain these in attributes called "initial_ebm" which would contain a fully formed ebm
    #   and "edits", which would contain a list of the edits.  These fields wouldn't be present in a scikit-learn
    #   generated EBM, but would appear if the user edited the EBM, or if they loaded one that had edits.

    inner = _to_json_inner(ebm, detail)

    outer = {}
    outer["version"] = "1.0"
    outer["ebm"] = inner

    return outer
