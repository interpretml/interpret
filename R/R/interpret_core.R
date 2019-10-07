# Copyright (c) 2018 Microsoft Corporation
# Licensed under the MIT license.
# Author: Paul Koch <code@koch.ninja>

# useful links.. check these periodically for issues
# https://CRAN.R-project.org/package=interpret
# https://cran.r-project.org/web/checks/check_results_interpret.html
# https://cran.r-project.org/web/checks/check_summary_by_package.html#summary_by_package
# https://cran.r-project.org/web/checks/check_flavors.html

# if archived, it will appear in:
# https://cran-archive.r-project.org/web/checks/2019-10-07_check_results_interpret.html
# https://cran.r-project.org/src/contrib/Archive/

# we can test our package against many different systesm with:
# https://builder.r-hub.io

# S3 data structures

ebm_feature <- function(count_bins, has_missing = FALSE, feature_type = "ordinal") {
   count_bins <- as.double(count_bins)
   stopifnot(is.logical(has_missing))
   feature_type <- match.arg(feature_type, c("ordinal", "nominal"))
   ret <- structure(list(count_bins = count_bins, has_missing = has_missing, feature_type = feature_type), class = "ebm_feature")
   return(ret)
}

ebm_feature_combination <- function(count_features_in_combination = 1) {
   count_features_in_combination <- as.double(count_features_in_combination)
   ret <- structure(list(count_features_in_combination = count_features_in_combination), class = "ebm_feature_combination")
   return(ret)
}

# Training functions

initialize_training_regression <- function(random_seed, features, feature_combinations, feature_combination_indexes, training_targets, training_binned_data, training_predictor_scores, validation_targets, validation_binned_data, validation_predictor_scores, count_inner_bags) {
   random_seed <- as.integer(random_seed)
   features <- as.list(features)
   feature_combinations <- as.list(feature_combinations)
   feature_combination_indexes <- as.double(feature_combination_indexes)
   training_targets <- as.double(training_targets)
   training_binned_data <- as.double(training_binned_data)
   if(!is.null(training_predictor_scores)) {
      training_predictor_scores <- as.double(training_predictor_scores)
   }
   validation_targets <- as.double(validation_targets)
   validation_binned_data <- as.double(validation_binned_data)
   if(!is.null(validation_predictor_scores)) {
      validation_predictor_scores <- as.double(validation_predictor_scores)
   }
   count_inner_bags <- as.integer(count_inner_bags)

   ebm_training <- .Call("InitializeTrainingRegression_R", random_seed, features, feature_combinations, feature_combination_indexes, training_targets, training_binned_data, training_predictor_scores, validation_targets, validation_binned_data, validation_predictor_scores, count_inner_bags)
   if(is.null(ebm_training)) {
      stop("error in InitializeTrainingRegression_R")
   }
   return(ebm_training)
}

initialize_training_classification <- function(random_seed, features, feature_combinations, feature_combination_indexes, count_target_classes, training_targets, training_binned_data, training_predictor_scores, validation_targets, validation_binned_data, validation_predictor_scores, count_inner_bags) {
   random_seed <- as.integer(random_seed)
   features <- as.list(features)
   feature_combinations <- as.list(feature_combinations)
   feature_combination_indexes <- as.double(feature_combination_indexes)
   count_target_classes <- as.double(count_target_classes)
   training_targets <- as.double(training_targets)
   training_binned_data <- as.double(training_binned_data)
   if(!is.null(training_predictor_scores)) {
      training_predictor_scores <- as.double(training_predictor_scores)
   }
   validation_targets <- as.double(validation_targets)
   validation_binned_data <- as.double(validation_binned_data)
   if(!is.null(validation_predictor_scores)) {
      validation_predictor_scores <- as.double(validation_predictor_scores)
   }
   count_inner_bags <- as.integer(count_inner_bags)

   ebm_training <- .Call("InitializeTrainingClassification_R", random_seed, features, feature_combinations, feature_combination_indexes, count_target_classes, training_targets, training_binned_data, training_predictor_scores, validation_targets, validation_binned_data, validation_predictor_scores, count_inner_bags)
   if(is.null(ebm_training)) {
      stop("error in InitializeTrainingClassification_R")
   }
   return(ebm_training)
}

training_step <- function(ebm_training, index_feature_combination, learning_rate, count_tree_splits_max, count_instances_required_for_parent_split_min, training_weights, validation_weights) {
   stopifnot(class(ebm_training) == "externalptr")
   index_feature_combination <- as.double(index_feature_combination)
   learning_rate <- as.double(learning_rate)
   count_tree_splits_max <- as.double(count_tree_splits_max)
   count_instances_required_for_parent_split_min <- as.double(count_instances_required_for_parent_split_min)
   if(!is.null(training_weights)) {
      training_weights <- as.double(training_weights)
   }
   if(!is.null(validation_weights)) {
      validation_weights <- as.double(validation_weights)
   }

   validation_metric <- .Call("TrainingStep_R", ebm_training, index_feature_combination, learning_rate, count_tree_splits_max, count_instances_required_for_parent_split_min, training_weights, validation_weights)
   if(is.null(validation_metric)) {
      stop("error in TrainingStep_R")
   }
   return(validation_metric)
}

get_current_model_feature_combination <- function(ebm_training, index_feature_combination) {
   stopifnot(class(ebm_training) == "externalptr")
   index_feature_combination <- as.double(index_feature_combination)

   model_feature_combination_tensor <- .Call("GetCurrentModelFeatureCombination_R", ebm_training, index_feature_combination)
   if(is.null(model_feature_combination_tensor)) {
      stop("error in GetCurrentModelFeatureCombination_R")
   }
   return(model_feature_combination_tensor)
}

get_best_model_feature_combination <- function(ebm_training, index_feature_combination) {
   stopifnot(class(ebm_training) == "externalptr")
   index_feature_combination <- as.double(index_feature_combination)

   model_feature_combination_tensor <- .Call("GetBestModelFeatureCombination_R", ebm_training, index_feature_combination)
   if(is.null(model_feature_combination_tensor)) {
      stop("error in GetBestModelFeatureCombination_R")
   }
   return(model_feature_combination_tensor)
}


# Interaction detection functions

initialize_interaction_regression <- function(features, targets, binned_data, predictor_scores) {
   features <- as.list(features)
   targets <- as.double(targets)
   binned_data <- as.double(binned_data)
   if(!is.null(predictor_scores)) {
      predictor_scores <- as.double(predictor_scores)
   }

   ebm_interaction <- .Call("InitializeInteractionRegression_R", features, targets, binned_data, predictor_scores)
   if(is.null(ebm_interaction)) {
      stop("error in InitializeInteractionRegression_R")
   }
   return(ebm_interaction)
}

initialize_interaction_classification <- function(features, count_target_classes, targets, binned_data, predictor_scores) {
   features <- as.list(features)
   count_target_classes <- as.double(count_target_classes)
   targets <- as.double(targets)
   binned_data <- as.double(binned_data)
   if(!is.null(predictor_scores)) {
      predictor_scores <- as.double(predictor_scores)
   }

   ebm_interaction <- .Call("InitializeInteractionClassification_R", features, count_target_classes, targets, binned_data, predictor_scores)
   if(is.null(ebm_interaction)) {
      stop("error in InitializeInteractionClassification_R")
   }
   return(ebm_interaction)
}

get_interaction_score <- function(ebm_interaction, feature_indexes) {
   stopifnot(class(ebm_interaction) == "externalptr")
   feature_indexes <- as.double(feature_indexes)

   interaction_score <- .Call("GetInteractionScore_R", ebm_interaction, feature_indexes)
   if(is.null(interaction_score)) {
      stop("error in GetInteractionScore_R")
   }
   return(interaction_score)
}

