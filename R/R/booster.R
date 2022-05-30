# Copyright (c) 2018 Microsoft Corporation
# Licensed under the MIT license.
# Author: Paul Koch <code@koch.ninja>

convert_feature_groups_to_c <- function(feature_groups) {
   n_feature_indexes <- 0
   for (feature_group in feature_groups) {
      n_feature_indexes <- n_feature_indexes + length(feature_group$feature_indexes)
   }

   feature_groups_feature_count <- vector("numeric", length(feature_groups))
   feature_groups_feature_indexes <- vector("numeric", n_feature_indexes)
   index_indexes <- 1

   for (index_feature_group in seq_along(feature_groups)) {
      feature_group <- feature_groups[[index_feature_group]]
      feature_indexes_in_group <- feature_group$feature_indexes
      feature_groups_feature_count[[index_feature_group]] <- length(feature_indexes_in_group)

      for(feature_index in feature_indexes_in_group) {
         feature_groups_feature_indexes[[index_indexes]] <- feature_index - 1
         index_indexes <- index_indexes + 1
      }
   }
   return(list(feature_groups_feature_count = feature_groups_feature_count, feature_groups_feature_indexes = feature_groups_feature_indexes))
}

create_classification_booster <- function(
   random_seed,
   count_target_classes, 
   features_categorical,
   features_bin_count,
   feature_groups_feature_count, 
   feature_groups_feature_indexes, 
   training_binned_data, 
   training_targets, 
   training_weights, 
   training_predictor_scores, 
   validation_binned_data, 
   validation_targets, 
   validation_weights, 
   validation_predictor_scores, 
   count_inner_bags
) {
   random_seed <- as.integer(random_seed)
   count_target_classes <- as.double(count_target_classes)
   features_categorical <- as.logical(features_categorical)
   features_bin_count <- as.double(features_bin_count)
   feature_groups_feature_count <- as.double(feature_groups_feature_count)
   feature_groups_feature_indexes <- as.double(feature_groups_feature_indexes)
   training_binned_data <- as.double(training_binned_data)
   training_targets <- as.double(training_targets)
   if(!is.null(training_weights)) {
      training_weights <- as.double(training_weights)
   }
   if(!is.null(training_predictor_scores)) {
      training_predictor_scores <- as.double(training_predictor_scores)
   }
   validation_binned_data <- as.double(validation_binned_data)
   validation_targets <- as.double(validation_targets)
   if(!is.null(validation_weights)) {
      validation_weights <- as.double(validation_weights)
   }
   if(!is.null(validation_predictor_scores)) {
      validation_predictor_scores <- as.double(validation_predictor_scores)
   }
   count_inner_bags <- as.integer(count_inner_bags)

   booster_handle <- .Call(
      CreateClassificationBooster_R, 
      random_seed,
      count_target_classes, 
      features_categorical,
      features_bin_count,
      feature_groups_feature_count, 
      feature_groups_feature_indexes, 
      training_binned_data, 
      training_targets, 
      training_weights, 
      training_predictor_scores, 
      validation_binned_data, 
      validation_targets, 
      validation_weights, 
      validation_predictor_scores, 
      count_inner_bags
   )
   if(is.null(booster_handle)) {
      stop("Error in CreateClassificationBooster")
   }
   return(booster_handle)
}

create_regression_booster <- function(
   random_seed,
   features_categorical,
   features_bin_count,
   feature_groups_feature_count, 
   feature_groups_feature_indexes, 
   training_binned_data, 
   training_targets, 
   training_weights, 
   training_predictor_scores, 
   validation_binned_data, 
   validation_targets, 
   validation_weights, 
   validation_predictor_scores, 
   count_inner_bags
) {
   random_seed <- as.integer(random_seed)
   features_categorical <- as.logical(features_categorical)
   features_bin_count <- as.double(features_bin_count)
   feature_groups_feature_count <- as.double(feature_groups_feature_count)
   feature_groups_feature_indexes <- as.double(feature_groups_feature_indexes)
   training_binned_data <- as.double(training_binned_data)
   training_targets <- as.double(training_targets)
   if(!is.null(training_weights)) {
      training_weights <- as.double(training_weights)
   }
   if(!is.null(training_predictor_scores)) {
      training_predictor_scores <- as.double(training_predictor_scores)
   }
   validation_binned_data <- as.double(validation_binned_data)
   validation_targets <- as.double(validation_targets)
   if(!is.null(validation_weights)) {
      validation_weights <- as.double(validation_weights)
   }
   if(!is.null(validation_predictor_scores)) {
      validation_predictor_scores <- as.double(validation_predictor_scores)
   }
   count_inner_bags <- as.integer(count_inner_bags)

   booster_handle <- .Call(
      CreateRegressionBooster_R, 
      random_seed,
      features_categorical,
      features_bin_count,
      feature_groups_feature_count, 
      feature_groups_feature_indexes, 
      training_binned_data, 
      training_targets, 
      training_weights, 
      training_predictor_scores, 
      validation_binned_data, 
      validation_targets, 
      validation_weights, 
      validation_predictor_scores, 
      count_inner_bags
   )
   if(is.null(booster_handle)) {
      stop("Error in CreateRegressionBooster")
   }
   return(booster_handle)
}

generate_model_update <- function(
   booster_handle, 
   index_feature_group, 
   learning_rate, 
   count_samples_required_for_child_split_min, 
   max_leaves
) {
   stopifnot(class(booster_handle) == "externalptr")
   index_feature_group <- as.double(index_feature_group)
   learning_rate <- as.double(learning_rate)
   count_samples_required_for_child_split_min <- as.double(count_samples_required_for_child_split_min)
   max_leaves <- as.double(max_leaves)

   avg_gain <- .Call(
      GenerateModelUpdate_R, 
      booster_handle, 
      index_feature_group, 
      learning_rate, 
      count_samples_required_for_child_split_min, 
      max_leaves
   )
   if(is.null(avg_gain)) {
      stop("error in GenerateModelUpdate_R")
   }
   return(avg_gain)
}

apply_model_update <- function(
   booster_handle
) {
   stopifnot(class(booster_handle) == "externalptr")

   validation_metric <- .Call(
      ApplyModelUpdate_R, 
      booster_handle
   )
   if(is.null(validation_metric)) {
      stop("error in BoostingStep_R")
   }
   return(validation_metric)
}

get_best_model_feature_group <- function(booster_handle, index_feature_group) {
   stopifnot(class(booster_handle) == "externalptr")
   index_feature_group <- as.double(index_feature_group)

   model_feature_group_tensor <- .Call(GetBestModelFeatureGroup_R, booster_handle, index_feature_group)
   if(is.null(model_feature_group_tensor)) {
      stop("error in GetBestModelFeatureGroup_R")
   }
   return(model_feature_group_tensor)
}

get_current_model_feature_group <- function(booster_handle, index_feature_group) {
   stopifnot(class(booster_handle) == "externalptr")
   index_feature_group <- as.double(index_feature_group)

   model_feature_group_tensor <- .Call(GetCurrentModelFeatureGroup_R, booster_handle, index_feature_group)
   if(is.null(model_feature_group_tensor)) {
      stop("error in GetCurrentModelFeatureGroup_R")
   }
   return(model_feature_group_tensor)
}

free_boosting <- function(booster_handle) {
   .Call(FreeBooster_R, booster_handle)
   return(NULL)
}

get_best_model <- function(booster) {
   model <- lapply(
      seq_along(booster$feature_groups), 
      function(i) { get_best_model_feature_group(booster$booster_handle, i - 1) }
   )
   return(model)
}

get_current_model <- function(booster) {
   model <- lapply(
      seq_along(booster$feature_groups), 
      function(i) { get_current_model_feature_group(booster$booster_handle, i - 1) }
   )
   return(model)
}

booster <- function(
   model_type,
   n_classes,
   features_categorical,
   features_bin_count,
   feature_groups,
   X_train,
   y_train,
   weights_train, 
   scores_train,
   X_val,
   y_val,
   weights_val, 
   scores_val,
   inner_bags,
   random_state
) {
   c_structs <- convert_feature_groups_to_c(feature_groups)

   if(model_type == "classification") {
      booster_handle <- create_classification_booster(
         random_state,
         n_classes, 
         features_categorical,
         features_bin_count,
         c_structs$feature_groups_feature_count, 
         c_structs$feature_groups_feature_indexes, 
         X_train, 
         y_train, 
         weights_train, 
         scores_train, 
         X_val, 
         y_val, 
         weights_val, 
         scores_val, 
         inner_bags
      )
   } else if(model_type == "regression") {
      booster_handle <- create_regression_booster(
         random_state,
         features_categorical,
         features_bin_count,
         c_structs$feature_groups_feature_count, 
         c_structs$feature_groups_feature_indexes, 
         X_train, 
         y_train, 
         weights_train, 
         scores_train, 
         X_val, 
         y_val, 
         weights_val, 
         scores_val, 
         inner_bags
      )
   } else {
      stop("Unrecognized model_type")
   }

   self <- structure(list(
      model_type = model_type, 
      n_classes = n_classes, 
      feature_groups = feature_groups, 
      booster_handle = booster_handle
   ), class = "booster")
   return(self)
}

cyclic_gradient_boost <- function(
   model_type,
   n_classes,
   features_categorical,
   features_bin_count,
   feature_groups,
   X_train,
   y_train,
   weights_train, 
   scores_train,
   X_val,
   y_val,
   weights_val, 
   scores_val,
   inner_bags,
   learning_rate,
   min_samples_leaf, 
   max_leaves, 
   early_stopping_rounds, 
   early_stopping_tolerance,
   max_rounds, 
   random_state
) {
   min_metric <- Inf
   episode_index <- 0

   ebm_booster <- booster(
      model_type,
      n_classes,
      features_categorical,
      features_bin_count,
      feature_groups,
      X_train,
      y_train,
      weights_train, 
      scores_train,
      X_val,
      y_val,
      weights_val, 
      scores_val,
      inner_bags,
      random_state
   )
   result_list <- tryCatch({
      no_change_run_length <- 0
      bp_metric <- Inf

      for(episode_index in 1:max_rounds) {
         for(feature_group_index in seq_along(feature_groups)) {
            avg_gain <- generate_model_update(
               ebm_booster$booster_handle, 
               feature_group_index - 1, 
               learning_rate, 
               min_samples_leaf, 
               max_leaves
            )

            validation_metric <- apply_model_update(ebm_booster$booster_handle)

            if(validation_metric < min_metric) {
               min_metric <- validation_metric
            }
         }

         if(no_change_run_length == 0) {
            bp_metric <- min_metric
         }
         if(min_metric + early_stopping_tolerance < bp_metric) {
            no_change_run_length <- 0
         } else {
            no_change_run_length <- no_change_run_length + 1
         }

         if(early_stopping_rounds >= 0 && no_change_run_length >= early_stopping_rounds) {
            break
         }
      }

      model_update <- get_best_model(ebm_booster)

      return(list(model_update = model_update, min_metric = min_metric, episode_index = episode_index))
   }, finally = {
      free_boosting(ebm_booster$booster_handle)
   })
   return(result_list)
}
