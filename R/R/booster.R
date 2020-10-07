# Copyright (c) 2018 Microsoft Corporation
# Licensed under the MIT license.
# Author: Paul Koch <code@koch.ninja>

ebm_feature_group_c <- function(n_features = 1) {
   n_features <- as.double(n_features)
   ret <- structure(list(n_features = n_features), class = "ebm_feature_group_c")
   return(ret)
}

convert_feature_groups_to_c <- function(feature_groups) {
   n_feature_indexes <- 0
   for (feature_group in feature_groups) {
      n_feature_indexes <- n_feature_indexes + length(feature_group$feature_indexes)
   }

   feature_group_indexes <- vector(mode = "list", n_feature_indexes)
   feature_groups_c <- vector(mode = "list", length(feature_groups))
   index_indexes <- 1

   for (index_feature_group in seq_along(feature_groups)) {
      feature_group <- feature_groups[[index_feature_group]]
      feature_indexes_in_group <- feature_group$feature_indexes
      feature_groups_c[[index_feature_group]] <- ebm_feature_group_c(length(feature_indexes_in_group))

      for(feature_index in feature_indexes_in_group) {
         feature_group_indexes[[index_indexes]] <- feature_index - 1
         index_indexes <- index_indexes + 1
      }
   }
   return(list(feature_groups_c = feature_groups_c, feature_group_indexes = feature_group_indexes))
}

initialize_boosting_classification <- function(
   n_classes, 
   features, 
   feature_groups, 
   feature_group_indexes, 
   training_binned_data, 
   training_targets, 
   training_predictor_scores, 
   validation_binned_data, 
   validation_targets, 
   validation_predictor_scores, 
   n_inner_bags, 
   random_seed
) {
   n_classes <- as.double(n_classes)
   features <- as.list(features)
   feature_groups <- as.list(feature_groups)
   feature_group_indexes <- as.double(feature_group_indexes)
   training_binned_data <- as.double(training_binned_data)
   training_targets <- as.double(training_targets)
   if(!is.null(training_predictor_scores)) {
      training_predictor_scores <- as.double(training_predictor_scores)
   }
   validation_binned_data <- as.double(validation_binned_data)
   validation_targets <- as.double(validation_targets)
   if(!is.null(validation_predictor_scores)) {
      validation_predictor_scores <- as.double(validation_predictor_scores)
   }
   n_inner_bags <- as.integer(n_inner_bags)
   random_seed <- as.integer(random_seed)

   ebm_boosting <- .Call(
      InitializeBoostingClassification_R, 
      n_classes, 
      features, 
      feature_groups, 
      feature_group_indexes, 
      training_binned_data, 
      training_targets, 
      training_predictor_scores, 
      validation_binned_data, 
      validation_targets, 
      validation_predictor_scores, 
      n_inner_bags, 
      random_seed
   )
   if(is.null(ebm_boosting)) {
      stop("Out of memory in InitializeBoostingClassification")
   }
   return(ebm_boosting)
}

initialize_boosting_regression <- function(
   features, 
   feature_groups, 
   feature_group_indexes, 
   training_binned_data, 
   training_targets, 
   training_predictor_scores, 
   validation_binned_data, 
   validation_targets, 
   validation_predictor_scores, 
   n_inner_bags, 
   random_seed
) {
   features <- as.list(features)
   feature_groups <- as.list(feature_groups)
   feature_group_indexes <- as.double(feature_group_indexes)
   training_binned_data <- as.double(training_binned_data)
   training_targets <- as.double(training_targets)
   if(!is.null(training_predictor_scores)) {
      training_predictor_scores <- as.double(training_predictor_scores)
   }
   validation_binned_data <- as.double(validation_binned_data)
   validation_targets <- as.double(validation_targets)
   if(!is.null(validation_predictor_scores)) {
      validation_predictor_scores <- as.double(validation_predictor_scores)
   }
   n_inner_bags <- as.integer(n_inner_bags)
   random_seed <- as.integer(random_seed)

   ebm_boosting <- .Call(
      InitializeBoostingRegression_R, 
      features, 
      feature_groups, 
      feature_group_indexes, 
      training_binned_data, 
      training_targets, 
      training_predictor_scores, 
      validation_binned_data, 
      validation_targets, 
      validation_predictor_scores, 
      n_inner_bags, 
      random_seed
   )
   if(is.null(ebm_boosting)) {
      stop("Out of memory in InitializeBoostingRegression")
   }
   return(ebm_boosting)
}

native_ebm_boosting_free <- function(native_ebm_boosting) {
   .Call(FreeBoosting_R, native_ebm_boosting$booster_pointer)
   return(NULL)
}

boosting_step <- function(
   ebm_boosting, 
   index_feature_group, 
   learning_rate, 
   n_tree_splits_max, 
   n_samples_required_for_child_split_min, 
   training_weights, 
   validation_weights
) {
   stopifnot(class(ebm_boosting) == "externalptr")
   index_feature_group <- as.double(index_feature_group)
   learning_rate <- as.double(learning_rate)
   n_tree_splits_max <- as.double(n_tree_splits_max)
   n_samples_required_for_child_split_min <- as.double(n_samples_required_for_child_split_min)
   if(!is.null(training_weights)) {
      training_weights <- as.double(training_weights)
   }
   if(!is.null(validation_weights)) {
      validation_weights <- as.double(validation_weights)
   }

   validation_metric <- .Call(
      BoostingStep_R, 
      ebm_boosting, 
      index_feature_group, 
      learning_rate, 
      n_tree_splits_max, 
      n_samples_required_for_child_split_min, 
      training_weights, 
      validation_weights
   )
   if(is.null(validation_metric)) {
      stop("error in BoostingStep_R")
   }
   return(validation_metric)
}

get_best_model_feature_group <- function(ebm_boosting, index_feature_group) {
   stopifnot(class(ebm_boosting) == "externalptr")
   index_feature_group <- as.double(index_feature_group)

   model_feature_group_tensor <- .Call(GetBestModelFeatureGroup_R, ebm_boosting, index_feature_group)
   if(is.null(model_feature_group_tensor)) {
      stop("error in GetBestModelFeatureGroup_R")
   }
   return(model_feature_group_tensor)
}

get_current_model_feature_group <- function(ebm_boosting, index_feature_group) {
   stopifnot(class(ebm_boosting) == "externalptr")
   index_feature_group <- as.double(index_feature_group)

   model_feature_group_tensor <- .Call(GetCurrentModelFeatureGroup_R, ebm_boosting, index_feature_group)
   if(is.null(model_feature_group_tensor)) {
      stop("error in GetCurrentModelFeatureGroup_R")
   }
   return(model_feature_group_tensor)
}

get_best_model <- function(native_ebm_boosting) {
   model <- lapply(
      seq_along(native_ebm_boosting$feature_groups), 
      function(i) { get_best_model_feature_group(native_ebm_boosting$booster_pointer, i - 1) }
   )
   return(model)
}

get_current_model <- function(native_ebm_boosting) {
   model <- lapply(
      seq_along(native_ebm_boosting$feature_groups), 
      function(i) { get_current_model_feature_group(native_ebm_boosting$booster_pointer, i - 1) }
   )
   return(model)
}

native_ebm_boosting <- function(
   model_type,
   n_classes,
   features,
   feature_groups,
   X_train,
   y_train,
   scores_train,
   X_val,
   y_val,
   scores_val,
   n_inner_bags,
   random_state
) {
   c_structs <- convert_feature_groups_to_c(feature_groups)

   if(model_type == "classification") {
      booster_pointer <- initialize_boosting_classification(
         n_classes, 
         features, 
         c_structs$feature_groups_c, 
         c_structs$feature_group_indexes, 
         X_train, 
         y_train, 
         scores_train, 
         X_val, 
         y_val, 
         scores_val, 
         n_inner_bags, 
         random_state
      )
   } else if(model_type == "regression") {
      booster_pointer <- initialize_boosting_regression(
         features, 
         c_structs$feature_groups_c, 
         c_structs$feature_group_indexes, 
         X_train, 
         y_train, 
         scores_train, 
         X_val, 
         y_val, 
         scores_val, 
         n_inner_bags, 
         random_state
      )
   } else {
      stop("Unrecognized model_type")
   }

   self <- structure(list(
      model_type = model_type, 
      n_classes = n_classes, 
      feature_groups = feature_groups, 
      booster_pointer = booster_pointer
      ), class = "native_ebm_boosting")
   return(self)
}

cyclic_gradient_boost <- function(
   model_type,
   n_classes,
   features,
   feature_groups,
   X_train,
   y_train,
   scores_train,
   X_val,
   y_val,
   scores_val,
   n_inner_bags,
   random_state,
   learning_rate,
   n_tree_splits_max, 
   n_samples_required_for_child_split_min, 
   data_n_episodes,
   early_stopping_run_length
#   name
) {
   min_metric <- Inf
   episode_index <- 0

   ebm_booster <- native_ebm_boosting(
      model_type,
      n_classes,
      features,
      feature_groups,
      X_train,
      y_train,
      scores_train,
      X_val,
      y_val,
      scores_val,
      n_inner_bags,
      random_state
   )
   result_list <- tryCatch({
      no_change_run_length <- 0
      bp_metric <- Inf

      for(episode_index in 1:data_n_episodes) {
         for(feature_group_index in seq_along(feature_groups)) {
            validation_metric <- boosting_step(
               ebm_booster$booster_pointer, 
               feature_group_index - 1, 
               learning_rate, 
               n_tree_splits_max, 
               n_samples_required_for_child_split_min, 
               NULL,
               NULL
            )
            if(validation_metric < min_metric) {
               min_metric <- validation_metric
            }
         }

         if(no_change_run_length == 0) {
            bp_metric <- min_metric
         }
         if(min_metric < bp_metric) {
            no_change_run_length <- 0
         } else {
            no_change_run_length <- no_change_run_length + 1
         }

         if(early_stopping_run_length >= 0 && no_change_run_length >= early_stopping_run_length) {
            break
         }
      }

      model_update <- get_best_model(ebm_booster)

      return(list(model_update = model_update, min_metric = min_metric, episode_index = episode_index))
   }, finally = {
      native_ebm_boosting_free(ebm_booster)
   })
   return(result_list)
}
