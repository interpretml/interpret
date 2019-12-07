# Copyright (c) 2018 Microsoft Corporation
# Licensed under the MIT license.
# Author: Paul Koch <code@koch.ninja>

# useful links.. check these periodically for issues
# https://CRAN.R-project.org/package=interpret
# https://cran.r-project.org/web/checks/check_results_interpret.html
# https://cran.r-project.org/web/checks/check_summary_by_package.html#summary_by_package
# https://cran.r-project.org/web/checks/check_flavors.html

# incoming status:
# https://cransays.itsalocke.com/articles/dashboard.html
# ftp://cran.r-project.org/incoming/

# if archived, it will appear in:
# https://cran-archive.r-project.org/web/checks/2019-10-07_check_results_interpret.html
# https://cran.r-project.org/src/contrib/Archive/

# we can test our package against many different systems with:
# https://builder.r-hub.io

# S3 data structures

ebm_feature <- function(n_bins, has_missing = FALSE, feature_type = "ordinal") {
   n_bins <- as.double(n_bins)
   stopifnot(is.logical(has_missing))
   feature_type <- match.arg(feature_type, c("ordinal", "nominal"))
   ret <- structure(list(n_bins = n_bins, has_missing = has_missing, feature_type = feature_type), class = "ebm_feature")
   return(ret)
}

ebm_feature_combination_c <- function(n_features = 1) {
   n_features <- as.double(n_features)
   ret <- structure(list(n_features = n_features), class = "ebm_feature_combination_c")
   return(ret)
}

ebm_feature_combination <- function(feature_indexes) {
   feature_indexes <- as.double(feature_indexes)
   ret <- structure(list(feature_indexes = feature_indexes), class = "ebm_feature_combination")
   return(ret)
}

create_main_feature_combinations <- function(features) {
   feature_combinations <- lapply(seq_along(features), function(i) { ebm_feature_combination(i) })
   return(feature_combinations)
}

convert_feature_combinations_to_c <- function(feature_combinations) {
   n_feature_indexes <- 0
   for (feature_combination in feature_combinations) {
      n_feature_indexes <- n_feature_indexes + length(feature_combination$feature_indexes)
   }

   feature_combination_indexes <- vector(mode = "list", n_feature_indexes)
   feature_combinations_c <- vector(mode = "list", length(feature_combinations))
   index_indexes <- 1

   for (index_feature_combination in seq_along(feature_combinations)) {
      feature_combination <- feature_combinations[[index_feature_combination]]
      feature_indexes_in_combination <- feature_combination$feature_indexes
      feature_combinations_c[[index_feature_combination]] <- ebm_feature_combination_c(length(feature_indexes_in_combination))

      for(feature_index in feature_indexes_in_combination) {
         feature_combination_indexes[[index_indexes]] <- feature_index - 1
         index_indexes <- index_indexes + 1
      }
   }
   return(list(feature_combinations_c = feature_combinations_c, feature_combination_indexes = feature_combination_indexes))
}






# Boosting functions

initialize_boosting_classification <- function(n_classes, features, feature_combinations, feature_combination_indexes, training_binned_data, training_targets, training_predictor_scores, validation_binned_data, validation_targets, validation_predictor_scores, n_inner_bags, random_seed) {
   n_classes <- as.double(n_classes)
   features <- as.list(features)
   feature_combinations <- as.list(feature_combinations)
   feature_combination_indexes <- as.double(feature_combination_indexes)
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

   ebm_boosting <- .Call(InitializeBoostingClassification_R, n_classes, features, feature_combinations, feature_combination_indexes, training_binned_data, training_targets, training_predictor_scores, validation_binned_data, validation_targets, validation_predictor_scores, n_inner_bags, random_seed)
   if(is.null(ebm_boosting)) {
      stop("Out of memory in InitializeBoostingClassification")
   }
   return(ebm_boosting)
}

initialize_boosting_regression <- function(features, feature_combinations, feature_combination_indexes, training_binned_data, training_targets, training_predictor_scores, validation_binned_data, validation_targets, validation_predictor_scores, n_inner_bags, random_seed) {
   features <- as.list(features)
   feature_combinations <- as.list(feature_combinations)
   feature_combination_indexes <- as.double(feature_combination_indexes)
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

   ebm_boosting <- .Call(InitializeBoostingRegression_R, features, feature_combinations, feature_combination_indexes, training_binned_data, training_targets, training_predictor_scores, validation_binned_data, validation_targets, validation_predictor_scores, n_inner_bags, random_seed)
   if(is.null(ebm_boosting)) {
      stop("Out of memory in InitializeBoostingRegression")
   }
   return(ebm_boosting)
}

boosting_step <- function(ebm_boosting, index_feature_combination, learning_rate, n_tree_splits_max, n_instances_required_for_parent_split_min, training_weights, validation_weights) {
   stopifnot(class(ebm_boosting) == "externalptr")
   index_feature_combination <- as.double(index_feature_combination)
   learning_rate <- as.double(learning_rate)
   n_tree_splits_max <- as.double(n_tree_splits_max)
   n_instances_required_for_parent_split_min <- as.double(n_instances_required_for_parent_split_min)
   if(!is.null(training_weights)) {
      training_weights <- as.double(training_weights)
   }
   if(!is.null(validation_weights)) {
      validation_weights <- as.double(validation_weights)
   }

   validation_metric <- .Call(BoostingStep_R, ebm_boosting, index_feature_combination, learning_rate, n_tree_splits_max, n_instances_required_for_parent_split_min, training_weights, validation_weights)
   if(is.null(validation_metric)) {
      stop("error in BoostingStep_R")
   }
   return(validation_metric)
}

get_best_model_feature_combination <- function(ebm_boosting, index_feature_combination) {
   stopifnot(class(ebm_boosting) == "externalptr")
   index_feature_combination <- as.double(index_feature_combination)

   model_feature_combination_tensor <- .Call(GetBestModelFeatureCombination_R, ebm_boosting, index_feature_combination)
   if(is.null(model_feature_combination_tensor)) {
      stop("error in GetBestModelFeatureCombination_R")
   }
   return(model_feature_combination_tensor)
}

get_current_model_feature_combination <- function(ebm_boosting, index_feature_combination) {
   stopifnot(class(ebm_boosting) == "externalptr")
   index_feature_combination <- as.double(index_feature_combination)

   model_feature_combination_tensor <- .Call(GetCurrentModelFeatureCombination_R, ebm_boosting, index_feature_combination)
   if(is.null(model_feature_combination_tensor)) {
      stop("error in GetCurrentModelFeatureCombination_R")
   }
   return(model_feature_combination_tensor)
}




native_ebm_boosting <- function(
   model_type,
   n_classes,
   features,
   feature_combinations,
   X_train,
   y_train,
   scores_train,
   X_val,
   y_val,
   scores_val,
   n_inner_bags,
   random_state
) {
   c_structs <- convert_feature_combinations_to_c(feature_combinations)

   if(model_type == "classification") {
      booster_pointer <- initialize_boosting_classification(
         n_classes, 
         features, 
         c_structs$feature_combinations_c, 
         c_structs$feature_combination_indexes, 
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
         c_structs$feature_combinations_c, 
         c_structs$feature_combination_indexes, 
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

   self <- structure(list(model_type = model_type, n_classes = n_classes, feature_combinations = feature_combinations, booster_pointer = booster_pointer), class = "native_ebm_boosting")
   return(self)
}

native_ebm_boosting_free <- function(native_ebm_boosting) {
   .Call(FreeBoosting_R, native_ebm_boosting$booster_pointer)
   return(NULL)
}

get_best_model <- function(native_ebm_boosting) {
   model <- lapply(seq_along(native_ebm_boosting$feature_combinations), function(i) { get_best_model_feature_combination(native_ebm_boosting$booster_pointer, i - 1) })
   return(model)
}

get_current_model <- function(native_ebm_boosting) {
   model <- lapply(seq_along(native_ebm_boosting$feature_combinations), function(i) { get_current_model_feature_combination(native_ebm_boosting$booster_pointer, i - 1) })
   return(model)
}



cyclic_gradient_boost <- function(
   model_type,
   n_classes,
   features,
   feature_combinations,
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
   n_instances_required_for_parent_split_min, 
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
      feature_combinations,
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
         for(feature_combination_index in seq_along(feature_combinations)) {
            validation_metric <- boosting_step(
               ebm_booster$booster_pointer, 
               feature_combination_index - 1, 
               learning_rate, 
               n_tree_splits_max, 
               n_instances_required_for_parent_split_min, 
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













# Interaction detection functions

initialize_interaction_classification <- function(n_classes, features, binned_data, targets, predictor_scores) {
   n_classes <- as.double(n_classes)
   features <- as.list(features)
   binned_data <- as.double(binned_data)
   targets <- as.double(targets)
   if(!is.null(predictor_scores)) {
      predictor_scores <- as.double(predictor_scores)
   }

   ebm_interaction <- .Call(InitializeInteractionClassification_R, n_classes, features, binned_data, targets, predictor_scores)
   if(is.null(ebm_interaction)) {
      stop("error in InitializeInteractionClassification_R")
   }
   return(ebm_interaction)
}

initialize_interaction_regression <- function(features, binned_data, targets, predictor_scores) {
   features <- as.list(features)
   binned_data <- as.double(binned_data)
   targets <- as.double(targets)
   if(!is.null(predictor_scores)) {
      predictor_scores <- as.double(predictor_scores)
   }

   ebm_interaction <- .Call(InitializeInteractionRegression_R, features, binned_data, targets, predictor_scores)
   if(is.null(ebm_interaction)) {
      stop("error in InitializeInteractionRegression_R")
   }
   return(ebm_interaction)
}

get_interaction_score <- function(ebm_interaction, feature_indexes) {
   stopifnot(class(ebm_interaction) == "externalptr")
   feature_indexes <- as.double(feature_indexes)

   interaction_score <- .Call(GetInteractionScore_R, ebm_interaction, feature_indexes)
   if(is.null(interaction_score)) {
      stop("error in GetInteractionScore_R")
   }
   return(interaction_score)
}

