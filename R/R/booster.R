# Copyright (c) 2018 Microsoft Corporation
# Licensed under the MIT license.
# Author: Paul Koch <code@koch.ninja>

convert_terms_to_c <- function(terms) {
   n_feature_indexes <- 0
   for (term in terms) {
      n_feature_indexes <- n_feature_indexes + length(term$feature_indexes)
   }

   feature_counts <- vector("numeric", length(terms))
   feature_indexes <- vector("numeric", n_feature_indexes)
   index_indexes <- 1

   for (index_term in seq_along(terms)) {
      term <- terms[[index_term]]
      term_feature_indexes <- term$feature_indexes
      feature_counts[[index_term]] <- length(term_feature_indexes)

      for(feature_index in term_feature_indexes) {
         feature_indexes[[index_indexes]] <- feature_index - 1
         index_indexes <- index_indexes + 1
      }
   }
   return(list(feature_counts = feature_counts, feature_indexes = feature_indexes))
}

create_classification_booster <- function(
   random_seed,
   count_classes, 
   features_categorical,
   features_bin_count,
   feature_counts, 
   feature_indexes, 
   training_bin_indexes, 
   training_targets, 
   training_weights, 
   training_init_scores, 
   validation_bin_indexes, 
   validation_targets, 
   validation_weights, 
   validation_init_scores, 
   count_inner_bags
) {
   random_seed <- as.integer(random_seed)
   count_classes <- as.double(count_classes)
   features_categorical <- as.logical(features_categorical)
   features_bin_count <- as.double(features_bin_count)
   feature_counts <- as.double(feature_counts)
   feature_indexes <- as.double(feature_indexes)
   training_bin_indexes <- as.double(training_bin_indexes)
   training_targets <- as.double(training_targets)
   if(!is.null(training_weights)) {
      training_weights <- as.double(training_weights)
   }
   if(!is.null(training_init_scores)) {
      training_init_scores <- as.double(training_init_scores)
   }
   validation_bin_indexes <- as.double(validation_bin_indexes)
   validation_targets <- as.double(validation_targets)
   if(!is.null(validation_weights)) {
      validation_weights <- as.double(validation_weights)
   }
   if(!is.null(validation_init_scores)) {
      validation_init_scores <- as.double(validation_init_scores)
   }
   count_inner_bags <- as.integer(count_inner_bags)

   booster_handle <- .Call(
      CreateClassificationBooster_R, 
      random_seed,
      count_classes, 
      features_categorical,
      features_bin_count,
      feature_counts, 
      feature_indexes, 
      training_bin_indexes, 
      training_targets, 
      training_weights, 
      training_init_scores, 
      validation_bin_indexes, 
      validation_targets, 
      validation_weights, 
      validation_init_scores, 
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
   feature_counts, 
   feature_indexes, 
   training_bin_indexes, 
   training_targets, 
   training_weights, 
   training_init_scores, 
   validation_bin_indexes, 
   validation_targets, 
   validation_weights, 
   validation_init_scores, 
   count_inner_bags
) {
   random_seed <- as.integer(random_seed)
   features_categorical <- as.logical(features_categorical)
   features_bin_count <- as.double(features_bin_count)
   feature_counts <- as.double(feature_counts)
   feature_indexes <- as.double(feature_indexes)
   training_bin_indexes <- as.double(training_bin_indexes)
   training_targets <- as.double(training_targets)
   if(!is.null(training_weights)) {
      training_weights <- as.double(training_weights)
   }
   if(!is.null(training_init_scores)) {
      training_init_scores <- as.double(training_init_scores)
   }
   validation_bin_indexes <- as.double(validation_bin_indexes)
   validation_targets <- as.double(validation_targets)
   if(!is.null(validation_weights)) {
      validation_weights <- as.double(validation_weights)
   }
   if(!is.null(validation_init_scores)) {
      validation_init_scores <- as.double(validation_init_scores)
   }
   count_inner_bags <- as.integer(count_inner_bags)

   booster_handle <- .Call(
      CreateRegressionBooster_R, 
      random_seed,
      features_categorical,
      features_bin_count,
      feature_counts, 
      feature_indexes, 
      training_bin_indexes, 
      training_targets, 
      training_weights, 
      training_init_scores, 
      validation_bin_indexes, 
      validation_targets, 
      validation_weights, 
      validation_init_scores, 
      count_inner_bags
   )
   if(is.null(booster_handle)) {
      stop("Error in CreateRegressionBooster")
   }
   return(booster_handle)
}

generate_term_update <- function(
   booster_handle, 
   index_term, 
   learning_rate, 
   count_samples_required_for_child_split_min, 
   max_leaves
) {
   stopifnot(class(booster_handle) == "externalptr")
   index_term <- as.double(index_term)
   learning_rate <- as.double(learning_rate)
   count_samples_required_for_child_split_min <- as.double(count_samples_required_for_child_split_min)
   max_leaves <- as.double(max_leaves)

   avg_gain <- .Call(
      GenerateTermUpdate_R, 
      booster_handle, 
      index_term, 
      learning_rate, 
      count_samples_required_for_child_split_min, 
      max_leaves
   )
   if(is.null(avg_gain)) {
      stop("error in GenerateTermUpdate_R")
   }
   return(avg_gain)
}

apply_term_update <- function(
   booster_handle
) {
   stopifnot(class(booster_handle) == "externalptr")

   validation_metric <- .Call(
      ApplyTermUpdate_R, 
      booster_handle
   )
   if(is.null(validation_metric)) {
      stop("error in BoostingStep_R")
   }
   return(validation_metric)
}

get_best_term_scores <- function(booster_handle, index_term) {
   stopifnot(class(booster_handle) == "externalptr")
   index_term <- as.double(index_term)

   term_scores <- .Call(GetBestTermScores_R, booster_handle, index_term)
   if(is.null(term_scores)) {
      stop("error in GetBestTermScores_R")
   }
   return(term_scores)
}

get_current_term_scores <- function(booster_handle, index_term) {
   stopifnot(class(booster_handle) == "externalptr")
   index_term <- as.double(index_term)

   term_scores <- .Call(GetCurrentTermScores_R, booster_handle, index_term)
   if(is.null(term_scores)) {
      stop("error in GetCurrentTermScores_R")
   }
   return(term_scores)
}

free_boosting <- function(booster_handle) {
   .Call(FreeBooster_R, booster_handle)
   return(NULL)
}

get_best_model <- function(booster) {
   model <- lapply(
      seq_along(booster$terms), 
      function(i) { get_best_term_scores(booster$booster_handle, i - 1) }
   )
   return(model)
}

get_current_model <- function(booster) {
   model <- lapply(
      seq_along(booster$terms), 
      function(i) { get_current_term_scores(booster$booster_handle, i - 1) }
   )
   return(model)
}

booster <- function(
   model_type,
   n_classes,
   features_categorical,
   features_bin_count,
   terms,
   X_train,
   y_train,
   weights_train, 
   init_scores_train,
   X_val,
   y_val,
   weights_val, 
   init_scores_val,
   inner_bags,
   random_state
) {
   c_structs <- convert_terms_to_c(terms)

   if(model_type == "classification") {
      booster_handle <- create_classification_booster(
         random_state,
         n_classes, 
         features_categorical,
         features_bin_count,
         c_structs$feature_counts, 
         c_structs$feature_indexes, 
         X_train, 
         y_train, 
         weights_train, 
         init_scores_train, 
         X_val, 
         y_val, 
         weights_val, 
         init_scores_val, 
         inner_bags
      )
   } else if(model_type == "regression") {
      booster_handle <- create_regression_booster(
         random_state,
         features_categorical,
         features_bin_count,
         c_structs$feature_counts, 
         c_structs$feature_indexes, 
         X_train, 
         y_train, 
         weights_train, 
         init_scores_train, 
         X_val, 
         y_val, 
         weights_val, 
         init_scores_val, 
         inner_bags
      )
   } else {
      stop("Unrecognized model_type")
   }

   self <- structure(list(
      model_type = model_type, 
      n_classes = n_classes, 
      terms = terms, 
      booster_handle = booster_handle
   ), class = "booster")
   return(self)
}

cyclic_gradient_boost <- function(
   model_type,
   n_classes,
   features_categorical,
   features_bin_count,
   terms,
   X_train,
   y_train,
   weights_train, 
   init_scores_train,
   X_val,
   y_val,
   weights_val, 
   init_scores_val,
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
      terms,
      X_train,
      y_train,
      weights_train, 
      init_scores_train,
      X_val,
      y_val,
      weights_val, 
      init_scores_val,
      inner_bags,
      random_state
   )
   result_list <- tryCatch({
      no_change_run_length <- 0
      bp_metric <- Inf

      for(episode_index in 1:max_rounds) {
         for(term_index in seq_along(terms)) {
            avg_gain <- generate_term_update(
               ebm_booster$booster_handle, 
               term_index - 1, 
               learning_rate, 
               min_samples_leaf, 
               max_leaves
            )

            validation_metric <- apply_term_update(ebm_booster$booster_handle)

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
