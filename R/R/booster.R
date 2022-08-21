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

create_booster <- function(
   random_state,
   dataset_handle,
   bag,
   init_scores,
   dimension_counts,
   feature_indexes,
   count_inner_bags
) {
   if(!is.null(random_state)) {
      random_state <- as.integer(random_state)
   }
   stopifnot(class(dataset_handle) == "externalptr")
   if(!is.null(bag)) {
      bag <- as.integer(bag)
   }
   if(!is.null(init_scores)) {
      init_scores <- as.double(init_scores)
   }
   dimension_counts <- as.double(dimension_counts)
   feature_indexes <- as.double(feature_indexes)
   count_inner_bags <- as.double(count_inner_bags)

   booster_handle <- .Call(
      CreateBooster_R, 
      random_state,
      dataset_handle,
      bag,
      init_scores,
      dimension_counts,
      feature_indexes,
      count_inner_bags
   )
   return(booster_handle)
}

free_booster <- function(booster_handle) {
   .Call(FreeBooster_R, booster_handle)
   return(NULL)
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
   return(avg_gain)
}

apply_term_update <- function(
   booster_handle
) {
   stopifnot(class(booster_handle) == "externalptr")

   avg_validation_metric <- .Call(ApplyTermUpdate_R, booster_handle)
   return(avg_validation_metric)
}

get_best_term_scores <- function(booster_handle, index_term) {
   stopifnot(class(booster_handle) == "externalptr")
   index_term <- as.double(index_term)

   term_scores <- .Call(GetBestTermScores_R, booster_handle, index_term)
   return(term_scores)
}

get_current_term_scores <- function(booster_handle, index_term) {
   stopifnot(class(booster_handle) == "externalptr")
   index_term <- as.double(index_term)

   term_scores <- .Call(GetCurrentTermScores_R, booster_handle, index_term)
   return(term_scores)
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
   dataset_handle,
   bag,
   init_scores,
   terms,
   inner_bags,
   random_state
) {
   c_structs <- convert_terms_to_c(terms)

   booster_handle <- create_booster(
      random_state,
      dataset_handle,
      bag,
      init_scores,
      c_structs$feature_counts,
      c_structs$feature_indexes,
      inner_bags
   )

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
   dataset_handle,
   bag,
   init_scores,
   terms,
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

   c_structs <- convert_terms_to_c(terms)

   ebm_booster <- booster(
      model_type,
      n_classes,
      dataset_handle,
      bag,
      init_scores,
      terms,
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

            avg_validation_metric <- apply_term_update(ebm_booster$booster_handle)

            if(avg_validation_metric < min_metric) {
               min_metric <- avg_validation_metric
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
      free_booster(ebm_booster$booster_handle)
   })
   return(result_list)
}
