# Copyright (c) 2018 Microsoft Corporation
# Licensed under the MIT license.
# Author: Paul Koch <code@koch.ninja>

create_interaction_detector <- function(
   dataset_handle,
   bag,
   init_scores
) {
   stopifnot(class(dataset_handle) == "externalptr")
   if(!is.null(bag)) {
      bag <- as.integer(bag)
   }
   if(!is.null(init_scores)) {
      init_scores <- as.double(init_scores)
   }
   interaction_handle <- .Call(
      CreateInteractionDetector_R, 
      dataset_handle,
      bag,
      init_scores
   )
   return(interaction_handle)
}

calc_interaction_strength <- function(interaction_handle, feature_indexes, count_samples_required_for_child_split_min) {
   stopifnot(class(interaction_handle) == "externalptr")
   feature_indexes <- as.double(feature_indexes)
   count_samples_required_for_child_split_min <- as.double(count_samples_required_for_child_split_min)

   interaction_strength <- .Call(
      CalcInteractionStrength_R, 
      interaction_handle, 
      feature_indexes, 
      count_samples_required_for_child_split_min
   )
   if(is.null(interaction_strength)) {
      stop("error in CalcInteractionStrength_R")
   }
   return(interaction_strength)
}
