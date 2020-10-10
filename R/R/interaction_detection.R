# Copyright (c) 2018 Microsoft Corporation
# Licensed under the MIT license.
# Author: Paul Koch <code@koch.ninja>

initialize_interaction_classification <- function(
   count_target_classes, 
   features, 
   binned_data, 
   targets, 
   predictor_scores
) {
   count_target_classes <- as.double(count_target_classes)
   features <- as.list(features)
   binned_data <- as.double(binned_data)
   targets <- as.double(targets)
   if(!is.null(predictor_scores)) {
      predictor_scores <- as.double(predictor_scores)
   }

   interaction_pointer <- .Call(
      InitializeInteractionClassification_R, 
      count_target_classes, 
      features, 
      binned_data, 
      targets, 
      predictor_scores
   )
   if(is.null(interaction_pointer)) {
      stop("error in InitializeInteractionClassification_R")
   }
   return(interaction_pointer)
}

initialize_interaction_regression <- function(features, binned_data, targets, predictor_scores) {
   features <- as.list(features)
   binned_data <- as.double(binned_data)
   targets <- as.double(targets)
   if(!is.null(predictor_scores)) {
      predictor_scores <- as.double(predictor_scores)
   }

   interaction_pointer <- .Call(InitializeInteractionRegression_R, features, binned_data, targets, predictor_scores)
   if(is.null(interaction_pointer)) {
      stop("error in InitializeInteractionRegression_R")
   }
   return(interaction_pointer)
}

calculate_interaction_score <- function(interaction_pointer, feature_indexes, count_samples_required_for_child_split_min) {
   stopifnot(class(interaction_pointer) == "externalptr")
   feature_indexes <- as.double(feature_indexes)
   count_samples_required_for_child_split_min <- as.double(count_samples_required_for_child_split_min)

   interaction_score <- .Call(
      CalculateInteractionScore_R, 
      interaction_pointer, 
      feature_indexes, 
      count_samples_required_for_child_split_min
   )
   if(is.null(interaction_score)) {
      stop("error in CalculateInteractionScore_R")
   }
   return(interaction_score)
}
