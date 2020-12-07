# Copyright (c) 2018 Microsoft Corporation
# Licensed under the MIT license.
# Author: Paul Koch <code@koch.ninja>

create_classification_interaction_detector <- function(
   count_target_classes, 
   features, 
   binned_data, 
   targets, 
   weights, 
   predictor_scores
) {
   count_target_classes <- as.double(count_target_classes)
   features <- as.list(features)
   binned_data <- as.double(binned_data)
   targets <- as.double(targets)
   if(!is.null(weights)) {
      weights <- as.double(weights)
   }
   if(!is.null(predictor_scores)) {
      predictor_scores <- as.double(predictor_scores)
   }

   interaction_handle <- .Call(
      CreateClassificationInteractionDetector_R, 
      count_target_classes, 
      features, 
      binned_data, 
      targets, 
      weights, 
      predictor_scores
   )
   if(is.null(interaction_handle)) {
      stop("error in CreateClassificationInteractionDetector_R")
   }
   return(interaction_handle)
}

create_regression_interaction_detector <- function(
   features, 
   binned_data, 
   targets, 
   weights, 
   predictor_scores
) {
   features <- as.list(features)
   binned_data <- as.double(binned_data)
   targets <- as.double(targets)
   if(!is.null(weights)) {
      weights <- as.double(weights)
   }
   if(!is.null(predictor_scores)) {
      predictor_scores <- as.double(predictor_scores)
   }

   interaction_handle <- .Call(
      CreateRegressionInteractionDetector_R, 
      features, 
      binned_data, 
      targets, 
      weights, 
      predictor_scores
   )
   if(is.null(interaction_handle)) {
      stop("error in CreateRegressionInteractionDetector_R")
   }
   return(interaction_handle)
}

calculate_interaction_score <- function(interaction_handle, feature_indexes, count_samples_required_for_child_split_min) {
   stopifnot(class(interaction_handle) == "externalptr")
   feature_indexes <- as.double(feature_indexes)
   count_samples_required_for_child_split_min <- as.double(count_samples_required_for_child_split_min)

   interaction_score <- .Call(
      CalculateInteractionScore_R, 
      interaction_handle, 
      feature_indexes, 
      count_samples_required_for_child_split_min
   )
   if(is.null(interaction_score)) {
      stop("error in CalculateInteractionScore_R")
   }
   return(interaction_score)
}
