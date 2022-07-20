# Copyright (c) 2018 Microsoft Corporation
# Licensed under the MIT license.
# Author: Paul Koch <code@koch.ninja>

create_classification_interaction_detector <- function(
   count_classes, 
   features_categorical,
   features_bin_count,
   bin_indexes, 
   targets, 
   weights, 
   predictor_scores
) {
   count_classes <- as.double(count_classes)
   features_categorical <- as.logical(features_categorical)
   features_bin_count <- as.double(features_bin_count)
   bin_indexes <- as.double(bin_indexes)
   targets <- as.double(targets)
   if(!is.null(weights)) {
      weights <- as.double(weights)
   }
   if(!is.null(predictor_scores)) {
      predictor_scores <- as.double(predictor_scores)
   }

   interaction_handle <- .Call(
      CreateClassificationInteractionDetector_R, 
      count_classes, 
      features_categorical,
      features_bin_count,
      bin_indexes, 
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
   features_categorical,
   features_bin_count,
   bin_indexes, 
   targets, 
   weights, 
   predictor_scores
) {
   features_categorical <- as.logical(features_categorical)
   features_bin_count <- as.double(features_bin_count)
   bin_indexes <- as.double(bin_indexes)
   targets <- as.double(targets)
   if(!is.null(weights)) {
      weights <- as.double(weights)
   }
   if(!is.null(predictor_scores)) {
      predictor_scores <- as.double(predictor_scores)
   }

   interaction_handle <- .Call(
      CreateRegressionInteractionDetector_R, 
      features_categorical,
      features_bin_count,
      bin_indexes, 
      targets, 
      weights, 
      predictor_scores
   )
   if(is.null(interaction_handle)) {
      stop("error in CreateRegressionInteractionDetector_R")
   }
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
