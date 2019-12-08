# Copyright (c) 2018 Microsoft Corporation
# Licensed under the MIT license.
# Author: Paul Koch <code@koch.ninja>

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

