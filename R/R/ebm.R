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

ebm_feature_group <- function(feature_indexes) {
   feature_indexes <- as.double(feature_indexes)
   ret <- structure(list(feature_indexes = feature_indexes), class = "ebm_feature_group")
   return(ret)
}

get_count_scores_c <- function(n_classes) {
   if(n_classes <= 2) {
      return (1)
   } else {
      return (n_classes)
   }
}

#TODO: implement ebm_regress

ebm_classify <- function(
   X, 
   y, 
   max_bins = 255,
   outer_bags = 16, 
   inner_bags = 0,
   learning_rate = 0.01, 
   validation_size = 0.15, 
   early_stopping_rounds = 50, 
   early_stopping_tolerance = 1e-4,
   max_rounds = 5000, 
   max_leaves = 3,
   min_samples_leaf = 2,
   random_state = 42
) {
   min_samples_bin <- 5
   humanized <- FALSE # TODO this should be it's own binning type 'quantile_humanized' eventually

   stopifnot(nrow(X) == length(y))
   stopifnot(!any(is.na(X)))
   stopifnot(!any(is.na(y)))
   y <- as.logical(y) # for now we just support binary classification

   random_state <- normalize_initial_random_seed(random_state)
   
   col_names <- colnames(X)
   n_features <- length(col_names)
   bin_cuts <- vector("list")
   features <- vector("list")
   # byrow = FALSE to ensure this matrix is column-major (FORTRAN ordered), which is the fastest memory ordering for us
   X_binned <- matrix(nrow = nrow(X), ncol = ncol(X), byrow = FALSE)
   discretized <- vector("numeric", length(y))
   for(i_feature in 1:n_features) {
      X_feature <- X[, i_feature] # if our originator X matrix is byrow, pay the transpose cost once
      feature_bin_cuts <- generate_quantile_bin_cuts(
         random_state, 
         X_feature, 
         min_samples_bin, 
         humanized, 
         max_bins
      )
      col_name <- col_names[i_feature]
      bin_cuts[[col_name]] <- feature_bin_cuts
      features[[col_name]] <- ebm_feature(n_bins = length(feature_bin_cuts) + 1)
      # WARNING: discretized is modified in-place
      discretize(X_feature, feature_bin_cuts, discretized)
      X_binned[, i_feature] <- discretized
   }

   # create the feature_groups for the mains
   feature_groups <- lapply(1:n_features, function(i) { ebm_feature_group(i) })

   additive_terms <- vector("list")
   for(col_name in col_names) {
      additive_terms[[col_name]] <- vector("numeric", length(bin_cuts[[col_name]]) + 1)
   }

   seed <- random_state
   is_included <- vector("logical", length(y))

   n_classes <- 2 # only binary classification for now
   num_scores <- get_count_scores_c(n_classes)

   validation_size <- ceiling(length(y) * validation_size)
   train_size <- length(y) - validation_size

   scores_train <- vector("numeric", num_scores * train_size)
   scores_val <- vector("numeric", num_scores * validation_size)

   for(i_outer_bag in 1:outer_bags) {
      seed <- generate_random_number(seed, 1416147523)
      # WARNING: is_included is modified in-place
      sampling_without_replacement(seed, validation_size, length(y), is_included)

      X_train <- X_binned[!is_included, ]
      y_train <- y[!is_included]
      X_val <- X_binned[is_included, ]
      y_val <- y[is_included] 

      result_list <- cyclic_gradient_boost(
         "classification",
         n_classes,
         features,
         feature_groups,
         X_train,
         y_train,
         scores_train,
         X_val,
         y_val,
         scores_val,
         inner_bags,
         seed,
         learning_rate,
         early_stopping_rounds,
         early_stopping_tolerance,
         max_rounds,
         max_leaves, 
         min_samples_leaf
      )
      for(i_feature in 1:n_features) {
         additive_terms[[col_names[i_feature]]] <- 
            additive_terms[[col_names[i_feature]]] + result_list$model_update[[i_feature]]
      }
   }
   for(col_name in col_names) {
      additive_terms[[col_name]] <- additive_terms[[col_name]] / outer_bags
   }

   # TODO PK : we're going to need to modify this structure in the future to handle interaction terms by making
   #           the additivie_terms by feature_group index instead of by feature name.  And also change the
   #           bin_cuts to be per-feature_group as well to support stage fitting in the future
   #           For now though, this is just a simple and nice way to present it since we just support mains

   model <- structure(list(bin_cuts = bin_cuts, additive_terms = additive_terms), class = "ebm_model")
   return(model)
}

convert_probability <- function(logit) {
  odds <- exp(logit)
  proba <- odds / (1 + odds)
  return(proba)
}

ebm_predict_proba <- function (model, X) {
   col_names <- colnames(X)
   discretized <- vector("numeric", nrow(X))
   scores <- vector("numeric", nrow(X))
   for(col_name in col_names) {
      # WARNING: discretized is modified in-place
      discretize(X[[col_name]], model$bin_cuts[[col_name]], discretized)

      additive_terms <- model$additive_terms[[col_name]]
      update_scores <- additive_terms[discretized + 1]
      scores <- scores + update_scores
   }

   probabilities <- convert_probability(scores)
   return(probabilities)
}
