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
# In particular, the "Oracle Developer Studio 12.6" is worth testing as that C++ compiler is picky, and CRAN tests it

# S3 data structures

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
   min_samples_leaf = 2, 
   max_leaves = 3, 
   random_state = 42
) {
   min_samples_bin <- 5
   rounded <- FALSE # TODO this should be it's own binning type 'rounded_quantile' eventually

   stopifnot(nrow(X) == length(y))
   stopifnot(!any(is.na(X)))
   stopifnot(!any(is.na(y)))
   y <- as.logical(y) # for now we just support binary classification

   random_state <- normalize_initial_random_seed(random_state)
   
   n_features <- ncol(X)
   col_names <- colnames(X)
   if(is.null(col_names)) {
      col_names <- 1:n_features
   }

   cuts <- vector("list")

   features_categorical <- vector("logical", n_features)
   features_bin_count <- vector("numeric", n_features)

   # byrow = FALSE to ensure this matrix is column-major (FORTRAN ordered), which is the fastest memory ordering for us
   X_binned <- matrix(nrow = nrow(X), ncol = ncol(X), byrow = FALSE)
   discretized <- vector("numeric", length(y))
   for(i_feature in 1:n_features) {
      X_feature <- X[, i_feature] # if our originator X matrix is byrow, pay the transpose cost once
      feature_cuts <- cut_quantile(
         X_feature, 
         min_samples_bin, 
         rounded, 
         max_bins
      )
      col_name <- col_names[i_feature]
      cuts[[col_name]] <- feature_cuts

      features_categorical[[i_feature]] <- FALSE
      # one more bin than cuts plus one more for the missing value
      features_bin_count[[i_feature]] <- length(feature_cuts) + 2

      # WARNING: discretized is modified in-place
      discretize(X_feature, feature_cuts, discretized)
      X_binned[, i_feature] <- discretized
   }

   # create the feature_groups for the mains
   feature_groups <- lapply(1:n_features, function(i) { ebm_feature_group(i) })

   additive_terms <- vector("list")
   for(col_name in col_names) {
      additive_terms[[col_name]] <- vector("numeric", length(cuts[[col_name]]) + 1)
   }

   sample_counts <- vector("numeric", length(y))

   n_classes <- 2 # only binary classification for now
   num_scores <- get_count_scores_c(n_classes)

   validation_size <- ceiling(length(y) * validation_size)
   train_size <- length(y) - validation_size

   scores_train <- vector("numeric", num_scores * train_size)
   scores_val <- vector("numeric", num_scores * validation_size)

   for(i_outer_bag in 1:outer_bags) {
      random_state <- generate_random_number(random_state, 1416147523)
      # WARNING: sample_counts is modified in-place
      sample_without_replacement(random_state, train_size, validation_size, sample_counts)

      X_train <- X_binned[0 < sample_counts, ]
      y_train <- y[0 < sample_counts]
      X_val <- X_binned[sample_counts < 0, ]
      y_val <- y[sample_counts < 0] 

      result_list <- cyclic_gradient_boost(
         "classification",
         n_classes,
         features_categorical,
         features_bin_count,
         feature_groups,
         X_train,
         y_train,
         NULL,
         scores_train,
         X_val,
         y_val,
         NULL,
         scores_val,
         inner_bags,
         learning_rate,
         min_samples_leaf, 
         max_leaves, 
         early_stopping_rounds,
         early_stopping_tolerance,
         max_rounds,
         random_state
      )
      for(i_feature in 1:n_features) {
         additive_terms[[col_names[i_feature]]] <- 
            additive_terms[[col_names[i_feature]]] + result_list$model_update[[i_feature]]
      }
   }
   for(col_name in col_names) {
      additive_terms[[col_name]] <- additive_terms[[col_name]] / outer_bags
      # for now, zero all missing values
      additive_terms[[col_name]][0] = 0
   }

   # TODO PK : we're going to need to modify this structure in the future to handle interaction terms by making
   #           the additivie_terms by feature_group index instead of by feature name.  And also change the
   #           cuts to be per-feature_group as well to support stage fitting in the future
   #           For now though, this is just a simple and nice way to present it since we just support mains

   model <- structure(list(cuts = cuts, additive_terms = additive_terms), class = "ebm_model")
   return(model)
}

convert_probability <- function(logit) {
  odds <- exp(logit)
  proba <- odds / (1 + odds)
  return(proba)
}

ebm_predict_proba <- function (model, X) {

   n_features <- ncol(X)
   col_names <- colnames(X)
   if(is.null(col_names)) {
      col_names <- 1:n_features
   }

   discretized <- vector("numeric", nrow(X))
   scores <- vector("numeric", nrow(X))
   for(i_feature in 1:n_features) {
      col_name <- col_names[[i_feature]]
      X_feature <- X[, i_feature]

      # WARNING: discretized is modified in-place
      discretize(X_feature, model$cuts[[col_name]], discretized)

      additive_terms <- model$additive_terms[[col_name]]
      update_scores <- additive_terms[discretized + 1]
      scores <- scores + update_scores
   }

   probabilities <- convert_probability(scores)
   return(probabilities)
}

ebm_show <- function (model, name) {
   cuts <- model$cuts[[name]]
   additive_terms <- model$additive_terms[[name]]

   if(0 == length(cuts)) {
      # plot seems to overflow if the values are higher
      low_val <- -1e307
      high_val <- 1e307
   } else if(1 == length(cuts)) {
      if(0 == cuts[1]) {
         low_val <- -1
         high_val <- 1
      } else {
         low_val <- cuts[1] * 0.9
         high_val <- cuts[1] * 1.1
      }
   } else {
      dist <- 0.1 * (cuts[length(cuts)] - cuts[1])
      low_val <- cuts[1] - dist
      high_val <- cuts[length(cuts)] + dist
   }

   x <- append(append(low_val, rep(cuts, each = 2)), high_val)
   # remove the missing bin at the start
   y <- rep(additive_terms[2:length(additive_terms)], each = 2)

   graphics::plot(x, y, type = "l", lty = 1, main = name, xlab="", ylab="score")
}
