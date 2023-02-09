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

ebm_term <- function(feature_indexes) {
   feature_indexes <- as.double(feature_indexes)
   ret <- structure(list(feature_indexes = feature_indexes), class = "ebm_term")
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
   is_rounded <- FALSE # TODO this should be it's own binning type 'rounded_quantile' eventually
   n_classes <- 2 # only binary classification for now

   n_features <- ncol(X)
   n_samples <- nrow(X)

   stopifnot(n_samples == length(y))
   stopifnot(!any(is.na(y)))
   # TODO: add missing value support for X
   stopifnot(!any(is.na(X)))

   random_state <- normalize_initial_seed(random_state)
   rng <- create_rng(random_state)
   
   col_names <- colnames(X)
   if(is.null(col_names)) {
      # TODO: should we accept feature names from our caller, and what if those do not match the colum names of X?
      col_names <- 1:n_features
   }

   data = make_dataset(n_classes, X, y, max_bins, col_names)
   dataset = data$dataset
   cuts = data$cuts

   # create the terms for the mains
   terms <- lapply(1:n_features, function(i) { ebm_term(i) })

   term_scores <- vector("list")
   bag <- vector("integer", n_samples)

   num_scores <- get_count_scores_c(n_classes)

   validation_size <- ceiling(n_samples * validation_size)
   train_size <- n_samples - validation_size

   for(i_outer_bag in 1:outer_bags) {
      # WARNING: bag is modified in-place
      sample_without_replacement(rng, train_size, validation_size, bag)

      result_list <- cyclic_gradient_boost(
         "classification",
         n_classes,
         dataset,
         bag,
         NULL,
         terms,
         inner_bags,
         learning_rate,
         min_samples_leaf, 
         max_leaves, 
         early_stopping_rounds, 
         early_stopping_tolerance,
         max_rounds, 
         rng
      )
      for(i_feature in 1:n_features) {
         term_scores[[col_names[i_feature]]] <- result_list$model_update[[i_feature]]
      }
   }
   for(col_name in col_names) {
      term_scores[[col_name]] <- term_scores[[col_name]] / outer_bags
      # for now, zero all missing values
      term_scores[[col_name]][1] <- 0
      # for now, zero all unknown values
      term_scores[[col_name]][length(term_scores[[col_name]])] <- 0
   }

   # TODO PK : we're going to need to modify this structure in the future to handle interaction terms by making
   #           the additivie_terms by term index instead of by feature name.  And also change the
   #           cuts to be per-term as well to support stage fitting in the future
   #           For now though, this is just a simple and nice way to present it since we just support mains

   model <- structure(list(cuts = cuts, term_scores = term_scores), class = "ebm_model")
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

   bin_indexes <- vector("numeric", nrow(X))
   scores <- vector("numeric", nrow(X))
   for(i_feature in 1:n_features) {
      col_name <- col_names[[i_feature]]
      X_col <- X[, i_feature]

      # WARNING: bin_indexes is modified in-place
      discretize(X_col, model$cuts[[col_name]], bin_indexes)

      term_scores <- model$term_scores[[col_name]]
      update_scores <- term_scores[bin_indexes + 1]
      scores <- scores + update_scores
   }

   probabilities <- convert_probability(scores)
   return(probabilities)
}

ebm_show <- function (model, name) {
   cuts <- model$cuts[[name]]
   term_scores <- model$term_scores[[name]]

   if(0 == length(cuts)) {
      # plot seems to overflow if the values are higher
      low_val <- -1e307
      high_val <- 1e307
   } else if(1 == length(cuts)) {
      if(0 == cuts[1]) {
         low_val <- -1
         high_val <- 1
      } else if(cuts[1] < 0) {
         low_val <- cuts[1] * 1.1
         high_val <- cuts[1] * 0.9
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
   # remove the missing bin at the start and remove the unknown bin at the end
   y <- rep(term_scores[2:(length(term_scores) - 1)], each = 2)

   graphics::plot(x, y, type = "l", lty = 1, main = name, xlab="", ylab="score")
}
