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

ebm_feature_combination <- function(feature_indexes) {
   feature_indexes <- as.double(feature_indexes)
   ret <- structure(list(feature_indexes = feature_indexes), class = "ebm_feature_combination")
   return(ret)
}

create_main_feature_combinations <- function(features) {
   feature_combinations <- lapply(seq_along(features), function(i) { ebm_feature_combination(i) })
   return(feature_combinations)
}

ebm_classify <- function(
   X, 
   y, 
   num_outer_bags = 16, 
   validation_size = 0.15, 
   max_epochs = 2000, 
   num_early_stopping_run_length = 50, 
   learning_rate = 0.01, 
   max_tree_splits = 2, 
   min_instances_for_split = 2, 
   random_state = 42
) {
   col_names = colnames(X)

   bin_edges <- vector(mode = "list") #, ncol(X))
   # TODO: I know this binning is buggy.  Review
   for(col_name in col_names) bin_edges[[col_name]] <- unique(quantile(X[[col_name]], seq(0,1, 1.0 / 256)))
   for(col_name in col_names) bin_edges[[col_name]] <- bin_edges[[col_name]][2:(length(bin_edges[[col_name]])-1)]
   for(col_name in col_names) X[[col_name]] <- as.integer(findInterval(X[[col_name]], bin_edges[[col_name]]))
   features <- lapply(col_names, function(col_name) { ebm_feature(n_bins = length(bin_edges[[col_name]]) + 1) })
   feature_combinations <- create_main_feature_combinations(features)
   
   set.seed(random_state)
   val_indexes = sample(1:length(y), ceiling(length(y) * validation_size))

   X_train = X[-val_indexes,]
   y_train = y[-val_indexes]
   X_val = X[val_indexes,]
   y_val = y[val_indexes] 

   X_train_vec <- vector(mode = "numeric") # , ncol(X_train) * nrow(X_train)
   for(col_name in col_names) X_train_vec[(length(X_train_vec) + 1):(length(X_train_vec) + length(X_train[[col_name]]))] <- X_train[[col_name]]

   X_val_vec <- vector(mode = "numeric") # , ncol(X_val) * nrow(X_val)
   for(col_name in col_names) X_val_vec[(length(X_val_vec) + 1):(length(X_val_vec) + length(X_val[[col_name]]))] <- X_val[[col_name]]

   n_classes = 2
   num_scores <- get_count_scores_c(n_classes) # only binary classification for now
   scores_train <- numeric(num_scores * length(y_train))
   scores_val <- numeric(num_scores * length(y_val))

   result_list = cyclic_gradient_boost(
      "classification",
      n_classes,
      features,
      feature_combinations,
      X_train_vec,
      y_train,
      scores_train,
      X_val_vec,
      y_val,
      scores_val,
      0,
      random_state,
      learning_rate,
      max_tree_splits, 
      min_instances_for_split, 
      max_epochs,
      num_early_stopping_run_length
   )
   model <- vector(mode = "list")
   for(i in seq_along(col_names)) {
      model[[col_names[[i]]]] <- result_list$model_update[[i]]
   }

   return(list(bin_edges = bin_edges, model = model))
}

convert_probability <- function(logit) {
  odds <- exp(logit)
  proba <- odds / (1 + odds)
  return(proba)
}

get_count_scores_c <- function(n_classes) {
   if(n_classes <= 2) {
      return (1)
   } else {
      return (n_classes)
   }
}

ebm_predict_proba <- function (model, X) {
   col_names = colnames(X)
   X_binned <- vector(mode = "list") #, ncol(X))
   for(col_name in col_names) X_binned[[col_name]] <- as.integer(findInterval(X[[col_name]], model$bin_edges[[col_name]]) + 1)

   scores <- vector(mode = "numeric", nrow(X))
   for(col_name in col_names) {
      bin_vals <- model$model[[col_name]]
      bin_indexes <- X_binned[[col_name]]
      update_scores <- bin_vals[bin_indexes]
      scores <- scores + update_scores
   }

   probabilities <- convert_probability(scores)
   return(probabilities)
}
