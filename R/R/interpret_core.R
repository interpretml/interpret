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

ebm_classify <- function(X, y, n_estimators = 16, test_size = 0.15, data_n_episodes = 2000, early_stopping_run_length = 50, learning_rate = 0.01, max_tree_splits = 2, min_cases_for_splits = 2, random_state = 42) {
   col_names = colnames(X)

   bin_edges <- vector(mode = "list", ncol(X))
   # TODO: I know this binning is buggy.  Review
   for(col_name in col_names) bin_edges[[col_name]] <- unique(quantile(X[[col_name]], seq(0,1, 1.0 / 256)))
   for(col_name in col_names) bin_edges[[col_name]] <- bin_edges[[col_name]][2:(length(bin_edges[[col_name]])-1)]
   for(col_name in col_names) X[[col_name]] <- as.integer(findInterval(X[[col_name]], bin_edges[[col_name]]))
   features <- lapply(col_names, function(col_name) { ebm_feature(n_bins = length(bin_edges[[col_name]])) })
   feature_combinations <- create_main_feature_combinations(features)
   
   set.seed(random_state)
   val_indexes = sample(1:length(y), ceiling(length(y) * test_size))

   X_train = X[-val_indexes,]
   y_train = y[-val_indexes]
   X_val = X[val_indexes,]
   y_val = y[val_indexes] 

   X_train_vec <- vector(mode = "numeric") # , ncol(X_train) * nrow(X_train)
   for(col_name in col_names) X_train_vec[(length(X_train_vec) + 1):(length(X_train_vec) + length(X_train[[col_name]]))] <- X_train[[col_name]]

   X_val_vec <- vector(mode = "numeric") # , ncol(X_val) * nrow(X_val)
   for(col_name in col_names) X_val_vec[(length(X_val_vec) + 1):(length(X_val_vec) + length(X_val[[col_name]]))] <- X_val[[col_name]]

   result_list = cyclic_gradient_boost(
      "classification",
      2,
      features,
      feature_combinations,
      X_train_vec,
      y_train,
      NULL,
      X_val_vec,
      y_val,
      NULL,
      0,
      random_state,
      learning_rate,
      max_tree_splits, 
      min_cases_for_splits, 
      data_n_episodes,
      early_stopping_run_length
   )
   return(result_list)
}
