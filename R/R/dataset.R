# Copyright (c) 2018 Microsoft Corporation
# Licensed under the MIT license.
# Author: Paul Koch <code@koch.ninja>

measure_dataset_header <- function(n_features, n_weights, n_targets) {
   n_features <- as.double(n_features)
   n_weights <- as.double(n_weights)
   n_targets <- as.double(n_targets)

   n_bytes <- .Call(MeasureDataSetHeader_R, n_features, n_weights, n_targets)

   return(n_bytes)
}

measure_feature <- function(n_bins, is_missing, is_unknown, is_nominal, bin_indexes) {
   n_bins <- as.double(n_bins)
   is_missing <- as.logical(is_missing)
   is_unknown <- as.logical(is_unknown)
   is_nominal <- as.logical(is_nominal)
   bin_indexes <- as.double(bin_indexes)

   n_bytes <- .Call(MeasureFeature_R, n_bins, is_missing, is_unknown, is_nominal, bin_indexes)

   return(n_bytes)
}

measure_classification_target <- function(n_classes, targets) {
   n_classes <- as.double(n_classes)
   targets <- as.double(targets)

   n_bytes <- .Call(MeasureClassificationTarget_R, n_classes, targets)

   return(n_bytes)
}

create_dataset <- function(n_bytes) {
   n_bytes <- as.double(n_bytes)
   dataset <- .Call(CreateDataSet_R, n_bytes)
   return(dataset)
}

free_dataset <- function(dataset) {
   .Call(FreeDataSet_R, dataset)
   return(NULL)
}

fill_dataset_header <- function(n_features, n_weights, n_targets, n_bytes_allocated, incomplete_dataset) {
   n_features <- as.double(n_features)
   n_weights <- as.double(n_weights)
   n_targets <- as.double(n_targets)
   n_bytes_allocated <- as.double(n_bytes_allocated)
   stopifnot(class(incomplete_dataset) == "externalptr")

   .Call(FillDataSetHeader_R, n_features, n_weights, n_targets, n_bytes_allocated, incomplete_dataset)
   
   return(NULL)
}

fill_feature <- function(n_bins, is_missing, is_unknown, is_nominal, bin_indexes, n_bytes_allocated, incomplete_dataset) {
   n_bins <- as.double(n_bins)
   is_missing <- as.logical(is_missing)
   is_unknown <- as.logical(is_unknown)
   is_nominal <- as.logical(is_nominal)
   bin_indexes <- as.double(bin_indexes)
   n_bytes_allocated <- as.double(n_bytes_allocated)
   stopifnot(class(incomplete_dataset) == "externalptr")

   .Call(FillFeature_R, n_bins, is_missing, is_unknown, is_nominal, bin_indexes, n_bytes_allocated, incomplete_dataset)

   return(NULL)
}

fill_classification_target <- function(n_classes, targets, n_bytes_allocated, incomplete_dataset) {
   n_classes <- as.double(n_classes)
   targets <- as.double(targets)
   n_bytes_allocated <- as.double(n_bytes_allocated)
   stopifnot(class(incomplete_dataset) == "externalptr")

   .Call(FillClassificationTarget_R, n_classes, targets, n_bytes_allocated, incomplete_dataset)

   return(NULL)
}

make_dataset <- function(n_classes, X, y, max_bins, col_names) {
   n_features <- ncol(X)
   n_weights <- 0
   n_targets <- 1

   min_samples_bin <- 5
   is_rounded <- FALSE # TODO this should be it's own binning type 'rounded_quantile' eventually

   cuts <- vector("list")
   bin_indexes <- vector("numeric", length(y))

   n_bytes <- measure_dataset_header(n_features, n_weights, n_targets)

   for(i_feature in 1:n_features) {
      X_col <- X[, i_feature]

      feature_cuts <- cut_quantile(
         X_col, 
         min_samples_bin, 
         is_rounded, 
         max_bins - 3
      )
      col_name <- col_names[i_feature]
      cuts[[col_name]] <- feature_cuts

      # WARNING: bin_indexes is modified in-place
      discretize(X_col, feature_cuts, bin_indexes)

      n_bins = length(feature_cuts) + 3
      is_missing <- TRUE
      is_unknown <- TRUE
      is_nominal <- FALSE

      n_bytes <- n_bytes + measure_feature(n_bins, is_missing, is_unknown, is_nominal, bin_indexes)
   }

   n_bytes <- n_bytes + measure_classification_target(n_classes, y)

   dataset = create_dataset(n_bytes)

   fill_dataset_header(n_features, n_weights, n_targets, n_bytes, dataset)

   for(i_feature in 1:n_features) {
      X_col <- X[, i_feature]

      col_name <- col_names[i_feature]
      feature_cuts <- cuts[[col_name]]

      # WARNING: bin_indexes is modified in-place
      discretize(X_col, feature_cuts, bin_indexes)

      n_bins = length(feature_cuts) + 3
      is_missing <- TRUE
      is_unknown <- TRUE
      is_nominal <- FALSE

      fill_feature(n_bins, is_missing, is_unknown, is_nominal, bin_indexes, n_bytes, dataset)
   }

   fill_classification_target(n_classes, y, n_bytes, dataset)

   return(list("dataset" = dataset, "cuts" = cuts))
}
