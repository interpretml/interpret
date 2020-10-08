# Copyright (c) 2018 Microsoft Corporation
# Licensed under the MIT license.
# Author: Paul Koch <code@koch.ninja>

generate_quantile_bin_cuts <- function(feature_values, count_samples_per_bin_min, is_humanized, random_seed, count_bin_cuts) {
   feature_values <- as.double(feature_values)
   count_samples_per_bin_min <- as.double(count_samples_per_bin_min)
   is_humanized <- as.logical(is_humanized)
   random_seed <- as.integer(random_seed)
   count_bin_cuts <- as.double(count_bin_cuts)

   bin_cuts <- .Call(GenerateQuantileBinCuts_R, feature_values, count_samples_per_bin_min, is_humanized, random_seed, count_bin_cuts)
   if(is.null(bin_cuts)) {
      stop("error in GenerateQuantileBinCuts_R")
   }
   return(bin_cuts)
}

discretize <- function(feature_values, bin_cuts_lower_bound_inclusive, discretized_out) {
   feature_values <- as.double(feature_values)
   bin_cuts_lower_bound_inclusive <- as.double(bin_cuts_lower_bound_inclusive)
   stopifnot(is.double(discretized_out))
   stopifnot(length(discretized_out) == length(feature_values))

   result <- .Call(Discretize_R, feature_values, bin_cuts_lower_bound_inclusive, discretized_out)
   if(is.null(result)) {
      stop("error in GenerateQuantileBinCuts_R")
   }
   return(NULL)
}
