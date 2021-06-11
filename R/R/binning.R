# Copyright (c) 2018 Microsoft Corporation
# Licensed under the MIT license.
# Author: Paul Koch <code@koch.ninja>

cut_quantile <- function(
   feature_values, 
   count_samples_per_bin_min, 
   is_humanized, 
   count_cuts
) {
   feature_values <- as.double(feature_values)
   count_samples_per_bin_min <- as.double(count_samples_per_bin_min)
   is_humanized <- as.logical(is_humanized)
   count_cuts <- as.double(count_cuts)

   cuts_lower_bound_inclusive <- .Call(
      CutQuantile_R, 
      feature_values, 
      count_samples_per_bin_min, 
      is_humanized, 
      count_cuts
   )
   if(is.null(cuts_lower_bound_inclusive)) {
      stop("error in CutQuantile_R")
   }
   return(cuts_lower_bound_inclusive)
}

discretize <- function(feature_values, cuts_lower_bound_inclusive, discretized_out) {
   feature_values <- as.double(feature_values)
   cuts_lower_bound_inclusive <- as.double(cuts_lower_bound_inclusive)
   stopifnot(is.double(discretized_out))
   stopifnot(length(feature_values) == length(discretized_out))
   
   # WARNING, discretized_out is modified in place, which breaks R norms, but is legal to do per:
   # 5.9.10 Named objects and copying [https://cran.r-project.org/doc/manuals/R-exts.html#Named-objects-and-copying]
   # we modify discretized_out to avoid extra allocations in the future where we might allocate a large vector
   # and fill it in prior to passing it into our InitializeBoosting functions
   result <- .Call(Discretize_R, feature_values, cuts_lower_bound_inclusive, discretized_out)
   if(is.null(result)) {
      stop("error in Discretize_R")
   }
   return(NULL)
}
