# Copyright (c) 2018 Microsoft Corporation
# Licensed under the MIT license.
# Author: Paul Koch <code@koch.ninja>

cut_quantile <- function(
   X_col, 
   min_samples_bin, 
   is_rounded, 
   count_cuts
) {
   X_col <- as.double(X_col)
   min_samples_bin <- as.double(min_samples_bin)
   is_rounded <- as.logical(is_rounded)
   count_cuts <- as.double(count_cuts)

   cuts_lower_bound_inclusive <- .Call(
      CutQuantile_R, 
      X_col, 
      min_samples_bin, 
      is_rounded, 
      count_cuts
   )
   if(is.null(cuts_lower_bound_inclusive)) {
      stop("error in CutQuantile_R")
   }
   return(cuts_lower_bound_inclusive)
}

bin_feature <- function(X_col, cuts_lower_bound_inclusive, bin_indexes_out) {
   X_col <- as.double(X_col)
   cuts_lower_bound_inclusive <- as.double(cuts_lower_bound_inclusive)
   stopifnot(is.double(bin_indexes_out))
   stopifnot(length(X_col) == length(bin_indexes_out))
   
   # WARNING, bin_indexes_out is modified in place, which breaks R norms, but is legal to do per:
   # 5.9.10 Named objects and copying [https://cran.r-project.org/doc/manuals/R-exts.html#Named-objects-and-copying]
   # we modify bin_indexes_out to avoid extra allocations in the future where we might allocate a large vector
   # and fill it in prior to passing it into our InitializeBoosting functions
   result <- .Call(BinFeature_R, X_col, cuts_lower_bound_inclusive, bin_indexes_out)
   if(is.null(result)) {
      stop("error in BinFeature_R")
   }
   return(NULL)
}
