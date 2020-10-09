# Copyright (c) 2018 Microsoft Corporation
# Licensed under the MIT license.
# Author: Paul Koch <code@koch.ninja>

sampling_without_replacement <- function(random_seed, count_included, count_samples, is_included_out) {
   random_seed <- as.integer(random_seed)
   count_included <- as.double(count_included)
   count_samples <- as.double(count_samples)
   stopifnot(is.logical(is_included_out))
   stopifnot(count_samples == length(is_included_out))

   # WARNING, is_included_out is modified in place, which breaks R norms, but is legal to do per:
   # 5.9.10 Named objects and copying [https://cran.r-project.org/doc/manuals/R-exts.html#Named-objects-and-copying]
   # we modify is_included_out to avoid extra allocations in the future where we might repeatedly reuse that
   # memory to fill in new samples.  This function is not meant to be used outside of this package
   result <- .Call(SamplingWithoutReplacement_R, random_seed, count_included, count_samples, is_included_out)
   if(is.null(result)) {
      stop("error in GenerateQuantileBinCuts_R")
   }
   return(NULL)
}
