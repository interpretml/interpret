# Copyright (c) 2018 Microsoft Corporation
# Licensed under the MIT license.
# Author: Paul Koch <code@koch.ninja>

sample_without_replacement <- function(random_seed, count_training_samples, count_samples, training_counts_out) {
   random_seed <- as.integer(random_seed)
   count_training_samples <- as.double(count_training_samples)
   count_samples <- as.double(count_samples)
   stopifnot(is.double(training_counts_out))
   stopifnot(count_samples == length(training_counts_out))

   # WARNING, training_counts_out is modified in place, which breaks R norms, but is legal to do per:
   # 5.9.10 Named objects and copying [https://cran.r-project.org/doc/manuals/R-exts.html#Named-objects-and-copying]
   # we modify training_counts_out to avoid extra allocations in the future where we might repeatedly reuse that
   # memory to fill in new samples.  This function is not meant to be used outside of this package
   result <- .Call(SampleWithoutReplacement_R, random_seed, count_training_samples, count_samples, training_counts_out)
   if(is.null(result)) {
      stop("error in SampleWithoutReplacement_R")
   }
   return(NULL)
}
