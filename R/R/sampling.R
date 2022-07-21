# Copyright (c) 2018 Microsoft Corporation
# Licensed under the MIT license.
# Author: Paul Koch <code@koch.ninja>

sample_without_replacement <- function(random_state, count_training_samples, count_validation_samples, sample_counts_out) {
   random_state <- as.integer(random_state)
   count_training_samples <- as.double(count_training_samples)
   count_validation_samples <- as.double(count_validation_samples)
   stopifnot(is.double(sample_counts_out))
   stopifnot(count_training_samples + count_validation_samples == length(sample_counts_out))

   # WARNING, sample_counts_out is modified in place, which breaks R norms, but is legal to do per:
   # 5.9.10 Named objects and copying [https://cran.r-project.org/doc/manuals/R-exts.html#Named-objects-and-copying]
   # we modify sample_counts_out to avoid extra allocations in the future where we might repeatedly reuse that
   # memory to fill in new samples.  This function is not meant to be used outside of this package
   result <- .Call(SampleWithoutReplacement_R, random_state, count_training_samples, count_validation_samples, sample_counts_out)
   if(is.null(result)) {
      stop("error in SampleWithoutReplacement_R")
   }
   return(NULL)
}
