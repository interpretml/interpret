# Copyright (c) 2023 The InterpretML Contributors
# Licensed under the MIT license.
# Author: Paul Koch <code@koch.ninja>

sample_without_replacement <- function(rng, count_training_samples, count_validation_samples, bag_out) {
   stopifnot(is.null(rng) || class(rng) == "externalptr")
   count_training_samples <- as.double(count_training_samples)
   count_validation_samples <- as.double(count_validation_samples)
   stopifnot(is.integer(bag_out))
   stopifnot(count_training_samples + count_validation_samples == length(bag_out))

   # WARNING, bag_out is modified in place, which breaks R norms, but is legal to do per:
   # 5.9.10 Named objects and copying [https://cran.r-project.org/doc/manuals/R-exts.html#Named-objects-and-copying]
   # we modify bag_out to avoid extra allocations in the future where we might repeatedly reuse that
   # memory to fill in new samples.  This function is not meant to be used outside of this package
   result <- .Call(SampleWithoutReplacement_R, rng, count_training_samples, count_validation_samples, bag_out)
   return(NULL)
}
