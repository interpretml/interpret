# Copyright (c) 2018 Microsoft Corporation
# Licensed under the MIT license.
# Author: Paul Koch <code@koch.ninja>

normalize_initial_random_seed <- function(seed) {
   # Some languages do not support 64-bit values.  Other languages do not support unsigned integers.
   # Almost all languages support signed 32-bit integers, so we standardize on that for our 
   # random number seed values.  If the caller passes us a number that doesn't fit into a 
   # 32-bit signed integer, we convert it.  This conversion doesn't need to generate completely 
   # uniform results provided they are reasonably uniform, since this is just the seed.
   # 
   # We use a simple conversion because we use the same method in multiple languages, 
   # and we need to keep the results identical between them, so simplicity is key.
   # 
   # The result of the modulo operator is not standardized accross languages for 
   # negative numbers, so take the negative before the modulo if the number is negative.
   # https://torstencurdt.com/tech/posts/modulo-of-negative-numbers

   if(2147483647 <= seed) {
      return(seed %% 2147483647)
   }
   if(seed <= -2147483647) {
      return(-((-seed) %% 2147483647))
   }
   return(seed)
}

generate_deterministic_seed <- function(random_seed, stage_randomization_mix) {
   random_seed <- as.integer(random_seed)
   stage_randomization_mix <- as.integer(stage_randomization_mix)

   ret <- .Call(GenerateDeterministicSeed_R, random_seed, stage_randomization_mix)
   if(is.null(ret)) {
      stop("error in GenerateDeterministicSeed_R")
   }
   return(ret)
}
