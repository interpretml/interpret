# Copyright (c) 2026 The InterpretML Contributors
#
# Licensed under the MIT license.

library(interpret)

X <- data.frame(x = seq_len(100))
y <- c(rep(0, 90), rep(1, 10))

model <- ebm_classify(
   X,
   y,
   outer_bags = 16,
   random_state = 42
)

probabilities <- ebm_predict_proba(model, X)
difference <- abs(mean(probabilities) - mean(y))

if(0.02 <= difference) {
   stop(
      sprintf(
         "Expected mean probability near %.9f, got %.9f",
         mean(y),
         mean(probabilities)
      ),
      call. = FALSE
   )
}
