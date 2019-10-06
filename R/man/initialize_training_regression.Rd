\name{initialize_training_regression}
\alias{initialize_training_regression}
\title{Initializes Regression Training}
\description{
  Initializes Regression Training
}
\usage{
initialize_training_regression(
  random_seed, 
  features, 
  feature_combinations, 
  feature_combination_indexes, 
  training_targets, 
  training_binned_data, 
  training_predictor_scores, 
  validation_targets, 
  validation_binned_data, 
  validation_predictor_scores, 
  count_inner_bags
)
}
\arguments{
  \item{random_seed}{random seed}
  \item{features}{features}
  \item{feature_combinations}{feature combinations}
  \item{feature_combination_indexes}{feature combination indexes}
  \item{training_targets}{training targets}
  \item{training_binned_data}{training binned data}
  \item{training_predictor_scores}{training predictor scores}
  \item{validation_targets}{validation targets}
  \item{validation_binned_data}{validation binned data}
  \item{validation_predictor_scores}{validation predictor scores}
  \item{count_inner_bags}{count inner bags}
}
\value{
  Returns an opaque externalptr object that can be passed to the function training_step to iteratively boost on features in the explainable boosting machine model, or to the functions get_best_model_feature_combination, or get_current_model_feature_combination to obtain the explainable boosting machine tensors
}
\examples{
  training_ptr <- initialize_training_regression(
    1L, 
    list(ebm_feature(1)), 
    list(ebm_feature_combination(1)), 
    c(0), 
    c(0), c(0), c(0), 
    c(0), c(0), c(0), 
    0L)
}
