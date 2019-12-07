\name{initialize_boosting_regression}
\alias{initialize_boosting_regression}
\title{Initializes Regression Boosting}
\description{
  Initializes Regression Boosting
}
\usage{
initialize_boosting_regression(
  features, 
  feature_combinations, 
  feature_combination_indexes, 
  training_binned_data, 
  training_targets, 
  training_predictor_scores, 
  validation_binned_data, 
  validation_targets, 
  validation_predictor_scores, 
  n_inner_bags,
  random_seed
)
}
\arguments{
  \item{features}{features}
  \item{feature_combinations}{feature combinations}
  \item{feature_combination_indexes}{feature combination indexes}
  \item{training_binned_data}{training binned data}
  \item{training_targets}{training targets}
  \item{training_predictor_scores}{training predictor scores}
  \item{validation_binned_data}{validation binned data}
  \item{validation_targets}{validation targets}
  \item{validation_predictor_scores}{validation predictor scores}
  \item{n_inner_bags}{count inner bags}
  \item{random_seed}{random seed}
}
\value{
  Returns an opaque externalptr object that can be passed to the function boosting_step to iteratively boost on features in the explainable boosting machine model, or to the functions get_best_model_feature_combination, or get_current_model_feature_combination to obtain the explainable boosting machine tensors
}
\examples{
  boosting_ptr <- initialize_boosting_regression(
    list(ebm_feature(1)), 
    list(ebm_feature_combination_c(1)), 
    c(0), 
    c(0), c(0), c(0), 
    c(0), c(0), c(0), 
    0L, 1L)
}
