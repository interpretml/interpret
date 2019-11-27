\name{training_step}
\alias{training_step}
\title{Training Step}
\description{
  Takes one Training Step
}
\usage{
training_step(
  ebm_boosting, 
  index_feature_combination, 
  learning_rate, 
  count_tree_splits_max, 
  count_instances_required_for_parent_split_min, 
  training_weights, 
  validation_weights
)
}
\arguments{
  \item{ebm_boosting}{ebm training}
  \item{index_feature_combination}{index feature combination}
  \item{learning_rate}{learning rate}
  \item{count_tree_splits_max}{count tree splits max}
  \item{count_instances_required_for_parent_split_min}{count instances required for parent split min}
  \item{training_weights}{training weights}
  \item{validation_weights}{validation weights}
}
\value{
  Returns the root mean squared error when the model being trained is for a regression problem, or the log loss for a classification problem
}
\examples{
  training_ptr <- initialize_training_regression(
    1L, 
    list(ebm_feature(2)), 
    list(ebm_feature_combination(1)),
    c(0), 
    c(10, 10), c(0, 1), c(0, 0), 
    c(12), c(1), c(0), 
    0L)

  validation_metric <- training_step(training_ptr, 0, 0.01, 2, 2, NULL, NULL)
}
