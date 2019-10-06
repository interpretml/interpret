\name{get_current_model_feature_combination}
\alias{get_current_model_feature_combination}
\title{Get Current Model Feature Combination}
\description{
  Get Current Model Feature Combination
}
\usage{
get_current_model_feature_combination(
  ebm_training, 
  index_feature_combination
)
}
\arguments{
  \item{ebm_training}{ebm training}
  \item{index_feature_combination}{index feature combination}
}
\value{
  Returns the current model tensor for a single explainable boosting machine feature combination
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

  model_feature_combination_tensor <- get_current_model_feature_combination(training_ptr, 0)
}
