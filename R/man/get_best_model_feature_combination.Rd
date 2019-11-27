\name{get_best_model_feature_combination}
\alias{get_best_model_feature_combination}
\title{Get Best Model Feature Combination}
\description{
  Get Best Model Feature Combination
}
\usage{
get_best_model_feature_combination(
  ebm_boosting, 
  index_feature_combination
)
}
\arguments{
  \item{ebm_boosting}{ebm boosting}
  \item{index_feature_combination}{index feature combination}
}
\value{
  Returns the best model tensor for a single explainable boosting machine feature combination
}
\examples{
  boosting_ptr <- initialize_boosting_regression(
    list(ebm_feature(2)), 
    list(ebm_feature_combination(1)), 
    c(0), 
    c(10, 10), c(0, 1), c(0, 0), 
    c(12), c(1), c(0), 
    0L, 1L)

  validation_metric <- boosting_step(boosting_ptr, 0, 0.01, 2, 2, NULL, NULL)

  model_feature_combination_tensor <- get_best_model_feature_combination(boosting_ptr, 0)
}
