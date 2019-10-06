\name{initialize_interaction_regression}
\alias{initialize_interaction_regression}
\title{Initializes Regression Interaction}
\description{
  Initializes Regression Interaction
}
\usage{
initialize_interaction_regression(
  features, 
  targets, 
  binned_data, 
  predictor_scores
)
}
\arguments{
  \item{features}{features}
  \item{targets}{targets}
  \item{binned_data}{binned data}
  \item{predictor_scores}{predictor scores}
}
\value{
  Returns an opaque externalptr object that can be passed to the function get_interaction_score to obtain the interaction score for any given set of features
}
\examples{
  interaction_ptr <- initialize_interaction_regression(
    list(ebm_feature(1)), 
    c(0), c(0), c(0))
}
