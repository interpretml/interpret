\name{initialize_interaction_classification}
\alias{initialize_interaction_classification}
\title{Initializes Classification Interaction}
\description{
  Initializes Classification Interaction
}
\usage{
initialize_interaction_classification(
  count_target_classes, 
  features, 
  binned_data, 
  targets, 
  predictor_scores
)
}
\arguments{
  \item{count_target_classes}{count target classes}
  \item{features}{features}
  \item{binned_data}{binned data}
  \item{targets}{targets}
  \item{predictor_scores}{predictor scores}
}
\value{
  Returns an opaque externalptr object that can be passed to the function get_interaction_score to obtain the interaction score for any given set of features
}
\examples{
  interaction_ptr <- initialize_interaction_classification(
    2, 
    list(ebm_feature(1)), 
    c(0), c(0), c(0))
}
