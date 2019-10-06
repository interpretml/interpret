\name{initialize_interaction_classification}
\alias{initialize_interaction_classification}
\title{Initializes Classification Interaction}
\description{
  Initializes Classification Interaction
}
\usage{
initialize_interaction_classification(
  features, 
  count_target_classes, 
  targets, 
  binned_data, 
  predictor_scores
)
}
\arguments{
  \item{features}{features}
  \item{count_target_classes}{count target classes}
  \item{targets}{targets}
  \item{binned_data}{binned data}
  \item{predictor_scores}{predictor scores}
}
\value{
  Returns an opaque externalptr object that can be passed to the function get_interaction_score to obtain the interaction score for any given set of features
}
\examples{
  interaction_ptr <- initialize_interaction_classification(
    list(ebm_feature(1)), 
    2, 
    c(0), c(0), c(0))
}
