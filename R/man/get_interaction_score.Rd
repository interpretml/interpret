\name{get_interaction_score}
\alias{get_interaction_score}
\title{Get Interaction Score}
\description{
  Get Interaction Score
}
\usage{
get_interaction_score(
  ebm_interaction, 
  feature_indexes
)
}
\arguments{
  \item{ebm_interaction}{ebm interaction}
  \item{feature_indexes}{feature indexes}
}
\value{
  Returns an interaction score that indicates the relative benefit of a proposed set of features as inputs to EBM boosting
}
\examples{
  interaction_ptr <- initialize_interaction_regression(
    list(ebm_feature(1)), 
    c(0), c(0), c(0))

  interaction_score <- get_interaction_score(interaction_ptr, 0)
}
