\name{ebm_feature}
\alias{ebm_feature}
\title{Create ebm_feature}
\description{
  Creates an ebm_feature
}
\usage{
ebm_feature(
  count_bins, 
  has_missing, 
  feature_type
)
}
\arguments{
  \item{count_bins}{count bins}
  \item{has_missing}{has missing}
  \item{feature_type}{feature type}
}
\value{
  Returns an S3 ebm_feature class that contains information about a single machine learning feature
}
\examples{
  feature <- ebm_feature(1, FALSE, "ordinal")
}
