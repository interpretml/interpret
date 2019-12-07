\name{ebm_feature_combination}
\alias{ebm_feature_combination}
\title{Create ebm_feature_combination}
\description{
  Creates an ebm_feature_combination
}
\usage{
ebm_feature_combination(
  feature_indexes
)
}
\arguments{
  \item{feature_indexes}{indexes of the features that compose the feature_combination}
}
\value{
  Returns an S3 ebm_feature_combination class that contains information about a combination of ebm_feature objects
}
\examples{
  feature_combination <- ebm_feature_combination(1)
}
