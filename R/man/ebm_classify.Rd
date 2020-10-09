\name{ebm_classify}
\alias{ebm_classify}
\title{Build an EBM classification model}
\description{
  Builds a classification model
}
\usage{
ebm_classify(
  X, 
  y, 
  max_bins = 255,
  outer_bags = 16, 
  inner_bags = 0,
  learning_rate = 0.01, 
  validation_size = 0.15, 
  early_stopping_rounds = 50, 
  early_stopping_tolerance = 1e-4,
  max_rounds = 5000, 
  max_leaves = 3,
  min_samples_leaf = 2,
  random_state = 42
)
}
\arguments{
  \item{X}{features}
  \item{y}{targets}
  \item{max_bins}{number of bins to create}
  \item{outer_bags}{number of outer bags}
  \item{inner_bags}{number of inner bags}
  \item{learning_rate}{learning rate}
  \item{validation_size}{amount of data to use for validation}
  \item{early_stopping_rounds}{how many rounds without improvement before we quit}
  \item{early_stopping_tolerance}{how much does the round need to improve by to be considered as an advacement}
  \item{max_rounds}{number of boosting rounds}
  \item{max_leaves}{how many leaves allowed}
  \item{min_samples_leaf}{number of samples required for a split}
  \item{random_state}{random seed}
}
\value{
  Returns an EBM model
}
\examples{
  feature1 <- c(1,2,1,2,1,2,1,2)
  feature2 <- c(1,2,1,2,1,2,1,2)
  X <- data.frame(feature1, feature2)
  y <- c(0,1,1,0,1,0,1,0)
  model <- ebm_classify(X, y)
}
