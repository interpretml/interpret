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
  min_hessian = 1e-3,
  max_leaves = 3,
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
  \item{early_stopping_tolerance}{how much does the round need to improve by to be considered as an advancement}
  \item{max_rounds}{number of boosting rounds}
  \item{min_hessian}{minimum hessian required for a split}
  \item{max_leaves}{how many leaves allowed}
  \item{random_state}{random seed}
}
\value{
  Returns an EBM model
}
\examples{
  data(mtcars)
  X <- subset(mtcars, select = -c(vs))
  y <- mtcars$vs

  set.seed(42)
  data_sample <- sample(length(y), length(y) * 0.8)

  X_train <- X[data_sample, ]
  y_train <- y[data_sample]
  X_test <- X[-data_sample, ]
  y_test <- y[-data_sample]

  ebm <- ebm_classify(X_train, y_train)
}
