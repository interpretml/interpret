\name{ebm_predict_proba}
\alias{ebm_predict_proba}
\title{ebm_predict_proba}
\description{
  Predicts probabilities using an EBM model
}
\usage{
ebm_predict_proba(
  model, 
  X
)
}
\arguments{
  \item{model}{the model}
  \item{X}{features}
}
\value{
  returns the probabilities predicted
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
  proba_test <- ebm_predict_proba(ebm, X_test)
}
