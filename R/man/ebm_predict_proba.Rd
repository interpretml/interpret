\name{ebm_predict_proba}
\alias{ebm_predict_proba}
\title{ebm_predict_proba}
\description{
  Predicts the probabilities of an EBM
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
  returns the probabilities for the predictions
}
\examples{
  feature1 <- c(1,2,1,2,1,2,1,2)
  feature2 <- c(1,2,1,2,1,2,1,2)
  X <- data.frame(feature1, feature2)
  y <- c(0,1,1,0,1,0,1,0)
  model <- ebm_classify(X, y)

  proba <- ebm_predict_proba(model, X)
}
