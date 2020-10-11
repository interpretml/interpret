\name{ebm_show}
\alias{ebm_show}
\title{ebm_show}
\description{
  Shows the GAM plot for a single feature
}
\usage{
ebm_show(
  model, 
  name
)
}
\arguments{
  \item{model}{the model}
  \item{name}{the name of the feature to plot}
}
\value{
  None
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
  ebm_show(ebm, "mpg")
}
