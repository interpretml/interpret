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
  num_outer_bags = 16, 
  validation_size = 0.15, 
  max_epochs = 2000, 
  num_early_stopping_run_length = 50, 
  learning_rate = 0.01, 
  max_tree_splits = 2, 
  min_instances_for_split = 2, 
  random_state = 42
)
}
\arguments{
  \item{X}{features}
  \item{y}{targets}
  \item{num_outer_bags}{number of outer bags}
  \item{validation_size}{amount of data to use for validation}
  \item{max_epochs}{number of boosting rounds}
  \item{num_early_stopping_run_length}{how many rounds without improvement before we quit}
  \item{learning_rate}{learning rate}
  \item{max_tree_splits}{how many splits allowed}
  \item{min_instances_for_split}{number of instances required for a split}
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
