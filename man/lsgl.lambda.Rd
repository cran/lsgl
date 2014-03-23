\name{lsgl.lambda}
\alias{lsgl.lambda}
\title{Compute a lambda sequence for the regularization path}
\usage{
  lsgl.lambda(x, y, intercept = TRUE,
    grouping = factor(1:ncol(x)),
    groupWeights = c(sqrt(ncol(y) * table(grouping))),
    parameterWeights = matrix(1, nrow = ncol(y), ncol = ncol(x)),
    alpha = 1, d = 100L, lambda.min,
    algorithm.config = lsgl.standard.config)
}
\arguments{
  \item{x}{design matrix, matrix of size \eqn{N \times p}.}

  \item{y}{response matrix, matrix of size \eqn{N \times
  K}.}

  \item{intercept}{should the model include intercept
  parameters}

  \item{grouping}{grouping of features, a factor or vector
  of length \eqn{p}. Each element of the factor/vector
  specifying the group of the feature.}

  \item{groupWeights}{the group weights, a vector of length
  \eqn{m} (the number of groups).}

  \item{parameterWeights}{a matrix of size \eqn{K \times
  p}.}

  \item{alpha}{the \eqn{\alpha} value 0 for group lasso, 1
  for lasso, between 0 and 1 gives a sparse group lasso
  penalty.}

  \item{d}{the length of lambda sequence}

  \item{lambda.min}{the smallest lambda value in the
  computed sequence.}

  \item{algorithm.config}{the algorithm configuration to be
  used.}
}
\value{
  a vector of length \code{d} containing the compute lambda
  sequence.
}
\description{
  Computes a decreasing lambda sequence of length \code{d}.
  The sequence ranges from a data determined maximal lambda
  \eqn{\lambda_\textrm{max}} to the user inputed
  \code{lambda.min}.
}
\author{
  Martin Vincent
}

