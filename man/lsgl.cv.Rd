\name{lsgl.cv}
\alias{lsgl.cv}
\title{Linear multiple output cross validation using multiple possessors}
\usage{
  lsgl.cv(x, y, intercept = TRUE,
    grouping = factor(1:ncol(x)),
    groupWeights = c(sqrt(ncol(y) * table(grouping))),
    parameterWeights = matrix(1, nrow = ncol(y), ncol = ncol(x)),
    alpha = 1, lambda, fold = 10L, cv.indices = list(),
    max.threads = 2L,
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

  \item{lambda}{the lambda sequence for the regularization
  path.}

  \item{fold}{the fold of the cross validation, an integer
  larger than 1 and less than \eqn{N+1}. Ignored if
  \code{cv.indices != NULL}.}

  \item{cv.indices}{a list of indices of a cross validation
  splitting. If \code{cv.indices = NULL} then a random
  splitting will be generated using the \code{fold}
  argument.}

  \item{max.threads}{the maximal number of threads to be
  used.}

  \item{algorithm.config}{the algorithm configuration to be
  used.}
}
\value{
  \item{Yhat}{the cross validation estimated response
  matrix} \item{Y.true}{the true response matrix, this is
  equal to the argument \code{y}} \item{cv.indices}{the
  cross validation splitting used} \item{features}{number
  of features used in the models} \item{parameters}{number
  of parameters used in the models.}
}
\description{
  Linear multiple output cross validation using multiple
  possessors
}
\examples{
set.seed(100) # This may be removed, it ensures consistency of the daily tests

## Simulate from Y=XB+E, the dimension of Y is N x K, X is N x p, B is p x K

N <- 100 #number of samples
p <- 50 #number of features
K <- 25  #number of groups

B<-matrix(sample(c(rep(1,p*K*0.1),rep(0, p*K-as.integer(p*K*0.1)))),nrow=p,ncol=K)
X1<-matrix(rnorm(N*p,1,1),nrow=N,ncol=p)
Y1 <-X1\%*\%B+matrix(rnorm(N*K,0,1),N,K)

##Do cross validation
lambda <- lsgl.lambda(X1, Y1, alpha = 1, d = 25L, lambda.min = 0.5, intercept = FALSE)
fit.cv <- lsgl.cv(X1, Y1, alpha = 1, lambda = lambda, intercept = FALSE)

## Cross validation errors (estimated expected generalization error)
Err(fit.cv)
}
\author{
  Martin Vincent
}

