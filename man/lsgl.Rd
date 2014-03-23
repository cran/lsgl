\name{lsgl}
\alias{lsgl}
\title{Fit a linear multiple output model using sparse group lasso}
\usage{
  lsgl(x, y, intercept = TRUE,
    grouping = factor(1:ncol(x)),
    groupWeights = c(sqrt(ncol(y) * table(grouping))),
    parameterWeights = matrix(1, nrow = ncol(y), ncol = ncol(x)),
    alpha = 1, lambda,
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

  \item{lambda}{lambda sequence.}

  \item{algorithm.config}{the algorithm configuration to be
  used.}
}
\value{
  \item{beta}{the fitted parameters -- the list
  \eqn{\hat\beta(\lambda(1)), \dots, \hat\beta(\lambda(d))}
  of length \code{length(return)}. With each entry of list
  holding the fitted parameters, that is matrices of size
  \eqn{K\times p} (if \code{intercept = TRUE} matrices of
  size \eqn{K\times (p+1)})} \item{loss}{the values of the
  loss function.} \item{objective}{the values of the
  objective function (i.e. loss + penalty).}
  \item{lambda}{the lambda values used.}
}
\description{
  For a linear multiple output model with \eqn{p} features
  (covariates) dived into \eqn{m} groups using sparse group
  lasso.
}
\details{
  This function computes a sequence of minimizers (one for
  each lambda given in the \code{lambda} argument) of
  \deqn{\frac{1}{N}\|Y-X\beta\|_F^2 + \lambda \left(
  (1-\alpha) \sum_{J=1}^m \gamma_J \|\beta^{(J)}\|_2 +
  \alpha \sum_{i=1}^{n} \xi_i |\beta_i| \right)} where
  \eqn{\|\cdot\|_F} is the frobenius norm. The vector
  \eqn{\beta^{(J)}} denotes the parameters associated with
  the \eqn{J}'th group of features. The group weights are
  denoted by \eqn{\gamma \in [0,\infty)^m} and the
  parameter weights by \eqn{\xi \in [0,\infty)^n}.
}
\examples{
set.seed(100) # This may be removed, it ensures consistency of the daily tests

## Simulate from Y=XB+E, the dimension of Y is N x K, X is N x p, B is p x K

N <- 50 #number of samples
p <- 50 #number of features
K <- 25  #number of groups

B<-matrix(sample(c(rep(1,p*K*0.1),rep(0, p*K-as.integer(p*K*0.1)))),nrow=p,ncol=K)

X<-matrix(rnorm(N*p,1,1),nrow=N,ncol=p)
Y<-X\%*\%B+matrix(rnorm(N*K,0,1),N,K)

lambda<-lsgl.lambda(X,Y, alpha=1, lambda.min=.5, intercept=FALSE)

fit <-lsgl(X,Y, alpha=1, lambda = lambda, intercept=FALSE)

## ||B - \\beta||_F
sapply(fit$beta, function(beta) sum((B - beta)^2))

## Plot
par(mfrow = c(3,1))
image(B, main = "True B")
image(as.matrix(fit$beta[[100]]), main = "Lasso estimate (lambda = 0.5)")
image(solve(t(X)\%*\%X)\%*\%t(X)\%*\%Y, main = "Least squares estimate")

# The training error of the models
Err(fit, X)
# In this cases this is simply the loss function
1/(sqrt(N)*K)*sqrt(fit$loss)
}
\author{
  Martin Vincent
}

