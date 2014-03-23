\name{Err.lsgl}
\alias{Err.lsgl}
\title{Compute error rates}
\usage{
  \method{Err}{lsgl} (object, data = NULL,
    response = object$Y.true, y = response, ...)
}
\arguments{
  \item{object}{a lsgl object}

  \item{data}{a design matrix (the \eqn{X} matrix)}

  \item{response}{redirected to \code{y}}

  \item{y}{a matrix of the true responses (the \eqn{Y}
  matrix)}

  \item{...}{ignored}
}
\value{
  a vector of error rates
}
\description{
  Compute and return the error \deqn{\frac{1}{NK}\|Y -
  X\hat B\|_F} for each model.
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

X2<-matrix(rnorm(N*p,1,1),nrow=N,ncol=p)
Y2 <-X2\%*\%B+matrix(rnorm(N*K,0,1),N,K)

#### Fit models using X1
lambda <- lsgl.lambda(X1, Y1, alpha = 1, d = 25L, lambda.min = 0.5, intercept = FALSE)
fit <- lsgl(X1, Y1, alpha = 1, lambda = lambda, intercept = FALSE)

## Training errors:
Err(fit, X1)

## Errors predicting Y2:
Err(fit, X2, Y2)

#### Do cross validation
fit.cv <- lsgl.cv(X1, Y1, alpha = 1, lambda = lambda, intercept = FALSE)

## Cross validation errors (estimated expected generalization error)
Err(fit.cv)
}
\author{
  Martin Vincent
}

