% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/navigate.R
\name{Err.lsgl}
\alias{Err.lsgl}
\title{Error Rates}
\usage{
\method{Err}{lsgl}(object, data = NULL, response = object$Y.true,
  loss = "RMSE", ...)
}
\arguments{
\item{object}{a lsgl object.}

\item{data}{a design matrix (the \eqn{X} matrix).}

\item{response}{a matrix of the true responses (the \eqn{Y} matrix).}

\item{loss}{the loss (error) function. Either a function taking two arguments or
one of the following character strings \code{RMSE}, \code{OVE} or \code{SOVE}.}

\item{...}{ignored.}
}
\value{
a vector of errors.
}
\description{
Compute and return an error for each model. The error may be spicifed in the \code{loss} argument.

The root-mean-square error (RMSE) is
\deqn{\frac{1}{K}\sum_{i = 1}^K \sqrt{\frac{1}{N}\sum_{j=1}^N (Y_{ji}-(X\hat \beta)_{ji})^2}}
RMSE is the default error.

The objective value error (OVE) is
\deqn{\|Y - X\hat \beta\|_F}

The scaled objective value error (SOVE) is
\deqn{\frac{1}{NK}\|Y - X\hat \beta\|_F}
}
\examples{

set.seed(100) # This may be removed, it ensures consistency of the daily tests

## Simulate from Y=XB+E, the dimension of Y is N x K, X is N x p, B is p x K

N <- 100 #number of samples
p <- 50 #number of features
K <- 15  #number of groups

# simulate beta matrix and X matrix
B<-matrix(sample(c(rep(1,p*K*0.1),rep(0, p*K-as.integer(p*K*0.1)))),nrow=p,ncol=K)
X1<-matrix(rnorm(N*p,1,1),nrow=N,ncol=p)
Y1 <-X1\%*\%B+matrix(rnorm(N*K,0,1),N,K)

X2<-matrix(rnorm(N*p,1,1),nrow=N,ncol=p)
Y2 <-X2\%*\%B+matrix(rnorm(N*K,0,1),N,K)

#### Fit models using X1
lambda <- lsgl::lambda(X1, Y1, alpha = 1, d = 25L, lambda.min = 5, intercept = FALSE)
fit <- lsgl::fit(X1, Y1, alpha = 1, lambda = lambda, intercept = FALSE)

## Training errors:
Err(fit, X1)

## Errors predicting Y2:
Err(fit, X2, Y2)

#### Do cross validation
fit.cv <- lsgl::cv(X1, Y1, alpha = 1, lambda = lambda, intercept = FALSE)

## Cross validation errors (estimated expected generalization error)
Err(fit.cv)

## Cross validation errors using objective value error measures
Err(fit.cv, loss = "OVE")

}
\author{
Martin Vincent
}

