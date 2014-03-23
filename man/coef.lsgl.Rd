\name{coef.lsgl}
\alias{coef.lsgl}
\title{Extract nonzero coefficients}
\usage{
  \method{coef}{lsgl} (object, index = 1:nmod(object), ...)
}
\arguments{
  \item{object}{a lsgl object}

  \item{index}{indices of the models}

  \item{...}{ignored}
}
\value{
  a list of length \code{length(index)} with nonzero
  coefficients of the models
}
\description{
  Extract nonzero coefficients
}
\examples{
set.seed(100) # This may be removed, it ensures consistency of the daily tests

## Simulate from Y=XB+E, the dimension of Y is N x K, X is N x p, B is p x K

N <- 100 #number of samples
p <- 50 #number of covariates
K <- 25  #number of groups

B<-matrix(sample(c(rep(1,p*K*0.1),rep(0, p*K-as.integer(p*K*0.1)))),nrow=p,ncol=K)

X<-matrix(rnorm(N*p,1,1),nrow=N,ncol=p)
Y<-X\%*\%B+matrix(rnorm(N*K,0,1),N,K)

lambda<-lsgl.lambda(X,Y, alpha=1, d = 25, lambda.min=.5, intercept=FALSE)
fit <-lsgl(X,Y, alpha=1, lambda = lambda, intercept=FALSE)

# the nonzero coefficients of the models 1, 2 and 5
coef(fit, index = c(1,2,5))
}
\author{
  Martin Vincent
}

