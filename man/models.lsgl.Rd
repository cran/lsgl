\name{models.lsgl}
\alias{models.lsgl}
\title{Exstract the fitted models}
\usage{
  \method{models}{lsgl} (object, index = 1:nmod(object),
    ...)
}
\arguments{
  \item{object}{a lsgl object}

  \item{index}{indices of the models to be returned}

  \item{...}{ignored}
}
\value{
  a list of \eqn{\beta} matrices.
}
\description{
  Returns the fitted models, that is the estimated
  \eqn{\beta} matrices.
}
\author{
  Martin Vincent
}

