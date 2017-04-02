params <-
structure(list(pkg = "lsgl"), .Names = "pkg")

## ---- echo = FALSE-------------------------------------------------------
set.seed(150)

pkg_version <- packageVersion(params$pkg)

x_dcf <- read.dcf(file = file.path(getwd(), "..", "DESCRIPTION"))

if("GitHubRepo" %in% colnames(x_dcf)) {
  pkg_branch <- x_dcf[1,"GitHubRepo"]

  pkg_version_type <- switch(pkg_branch,
    release = "release",
    master = "release candidate",
    develop = "development version"
  )
} else {
  pkg_branch <- "release"
  pkg_version_type <- "release"
}

## ----eval = FALSE--------------------------------------------------------
#  install.packages("lsgl")

## ----eval = FALSE--------------------------------------------------------
#  # install.packages("devtools")
#  devtools::install_github("vincent-dk/sglOptim")
#  devtools::install_github("vincent-dk/lsgl")

## ----eval = FALSE--------------------------------------------------------
#  # install.packages("devtools")
#  devtools::install_github("vincent-dk/sglOptim", ref = "develop")
#  devtools::install_github("vincent-dk/lsgl", ref = "develop")

## ----message = FALSE-----------------------------------------------------
library(lsgl)

# Load sone data
data(AirlineTicketPrices)

# Setup 2 parallel units
cl <- makeCluster(2)
registerDoParallel(cl)

# Do 10-fold cross validation on 100 models with increasing complexity, using the 2 parallel units
fit.cv <- lsgl::cv(
  x = X,
  y = Y,
  alpha = 0.5,
  lambda = 0.01,
  use_parallel = TRUE
)

stopCluster(cl)

# Print information about models
# and cross validation errors
fit.cv

