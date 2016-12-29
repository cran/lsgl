## ---- echo = FALSE-------------------------------------------------------
set.seed(150)

## ----results="hide"------------------------------------------------------
library(lsgl)

## ----eval = FALSE--------------------------------------------------------
#  X <- # load design matrix (of size N x p)
#  Y <- # load response matrix (of size N x K)

## ------------------------------------------------------------------------
data(AirlineTicketPrices)
dim(X)
dim(Y)

## ------------------------------------------------------------------------
idx <- sample(1:nrow(X), size = 50)

Xtest <- X[idx, ]
Ytest <- Y[idx, ]
X <- X[-idx, ]
Y <- Y[-idx, ]

## ------------------------------------------------------------------------
cl <- makeCluster(2)
registerDoParallel(cl)

# Do cross validation -- this may take some time
fit.cv <- lsgl::cv(X, Y, fold = 10, alpha = 0.5, lambda = 0.001, use_parallel = TRUE)

stopCluster(cl)

## ------------------------------------------------------------------------
fit.cv

## ------------------------------------------------------------------------
fit <- lsgl::fit(X, Y, alpha = 0.5, lambda = 0.001)

## ------------------------------------------------------------------------
fit

## ------------------------------------------------------------------------
features(fit)[[best_model(fit.cv)]][1:10] # Ten first non-zero features in best model

## ------------------------------------------------------------------------
image(parameters(fit)[[best_model(fit.cv)]])

## ------------------------------------------------------------------------
coef(fit, best_model(fit.cv))[,1:5] # First 5 non-zero parameters of best model

## ----eval = FALSE--------------------------------------------------------
#  Xtest <- # load matrix with test data (of size M x p)

## ------------------------------------------------------------------------
res <- predict(fit, Xtest)

## ------------------------------------------------------------------------
image(Ytest, main = "Observed prices")
image(res$Yhat[[best_model(fit.cv)]], main = "Predicted prices")

## ------------------------------------------------------------------------
plot(Err(fit, Xtest, Ytest), xlab = "lambda index")

