# BayRepulsive R package

This package provides a tool to implement the BayRepulsive algorithm: a Bayesian repulsive deconvolution model. 

## Installation


You can manually obtain BayRepulsive.tar.gz and install it directly:

```
R CMD install BayRepulsive.tar.gz
```

Before using the package, please make sure that R packages "mvtnorm", "alabama", "psych", "optimx" are also installed. 


## Usage

```R
rm(list=ls())
library(BayRepulsive)
data(CCLE)
set.seed(1)
Nobs     <- 24
Nfeature <- 100
K0       <- 3
### randomly generate weight matrix W for 24 mixing samples
W        <- matrix(0,nrow = K0, ncol = Nobs)
for(i in 1:Nobs){
  Theta <- rgamma(K0,1/K0,1)
  W[,i] <- Theta/sum(Theta)
}
### add some noise
error    <- t(matrix(rnorm(Nfeature * Nobs, mean = 0, sd = 0.5), nrow = Nobs))
DATA     <- CCLE$Z%*%W + error
### Note: please make sure that there are no negative values after adding the noise
result1  <- BayRepulsive_known(DATA = DATA, K = K0, Nobs = Nobs,
                               Nfeature = Nfeature)
cor(as.vector(result1$W), as.vector(W))



```

Full documentation and usage information is available in the manual:

* [BayRepulsive.pdf](https://github.com/bruce1995/BayRepulsive/blob/master/BayRepulsive.pdf)


## License

This library is made available under the Johns Hopkins University License. This is an open-source project of Yanxun's lab.
