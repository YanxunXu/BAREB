\docType{data}

\name{initial}

\alias{initial}

\title{The initial value of patient- and site-level covariates for simulation}


\usage{
data("initial")
}

\description{
These data record the initial value of patient- and site-level covariates in simulation in the paper "BAREB: A Bayesian Repulsive Biclustering Model for Periodontal Data". It is obtained by simple linear regression.

The variables are:
\tabular{ll}{
	\code{Beta0}          \tab  the initial value of patient-level covariates \cr
	\code{Gamma0}         \tab  the initial value of site-level covariates
	
}
}

\examples{
# output patient level covariates
data("initial")
Beta0
Gamma0
}
\keyword{datasets}
