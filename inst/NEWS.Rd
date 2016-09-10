\name{NEWS}
\title{News for Package 'lsgl'}

\section{Changes in lsgl version 1.2.1 (2016-09-10)}{
\itemize{
\item Improved documentation and functionality:
\itemize{
	\item added functions for extracting number of nonzero features and parameters (\code{feature_stat} and \code{parameter:stat})
	\item Minor updates and documentation corrections
  \item subsampling functionality added
	\item subsampling supports different lambda sequences for each subsample
}
 \item Fixes :
     \itemize{
     	\item minor fixes for some numerical computations
     }
}
}

\section{Changes in lsgl version 1.2.0 (2015-09-16)}{
\itemize{
  \itemize{
  \item Improved reliability and usability:
  \itemize{
  	\item added handling of K = 1 cases
  	\item added support for sample weights
    \item correct handling of non penalized parameters when computing lambda.max (fit non penalized parameters before computing critical lambdas)
    \item increased speed when optimizing with a high number of groups/classes (> 100)
    \item some improvements on error reporting and argument validation
   }

   \item Fixes :
     \itemize{
        \item fixed bug related to grouping
        \item fixed seval minor bugs
        \item fixed startup messages
     }
  }
}
}
