# Copyright (c) 2018 Microsoft Corporation
# Licensed under the MIT license.
# Author: Paul Koch <code@koch.ninja>

# useful links.. check these periodically for issues
# https://CRAN.R-project.org/package=interpret
# https://cran.r-project.org/web/checks/check_results_interpret.html
# https://cran.r-project.org/web/checks/check_summary_by_package.html#summary_by_package
# https://cran.r-project.org/web/checks/check_flavors.html

# incoming status:
# https://cransays.itsalocke.com/articles/dashboard.html
# ftp://cran.r-project.org/incoming/

# if archived, it will appear in:
# https://cran-archive.r-project.org/web/checks/2019-10-07_check_results_interpret.html
# https://cran.r-project.org/src/contrib/Archive/

# we can test our package against many different systems with:
# https://builder.r-hub.io

# S3 data structures

ebm_feature <- function(n_bins, has_missing = FALSE, feature_type = "ordinal") {
   n_bins <- as.double(n_bins)
   stopifnot(is.logical(has_missing))
   feature_type <- match.arg(feature_type, c("ordinal", "nominal"))
   ret <- structure(list(n_bins = n_bins, has_missing = has_missing, feature_type = feature_type), class = "ebm_feature")
   return(ret)
}

ebm_feature_combination <- function(feature_indexes) {
   feature_indexes <- as.double(feature_indexes)
   ret <- structure(list(feature_indexes = feature_indexes), class = "ebm_feature_combination")
   return(ret)
}

create_main_feature_combinations <- function(features) {
   feature_combinations <- lapply(seq_along(features), function(i) { ebm_feature_combination(i) })
   return(feature_combinations)
}


