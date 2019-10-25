// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include <stddef.h> // size_t, ptrdiff_t
#include <inttypes.h> // any of the fixed size types like int64_t
#include <limits> // numeric_limits
#include <type_traits> // std::is_standard_layout, std::is_integral
#include <stdlib.h> // malloc, realloc, free
#include <string.h> // memset
#include <new> // std::nothrow

#include <vector>
#include <queue>

#include <random>
#include <cmath> // log, exp, sqrt, etc.  Use cmath instead of math.h so that we get type overloading for these functions for seemless float/double useage

#include <stdio.h> // snprintf/vsnprintf for logging
#include <stdarg.h> // va_start, va_end
#include <assert.h>
