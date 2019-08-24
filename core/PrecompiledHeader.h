// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include <type_traits> // std::is_pod
#include <inttypes.h> // any of the fixed size types like int64_t
#include <vector>
#include <map>
#include <random>
#include <new> // std::nothrow
#include <assert.h>
#include <queue>
#include <string.h> // memset
#include <stdlib.h> // malloc, realloc, free
#include <cmath> // log, exp, sqrt, etc.  Use cmath instead of math.h so that we get type overloading for these functions for seemless float/double useage
#include <stddef.h> // size_t, ptrdiff_t
#include <limits> // numeric_limits
#include <tuple>
#include <stdio.h> // snprintf/vsnprintf for logging
#include <stdarg.h> // va_start, va_end
