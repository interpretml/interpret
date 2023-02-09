// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#define _CRT_SECURE_NO_DEPRECATE

#include <stddef.h> // size_t, ptrdiff_t
#include <inttypes.h> // fixed sized integer types and printf strings.  Includes stdint.h
#include <limits> // numeric_limits
#include <type_traits> // std::is_standard_layout, std::is_integral
#include <stdlib.h> // malloc, free
#include <string.h> // strchr, memmove, memcpy
#include <algorithm> // std::sort

#include <vector>
#include <queue>
#include <exception>
#include <functional> // std::function
#include <memory> // std::shared_ptr
  

#include <random>
#include <cmath> // log, exp, etc.  Use cmath instead of math.h so that we get type overloading for these functions for seemless float/double useage

#include <stdio.h> // snprintf/vsnprintf for logging
#include <stdarg.h> // va_start, va_end
#include <assert.h>
