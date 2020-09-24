// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

//#define LOG_SUPERVERBOSE_DISCRETIZATION_ORDERED
//#define LOG_SUPERVERBOSE_DISCRETIZATION_UNORDERED

// TODO: use noexcept throughout our codebase (exception extern "C" functions) !  The compiler can optimize functions better if it knows there are no exceptions
// TODO: review all the C++ library calls, including things like std::abs and verify that none of them throw exceptions, otherwise use the C versions that provide this guarantee

#include <stddef.h> // size_t, ptrdiff_t
#include <limits> // std::numeric_limits
#include <algorithm> // std::sort
#include <cmath> // std::round
#include <vector> // std::vector (used in std::priority_queue)
#include <queue> // std::priority_queue
#include <stdio.h> // snprintf
#include <set> // std::set
#include <string.h> // strchr, memmove

#include "ebm_native.h"
#include "EbmInternal.h"
#include "Logging.h" // EBM_ASSERT & LOG
#include "RandomStream.h"

EBM_NATIVE_IMPORT_EXPORT_BODY IntEbmType EBM_NATIVE_CALLING_CONVENTION GenerateWinsorizedBinCuts(
   IntEbmType countSamples,
   FloatEbmType * featureValues,
   IntEbmType randomSeed,
   IntEbmType * countBinCutsInOut,
   FloatEbmType * binCutsLowerBoundInclusiveOut,
   IntEbmType * countMissingValuesOut,
   FloatEbmType * minNonInfinityValueOut,
   IntEbmType * countNegativeInfinityOut,
   FloatEbmType * maxNonInfinityValueOut,
   IntEbmType * countPositiveInfinityOut
) {
   UNUSED(countSamples);
   UNUSED(featureValues);
   UNUSED(randomSeed);
   UNUSED(countBinCutsInOut);
   UNUSED(binCutsLowerBoundInclusiveOut);
   UNUSED(countMissingValuesOut);
   UNUSED(minNonInfinityValueOut);
   UNUSED(countNegativeInfinityOut);
   UNUSED(maxNonInfinityValueOut);
   UNUSED(countPositiveInfinityOut);

   // TODO: IMPLEMENT

   return IntEbmType { 1 };
}

