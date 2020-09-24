// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <stddef.h> // size_t, ptrdiff_t
#include <limits> // std::numeric_limits

#include "ebm_native.h"
#include "EbmInternal.h"
#include "Logging.h" // EBM_ASSERT & LOG

EBM_NATIVE_IMPORT_EXPORT_BODY IntEbmType EBM_NATIVE_CALLING_CONVENTION GenerateUniformBinCuts(
   IntEbmType countSamples,
   FloatEbmType * featureValues,
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

