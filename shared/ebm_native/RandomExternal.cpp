// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include "ebm_native.h" // IntEbmType
#include "EbmInternal.h" // INLINE_ALWAYS
#include "RandomStream.h"

EBM_NATIVE_IMPORT_EXPORT_BODY IntEbmType EBM_NATIVE_CALLING_CONVENTION GenerateRandomNumber(IntEbmType randomSeed) {
   UNUSED(randomSeed);

   // TODO: implement this

   return IntEbmType { 1 };
}

EBM_NATIVE_IMPORT_EXPORT_BODY void EBM_NATIVE_CALLING_CONVENTION SamplingWithoutReplacement(
   IntEbmType randomSeed,
   IntEbmType countSamples,
   IntEbmType countIncluded,
   IntEbmType * isIncludedOut
) {
   UNUSED(randomSeed);
   UNUSED(countSamples);
   UNUSED(countIncluded);
   UNUSED(isIncludedOut);

   // TODO: implement this
}
