// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <stddef.h> // size_t, ptrdiff_t
#include <limits> // numeric_limits

#include "ebm_native.h"

#include "EbmInternal.h"
// very independent includes
#include "Logging.h" // EBM_ASSERT & LOG

EBM_NATIVE_IMPORT_EXPORT_BODY IntEbmType EBM_NATIVE_CALLING_CONVENTION GenerateDiscretizationCutPoints(
   IntEbmType countInstances,
   FloatEbmType * singleFeatureValues,
   IntEbmType countMaximumBins,
   IntEbmType countMinimumInstancesPerBin,
   FloatEbmType * cutPointsLowerBoundInclusive,
   IntEbmType * countCutPoints,
   IntEbmType * isMissing
) {
   LOG_N(TraceLevelInfo, "Entered GenerateDiscretizationCutPoints: countInstances=%" IntEbmTypePrintf ", singleFeatureValues=%p, countMaximumBins=%"
      IntEbmTypePrintf ", countMinimumInstancesPerBin=%" IntEbmTypePrintf ", cutPointsLowerBoundInclusive=%p, countCutPoints=%p, isMissingPresent=%p",
      countInstances, 
      static_cast<void *>(singleFeatureValues), 
      countMaximumBins, 
      countMinimumInstancesPerBin, 
      static_cast<void *>(cutPointsLowerBoundInclusive), 
      static_cast<void *>(countCutPoints),
      static_cast<void *>(isMissing)
   );
   IntEbmType ret = 0;


   if(0 != ret) {
      LOG_N(TraceLevelWarning, "WARNING GenerateDiscretizationCutPoints returned %" IntEbmTypePrintf, ret);
   } else {
      LOG_N(TraceLevelInfo, "Exited GenerateDiscretizationCutPoints countCutPoints=%" IntEbmTypePrintf ", isMissing=%" IntEbmTypePrintf,
         *countCutPoints,
         *isMissing
      );
   }
   return ret;
}

EBM_NATIVE_IMPORT_EXPORT_BODY void EBM_NATIVE_CALLING_CONVENTION Discretize(
   IntEbmType isMissing,
   IntEbmType countCutPoints,
   const FloatEbmType * cutPointsLowerBoundInclusive,
   IntEbmType countInstances,
   const FloatEbmType * singleFeatureValues,
   IntEbmType * singleFeatureDiscretized
) {
   LOG_N(TraceLevelInfo, "Entered Discretize: isMissing=%" IntEbmTypePrintf ", countCutPoints=%" IntEbmTypePrintf 
      ", cutPointsLowerBoundInclusive=%p, countInstances=%" IntEbmTypePrintf ", singleFeatureValues=%p, singleFeatureDiscretized=%p",
      isMissing,
      countCutPoints,
      static_cast<const void *>(cutPointsLowerBoundInclusive),
      countInstances,
      static_cast<const void *>(singleFeatureValues),
      static_cast<void *>(singleFeatureDiscretized)
   );


   LOG_0(TraceLevelInfo, "Exited Discretize");
}
