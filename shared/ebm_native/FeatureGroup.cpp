// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <stddef.h> // size_t, ptrdiff_t

#include "EbmInternal.h" // EBM_INLINE
#include "Logging.h" // EBM_ASSERT & LOG
#include "FeatureAtomic.h"
#include "FeatureGroup.h"

FeatureCombination * FeatureCombination::Allocate(const size_t cFeatures, const size_t iFeatureCombination) {
   const size_t cBytes = GetFeatureCombinationCountBytes(cFeatures);
   EBM_ASSERT(0 < cBytes);
   FeatureCombination * const pFeatureCombination = static_cast<FeatureCombination *>(EbmMalloc<void>(cBytes));
   if(UNLIKELY(nullptr == pFeatureCombination)) {
      return nullptr;
   }
   pFeatureCombination->Initialize(cFeatures, iFeatureCombination);
   return pFeatureCombination;
}

FeatureCombination ** FeatureCombination::AllocateFeatureCombinations(const size_t cFeatureCombinations) {
   LOG_0(TraceLevelInfo, "Entered FeatureCombination::AllocateFeatureCombinations");

   EBM_ASSERT(0 < cFeatureCombinations);
   FeatureCombination ** const apFeatureCombinations = EbmMalloc<FeatureCombination *>(cFeatureCombinations);
   if(nullptr != apFeatureCombinations) {
      for(size_t i = 0; i < cFeatureCombinations; ++i) {
         apFeatureCombinations[i] = nullptr;
      }
   }

   LOG_0(TraceLevelInfo, "Exited FeatureCombination::AllocateFeatureCombinations");
   return apFeatureCombinations;
}

void FeatureCombination::FreeFeatureCombinations(const size_t cFeatureCombinations, FeatureCombination ** apFeatureCombinations) {
   LOG_0(TraceLevelInfo, "Entered FeatureCombination::FreeFeatureCombinations");
   if(nullptr != apFeatureCombinations) {
      EBM_ASSERT(0 < cFeatureCombinations);
      for(size_t i = 0; i < cFeatureCombinations; ++i) {
         if(nullptr != apFeatureCombinations[i]) {
            FeatureCombination::Free(apFeatureCombinations[i]);
         }
      }
      free(apFeatureCombinations);
   }
   LOG_0(TraceLevelInfo, "Exited FeatureCombination::FreeFeatureCombinations");
}

