// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <stddef.h> // size_t, ptrdiff_t

#include "EbmInternal.h" // INLINE_ALWAYS
#include "Logging.h" // EBM_ASSERT & LOG
#include "FeatureAtomic.h"
#include "FeatureGroup.h"

FeatureGroup * FeatureGroup::Allocate(const size_t cFeatures, const size_t iFeatureGroup) {
   const size_t cBytes = GetFeatureGroupCountBytes(cFeatures);
   EBM_ASSERT(0 < cBytes);
   FeatureGroup * const pFeatureGroup = static_cast<FeatureGroup *>(EbmMalloc<void>(cBytes));
   if(UNLIKELY(nullptr == pFeatureGroup)) {
      return nullptr;
   }
   pFeatureGroup->Initialize(cFeatures, iFeatureGroup);
   return pFeatureGroup;
}

FeatureGroup ** FeatureGroup::AllocateFeatureGroups(const size_t cFeatureGroups) {
   LOG_0(TraceLevelInfo, "Entered FeatureGroup::AllocateFeatureGroups");

   EBM_ASSERT(0 < cFeatureGroups);
   FeatureGroup ** const apFeatureGroups = EbmMalloc<FeatureGroup *>(cFeatureGroups);
   if(nullptr != apFeatureGroups) {
      for(size_t i = 0; i < cFeatureGroups; ++i) {
         apFeatureGroups[i] = nullptr;
      }
   }

   LOG_0(TraceLevelInfo, "Exited FeatureGroup::AllocateFeatureGroups");
   return apFeatureGroups;
}

void FeatureGroup::FreeFeatureGroups(const size_t cFeatureGroups, FeatureGroup ** apFeatureGroups) {
   LOG_0(TraceLevelInfo, "Entered FeatureGroup::FreeFeatureGroups");
   if(nullptr != apFeatureGroups) {
      EBM_ASSERT(0 < cFeatureGroups);
      for(size_t i = 0; i < cFeatureGroups; ++i) {
         if(nullptr != apFeatureGroups[i]) {
            FeatureGroup::Free(apFeatureGroups[i]);
         }
      }
      free(apFeatureGroups);
   }
   LOG_0(TraceLevelInfo, "Exited FeatureGroup::FreeFeatureGroups");
}

