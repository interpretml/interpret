// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <stddef.h> // size_t, ptrdiff_t
#include <vector>
#include <algorithm>

#include "EbmInternal.h" // INLINE_ALWAYS
#include "Logging.h" // EBM_ASSERT & LOG
#include "FeatureGroup.h"
#include "ThreadStateBoosting.h"

#include "Loss.h"
#include "RegisterLoss.h"

void RegisterLossBase::FinalCheckParameters(const char * sLoss, std::vector<const char *> & usedLocations) {
   std::sort(usedLocations.begin(), usedLocations.end());

   for(const char * sParam : usedLocations) {
      if(sParam != sLoss) {
         throw EbmException(Error_LossParameterUnknown);
      }
      sLoss = strchr(sLoss, ',');
      if(nullptr == sLoss) {
         return;
      }
      ++sLoss;
      if(0 == *SkipWhitespace(sLoss)) {
         return;
      }
   }
   if(0 != *SkipWhitespace(sLoss)) {
      throw EbmException(Error_LossParameterUnknown);
   }
}

const char * RegisterLossBase::CheckLossName(const char * sLoss) const {
   EBM_ASSERT(nullptr != sLoss);

   sLoss = IsStringEqualsCaseInsensitive(sLoss, m_sLossName);
   if(nullptr == sLoss) {
      // we are not the specified loss function
      return nullptr;
   }
   if(0 != *sLoss) {
      if(':' != *sLoss) {
         // we are not the specified objective, but the objective could still be something with a longer string
         // eg: the given tag was "something_else:" but our tag was "something:", so we matched on "something" only
         return nullptr;
      }
      sLoss = SkipWhitespace(sLoss + 1);
   }
   return sLoss;
}

