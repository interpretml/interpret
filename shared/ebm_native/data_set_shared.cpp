// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t
#include <string.h> // memcpy

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "ebm_internal.hpp"

#include "Feature.hpp"
#include "FeatureGroup.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

size_t AppendDenseFeatureData(
   const size_t cSamples,
   const size_t cStrideItems,
   //const size_t cBinItems,
   const IntEbmType * const aInputData,
   const size_t iByteFirst,
   char * const pFillMem
) {
   size_t iByteCur = iByteFirst;
   if(0 != cSamples) {
      const size_t cBytesStride = cStrideItems * sizeof(IntEbmType);
      const IntEbmType * pInputData = aInputData;
      const IntEbmType * const aInputDataEnd = aInputData + cSamples;
      do {
         if(nullptr != pFillMem) {
            // TODO: bit compact this
            *reinterpret_cast<IntEbmType *>(pFillMem + iByteCur) = *pInputData;
         }
         // TODO: increment by the bit compaction
         iByteCur += cBytesStride;
         ++pInputData;
      } while(aInputDataEnd != pInputData);
   }
   return iByteCur;
}


} // DEFINED_ZONE_NAME
