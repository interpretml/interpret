// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef TRANSPOSE_HPP
#define TRANSPOSE_HPP

#include <stddef.h> // size_t, ptrdiff_t

#include "logging.h" // EBM_ASSERT
#include "zones.h"

#include "Feature.hpp"
#include "Term.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

struct TransposeDimension {
   size_t cBins;
   size_t iTranspose;
   bool bDropFirst;
   bool bDropLast;

   size_t cBinsReduced;
   size_t iBinsRemaining;
   size_t cBytesStride;
};

template<bool bCopyToIncrement, typename TIncrement, typename TStride>
extern void Transpose(
   TIncrement * pIncrement,
   TStride * pStride,
   const size_t cDimensions,
   TransposeDimension * const aDim
) {
   // TODO: can we can have 0 dimensions here?
   EBM_ASSERT(1 <= cDimensions);

   TransposeDimension * pDimInit = aDim;
   const TransposeDimension * const pDimEnd = &aDim[cDimensions];
   size_t cBytesStride = sizeof(*pStride);
   size_t cSkipLevelInit = 1;
   size_t cSkip;
   if(!bCopyToIncrement) {
      cSkip = 0;
   }
   do {
      // it is possible to have just missing and unknown bins
      EBM_ASSERT(2 <= pDimInit->cBins);

      if(pDimInit->bDropFirst) {
         if(!bCopyToIncrement) {
            cSkip += cSkipLevelInit;
         }
      }

      TransposeDimension * const pDimTransposed = &aDim[pDimInit->iTranspose];
      pDimTransposed->cBytesStride = cBytesStride;

      const size_t cBinsReduced = pDimTransposed->cBins - (pDimTransposed->bDropFirst ? size_t { 1 } : size_t { 0 }) - (pDimTransposed->bDropLast ? size_t { 1 } : size_t { 0 });
      pDimTransposed->cBinsReduced = cBinsReduced;

      cBytesStride *= cBinsReduced;

      cSkipLevelInit *= pDimInit->cBins - (pDimInit->bDropFirst ? size_t { 1 } : size_t { 0 }) - (pDimInit->bDropLast ? size_t { 1 } : size_t { 0 });
      pDimInit->iBinsRemaining = pDimInit->cBins;

      ++pDimInit;
   } while(pDimEnd != pDimInit);

   // it should not be possible to eliminate all bins along a dimension unless there are zero samples in the
   // entire dataset, in which case we should not be reaching this function
   // TODO: maybe we should return an error if the caller was bad and specified a dimension with zero real cells
   EBM_ASSERT(0 != cSkipLevelInit);

   while(true) {
      if(bCopyToIncrement) {
         *pIncrement = static_cast<TIncrement>(*pStride);
      } else {
         if(0 == cSkip) {
            *pStride = static_cast<TStride>(*pIncrement);
         } else {
            // TODO: instead of using a counter, use a pointer that we compare to *pIncrement. No need to decrement it 
            // then and we can use a branchless comparison to update the pStrideSkipTo pointer when adding to the 
            // equivalent to cSkip which will be executed less than this decrement that happen each loop
            --cSkip;
         }
      }

      ++pIncrement;

      size_t cSkipLevel;
      if(!bCopyToIncrement) {
         cSkipLevel = 1;
      }
      TransposeDimension * pDim = aDim;
      while(true) {
         // TODO: instead of iBinsRemaining, use a pFirst and pLast on each dimension to keep track of our state
         // and to keep track of where we want to return to after we have finished the stripe
         size_t iBinsRemaining = pDim->iBinsRemaining;
         --iBinsRemaining;
         pDim->iBinsRemaining = iBinsRemaining;
         if(1 == iBinsRemaining) {
            // TODO: when we replace iBinsRemaining with a pointer, we can set that pointer to the last bin
            //       if bDropLast is true and then check the value of bDropLast to see if we need to progress one
            //       more cell to the real last one

            // we're moving into the last position
            if(!pDim->bDropLast) {
               if(2 != pDim->cBins || !pDim->bDropFirst) {
                  pStride = reinterpret_cast<TStride *>(reinterpret_cast<char *>(pStride) + pDim->cBytesStride);
               }
            } else {
               if(!bCopyToIncrement) {
                  cSkip += cSkipLevel;
               }
            }
            break;
         } else if(pDim->cBins - 1 == iBinsRemaining) {
            // TODO: when we replace iBinsRemaining with a pointer, we can probably fill that pointer with nullptr
            //       if bDropFirst is first when we wrap and then check for nullptr here and then fill it with the
            //       real value of the last or end based on bDropLast

            // we're moving away from the first position
            if(!pDim->bDropFirst) {
               pStride = reinterpret_cast<TStride *>(reinterpret_cast<char *>(pStride) + pDim->cBytesStride);
            }
            break;
         } else if(0 != iBinsRemaining) {
            pStride = reinterpret_cast<TStride *>(reinterpret_cast<char *>(pStride) + pDim->cBytesStride);
            break;
         }

         // we're moving into the first position again after wrapping this dimension

         if(!bCopyToIncrement) {
            if(pDim->bDropFirst) {
               cSkip += cSkipLevel;
            }
            cSkipLevel *= pDim->cBinsReduced;
         }

         // reset
         pDim->iBinsRemaining = pDim->cBins;

         const size_t cBinsStride = pDim->cBinsReduced - 1;
         pStride = reinterpret_cast<TStride *>(reinterpret_cast<char *>(pStride) - pDim->cBytesStride * cBinsStride);

         ++pDim;
         if(pDimEnd == pDim) {
            return;
         }
      }
   }
}

} // DEFINED_ZONE_NAME

#endif // TRANSPOSE_HPP
