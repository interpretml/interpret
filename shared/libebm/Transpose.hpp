// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef TRANSPOSE_HPP
#define TRANSPOSE_HPP

#include <stddef.h> // size_t, ptrdiff_t

#include "logging.h" // EBM_ASSERT

#include "Feature.hpp"
#include "Term.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

struct TransposeDimension {
   size_t cBins;
   bool bDropFirst;
   bool bDropLast;
   size_t cBinsReduced;
   size_t iBinsRemaining;

   size_t cBytesStride;
};

template<bool bCopyToIncrement, typename TIncrement, typename TStride>
extern void Transpose(const Term* const pTerm, const size_t cScores, TIncrement* pIncrement, TStride* pStride) {
   EBM_ASSERT(0 < cScores);
   const size_t cBytesPerCell = sizeof(*pStride) * cScores;

   if(nullptr == pTerm || size_t{0} == pTerm->GetCountDimensions()) {
      TIncrement* const pIncrementEnd = pIncrement + cScores;
      do {
         if(bCopyToIncrement) {
            *pIncrement = static_cast<TIncrement>(*pStride);
         } else {
            *pStride = static_cast<TStride>(*pIncrement);
         }
         ++pStride;
         ++pIncrement;
      } while(pIncrementEnd != pIncrement);

      return;
   }
   const size_t cDimensions = pTerm->GetCountDimensions();

   const TermFeature* const aTermFeatures = pTerm->GetTermFeatures();
   const TermFeature* pTermFeature = aTermFeatures;

   TransposeDimension aDim[k_cDimensionsMax];

   TransposeDimension* pDimInit = aDim;
   const TransposeDimension* const pDimEnd = &aDim[cDimensions];

   size_t cSkip;
   size_t cSkipLevelInit;
   if(!bCopyToIncrement) {
      cSkip = 0;
      cSkipLevelInit = 1;
   }
   do {
      // we process this in the order of pIncrement. The m_iTranspose of the TermFeature indicates where
      // we need to look to find the feature we are transposing to the location of the pIncrement dimension

      const FeatureBoosting* const pFeature = aTermFeatures[pTermFeature->m_iTranspose].m_pFeature;
      pDimInit->cBytesStride = aTermFeatures[pTermFeature->m_iTranspose].m_cStride * cBytesPerCell;

      const size_t cBinsReduced = pFeature->GetCountBins();
      EBM_ASSERT(1 <= cBinsReduced); // otherwise we should have exited in the caller
      bool bMissing = pFeature->IsMissing();
      bool bUnseen = pFeature->IsUnseen();
      const size_t cBins = cBinsReduced + (bMissing ? size_t{0} : size_t{1}) + (bUnseen ? size_t{0} : size_t{1});
      EBM_ASSERT(2 <= cBins); // just missing and unseen required

      pDimInit->cBins = cBins;
      pDimInit->bDropFirst = !bMissing;
      pDimInit->bDropLast = !bUnseen;

      pDimInit->cBinsReduced = cBinsReduced;
      pDimInit->iBinsRemaining = cBins;

      if(!bCopyToIncrement) {
         if(!bMissing) {
            cSkip += cSkipLevelInit;
         }
         cSkipLevelInit *= cBinsReduced;
      }

      ++pTermFeature;
      ++pDimInit;
   } while(pDimEnd != pDimInit);

   // it should not be possible to eliminate all bins along a dimension unless there are zero samples in the
   // entire dataset, in which case we should not be reaching this function
   // TODO: maybe we should return an error if the caller was bad and specified a dimension with zero real cells
   if(!bCopyToIncrement) {
      EBM_ASSERT(0 != cSkipLevelInit);
   }

   while(true) {
      TIncrement* const pIncrementEnd = pIncrement + cScores;

      size_t cSkipLevel;
      if(bCopyToIncrement) {
         TStride* pStrideTemp = pStride;
         do {
            *pIncrement = static_cast<TIncrement>(*pStrideTemp);
            ++pStrideTemp;
            ++pIncrement;
         } while(pIncrementEnd != pIncrement);
      } else {
         if(0 == cSkip) {
            TStride* pStrideTemp = pStride;
            do {
               *pStrideTemp = static_cast<TStride>(*pIncrement);
               ++pStrideTemp;
               ++pIncrement;
            } while(pIncrementEnd != pIncrement);
         } else {
            // TODO: instead of using a counter, use a pointer that we compare to *pIncrement. No need to decrement it
            // then and we can use a branchless comparison to update the pStrideSkipTo pointer when adding to the
            // equivalent to cSkip which will be executed less than this decrement that happen each loop
            --cSkip;
            pIncrement = pIncrementEnd;
         }
         cSkipLevel = 1;
      }

      TransposeDimension* pDim = aDim;
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
               // there is a corner case to handle where we're leaving the missing and entering the unseen bin
               if(1 != pDim->cBinsReduced) {
                  pStride = reinterpret_cast<TStride*>(reinterpret_cast<char*>(pStride) + pDim->cBytesStride);
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
               pStride = reinterpret_cast<TStride*>(reinterpret_cast<char*>(pStride) + pDim->cBytesStride);
            }
            break;
         } else if(0 != iBinsRemaining) {
            pStride = reinterpret_cast<TStride*>(reinterpret_cast<char*>(pStride) + pDim->cBytesStride);
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
         pStride = reinterpret_cast<TStride*>(reinterpret_cast<char*>(pStride) - pDim->cBytesStride * cBinsStride);

         ++pDim;
         if(pDimEnd == pDim) {
            return;
         }
      }
   }
}

} // namespace DEFINED_ZONE_NAME

#endif // TRANSPOSE_HPP
