// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stddef.h> // size_t, ptrdiff_t, offsetof

#include "logging.h" // EBM_ASSERT
#include "common_c.h" // FloatBig, FloatFast
#include "zones.h"

#include "Bin.hpp"

#include "ebm_internal.hpp" // SafeConvertFloat

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

extern void ConvertAddBin(
   const size_t cScores,
   const bool bHessian,
   const size_t cBins,
   const bool bDoubleDest,
   void * const aAddDest,
   const bool bDoubleSrc,
   const void * const aSrc
) {
   EBM_ASSERT(0 < cScores);
   EBM_ASSERT(0 < cBins);
   EBM_ASSERT(nullptr != aAddDest);
   EBM_ASSERT(nullptr != aSrc);

   size_t cDestBinBytes;
   ptrdiff_t iDestSamples = -1;
   ptrdiff_t iDestWeight = -1;
   ptrdiff_t iDestArray;
   size_t cDestArrayItemBytes;
   ptrdiff_t iDestGradient;
   ptrdiff_t iDestHessian = -1;

   if(bDoubleDest) {
      typedef double TFloatSpecific;
      cDestBinBytes = GetBinSize<TFloatSpecific>(bHessian, cScores);
      if(bHessian) {
         constexpr bool bHessianSpecific = true;
         typedef Bin<TFloatSpecific, bHessianSpecific> BinSpecific;

         iDestSamples = offsetof(BinSpecific, m_cSamples);
         iDestWeight = offsetof(BinSpecific, m_weight);
         iDestArray = offsetof(BinSpecific, m_aGradientPairs);
         cDestArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
         using GradientPairSpecific = typename std::remove_reference<decltype(BinSpecific::m_aGradientPairs[0])>::type;
         iDestGradient = offsetof(GradientPairSpecific, m_sumGradients);
         iDestHessian = offsetof(GradientPairSpecific, m_sumHessians);
      } else {
         constexpr bool bHessianSpecific = false;
         typedef Bin<TFloatSpecific, bHessianSpecific> BinSpecific;

         iDestSamples = offsetof(BinSpecific, m_cSamples);
         iDestWeight = offsetof(BinSpecific, m_weight);
         iDestArray = offsetof(BinSpecific, m_aGradientPairs);
         cDestArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
         using GradientPairSpecific = typename std::remove_reference<decltype(BinSpecific::m_aGradientPairs[0])>::type;
         iDestGradient = offsetof(GradientPairSpecific, m_sumGradients);
      }
   } else {
      typedef float TFloatSpecific;
      cDestBinBytes = GetBinSize<TFloatSpecific>(bHessian, cScores);
      if(bHessian) {
         constexpr bool bHessianSpecific = true;
         typedef Bin<TFloatSpecific, bHessianSpecific> BinSpecific;

         iDestSamples = offsetof(BinSpecific, m_cSamples);
         iDestWeight = offsetof(BinSpecific, m_weight);
         iDestArray = offsetof(BinSpecific, m_aGradientPairs);
         cDestArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
         using GradientPairSpecific = typename std::remove_reference<decltype(BinSpecific::m_aGradientPairs[0])>::type;
         iDestGradient = offsetof(GradientPairSpecific, m_sumGradients);
         iDestHessian = offsetof(GradientPairSpecific, m_sumHessians);
      } else {
         constexpr bool bHessianSpecific = false;
         typedef Bin<TFloatSpecific, bHessianSpecific> BinSpecific;

         iDestSamples = offsetof(BinSpecific, m_cSamples);
         iDestWeight = offsetof(BinSpecific, m_weight);
         iDestArray = offsetof(BinSpecific, m_aGradientPairs);
         cDestArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
         using GradientPairSpecific = typename std::remove_reference<decltype(BinSpecific::m_aGradientPairs[0])>::type;
         iDestGradient = offsetof(GradientPairSpecific, m_sumGradients);
      }
   }

   size_t cSrcBinBytes;
   ptrdiff_t iSrcSamples = -1;
   ptrdiff_t iSrcWeight = -1;
   ptrdiff_t iSrcArray;
   size_t cSrcArrayItemBytes;
   ptrdiff_t iSrcGradient;
   ptrdiff_t iSrcHessian = -1;

   if(bDoubleSrc) {
      typedef double TFloatSpecific;
      cSrcBinBytes = GetBinSize<TFloatSpecific>(bHessian, cScores);
      if(bHessian) {
         constexpr bool bHessianSpecific = true;
         typedef Bin<TFloatSpecific, bHessianSpecific> BinSpecific;

         iSrcSamples = offsetof(BinSpecific, m_cSamples);
         iSrcWeight = offsetof(BinSpecific, m_weight);
         iSrcArray = offsetof(BinSpecific, m_aGradientPairs);
         cSrcArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
         using GradientPairSpecific = typename std::remove_reference<decltype(BinSpecific::m_aGradientPairs[0])>::type;
         iSrcGradient = offsetof(GradientPairSpecific, m_sumGradients);
         iSrcHessian = offsetof(GradientPairSpecific, m_sumHessians);
      } else {
         constexpr bool bHessianSpecific = false;
         typedef Bin<TFloatSpecific, bHessianSpecific> BinSpecific;

         iSrcSamples = offsetof(BinSpecific, m_cSamples);
         iSrcWeight = offsetof(BinSpecific, m_weight);
         iSrcArray = offsetof(BinSpecific, m_aGradientPairs);
         cSrcArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
         using GradientPairSpecific = typename std::remove_reference<decltype(BinSpecific::m_aGradientPairs[0])>::type;
         iSrcGradient = offsetof(GradientPairSpecific, m_sumGradients);
      }
   } else {
      typedef float TFloatSpecific;
      cSrcBinBytes = GetBinSize<TFloatSpecific>(bHessian, cScores);
      if(bHessian) {
         constexpr bool bHessianSpecific = true;
         typedef Bin<TFloatSpecific, bHessianSpecific> BinSpecific;

         iSrcSamples = offsetof(BinSpecific, m_cSamples);
         iSrcWeight = offsetof(BinSpecific, m_weight);
         iSrcArray = offsetof(BinSpecific, m_aGradientPairs);
         cSrcArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
         using GradientPairSpecific = typename std::remove_reference<decltype(BinSpecific::m_aGradientPairs[0])>::type;
         iSrcGradient = offsetof(GradientPairSpecific, m_sumGradients);
         iSrcHessian = offsetof(GradientPairSpecific, m_sumHessians);
      } else {
         constexpr bool bHessianSpecific = false;
         typedef Bin<TFloatSpecific, bHessianSpecific> BinSpecific;

         iSrcSamples = offsetof(BinSpecific, m_cSamples);
         iSrcWeight = offsetof(BinSpecific, m_weight);
         iSrcArray = offsetof(BinSpecific, m_aGradientPairs);
         cSrcArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
         using GradientPairSpecific = typename std::remove_reference<decltype(BinSpecific::m_aGradientPairs[0])>::type;
         iSrcGradient = offsetof(GradientPairSpecific, m_sumGradients);
      }
   }

   EBM_ASSERT(0 <= iDestSamples);
   EBM_ASSERT(0 <= iSrcSamples);

   EBM_ASSERT(0 <= iDestWeight);
   EBM_ASSERT(0 <= iSrcWeight);

   EBM_ASSERT(0 <= iDestHessian && 0 <= iSrcHessian || iDestHessian < 0 && iSrcHessian < 0);

   unsigned char * pAddDest = reinterpret_cast<unsigned char *>(aAddDest);
   const unsigned char * pSrc = reinterpret_cast<const unsigned char *>(aSrc);
   const unsigned char * const pSrcEnd = pSrc + cSrcBinBytes * cBins;
   const size_t cSrcArrayTotalBytes = cSrcArrayItemBytes * cScores;
   do {
      *reinterpret_cast<size_t *>(pAddDest + iDestSamples) += *reinterpret_cast<const size_t *>(pSrc + iSrcSamples);

      if(bDoubleSrc) {
         const double src = *reinterpret_cast<const double *>(pSrc + iSrcWeight);
         if(bDoubleDest) {
            *reinterpret_cast<double *>(pAddDest + iDestWeight) += SafeConvertFloat<double>(src);
         } else {
            *reinterpret_cast<float *>(pAddDest + iDestWeight) += SafeConvertFloat<float>(src);
         }
      } else {
         const float src = *reinterpret_cast<const float *>(pSrc + iSrcWeight);
         if(bDoubleDest) {
            *reinterpret_cast<double *>(pAddDest + iDestWeight) += SafeConvertFloat<double>(src);
         } else {
            *reinterpret_cast<float *>(pAddDest + iDestWeight) += SafeConvertFloat<float>(src);
         }
      }

      unsigned char * pDestArray = pAddDest + iDestArray;
      const unsigned char * pSrcArray = pSrc + iSrcArray;
      const unsigned char * pSrcArrayEnd = pSrcArray + cSrcArrayTotalBytes;
      do {
         if(bDoubleSrc) {
            const double src = *reinterpret_cast<const double *>(pSrcArray + iSrcGradient);
            if(bDoubleDest) {
               *reinterpret_cast<double *>(pDestArray + iDestGradient) += SafeConvertFloat<double>(src);
               if(0 <= iDestHessian) {
                  *reinterpret_cast<double *>(pDestArray + iDestHessian) += SafeConvertFloat<double>(*reinterpret_cast<const double *>(pSrcArray + iSrcHessian));
               }
            } else {
               *reinterpret_cast<float *>(pDestArray + iDestGradient) += SafeConvertFloat<float>(src);
               if(0 <= iDestHessian) {
                  *reinterpret_cast<float *>(pDestArray + iDestHessian) += SafeConvertFloat<float>(*reinterpret_cast<const double *>(pSrcArray + iSrcHessian));
               }
            }
         } else {
            const float src = *reinterpret_cast<const float *>(pSrcArray + iSrcGradient);
            if(bDoubleDest) {
               *reinterpret_cast<double *>(pDestArray + iDestGradient) += SafeConvertFloat<double>(src);
               if(0 <= iDestHessian) {
                  *reinterpret_cast<double *>(pDestArray + iDestHessian) += SafeConvertFloat<double>(*reinterpret_cast<const float *>(pSrcArray + iSrcHessian));
               }
            } else {
               *reinterpret_cast<float *>(pDestArray + iDestGradient) += SafeConvertFloat<float>(src);
               if(0 <= iDestHessian) {
                  *reinterpret_cast<float *>(pDestArray + iDestHessian) += SafeConvertFloat<float>(*reinterpret_cast<const float *>(pSrcArray + iSrcHessian));
               }
            }
         }
         pDestArray += cDestArrayItemBytes;
         pSrcArray += cSrcArrayItemBytes;
      } while(pSrcArrayEnd != pSrcArray);

      pAddDest += cDestBinBytes;
      pSrc += cSrcBinBytes;
   } while(pSrcEnd != pSrc);
}

} // DEFINED_ZONE_NAME
