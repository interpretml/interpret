// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "pch.hpp"

#include <stddef.h> // size_t, ptrdiff_t, offsetof

#include "logging.h" // EBM_ASSERT
#include "unzoned.h"

#define ZONE_main
#include "zones.h"

#include "Bin.hpp"

#include "ebm_internal.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

extern void ConvertAddBin(const size_t cScores,
      const bool bHessian,
      const size_t cBins,
      const bool bUInt64Src,
      const bool bDoubleSrc,
      const void* const aSrc,
      const bool bUInt64Dest,
      const bool bDoubleDest,
      void* const aAddDest) {
   EBM_ASSERT(0 < cScores);
   EBM_ASSERT(0 < cBins);
   EBM_ASSERT(nullptr != aSrc);
   EBM_ASSERT(nullptr != aAddDest);

   size_t cSrcBinBytes;
   ptrdiff_t iSrcSamples = -1;
   ptrdiff_t iSrcWeight = -1;
   ptrdiff_t iSrcArray;
   size_t cSrcArrayItemBytes;
   ptrdiff_t iSrcGradient;
   ptrdiff_t iSrcHessian = -1;

   if(bUInt64Src) {
      typedef uint64_t TUIntSpecific;
      if(bDoubleSrc) {
         typedef double TFloatSpecific;
         cSrcBinBytes = GetBinSize<TFloatSpecific, TUIntSpecific>(bHessian, cScores);
         if(bHessian) {
            constexpr bool bHessianSpecific = true;
            typedef Bin<TFloatSpecific, TUIntSpecific, bHessianSpecific> BinSpecific;

            iSrcSamples = offsetof(BinSpecific, m_cSamples);
            iSrcWeight = offsetof(BinSpecific, m_weight);
            iSrcArray = offsetof(BinSpecific, m_aGradientPairs);
            cSrcArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
            using GradientPairSpecific =
                  typename std::remove_reference<decltype(BinSpecific::m_aGradientPairs[0])>::type;
            iSrcGradient = offsetof(GradientPairSpecific, m_sumGradients);
            iSrcHessian = offsetof(GradientPairSpecific, m_sumHessians);
         } else {
            constexpr bool bHessianSpecific = false;
            typedef Bin<TFloatSpecific, TUIntSpecific, bHessianSpecific> BinSpecific;

            iSrcSamples = offsetof(BinSpecific, m_cSamples);
            iSrcWeight = offsetof(BinSpecific, m_weight);
            iSrcArray = offsetof(BinSpecific, m_aGradientPairs);
            cSrcArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
            using GradientPairSpecific =
                  typename std::remove_reference<decltype(BinSpecific::m_aGradientPairs[0])>::type;
            iSrcGradient = offsetof(GradientPairSpecific, m_sumGradients);
         }
      } else {
         typedef float TFloatSpecific;
         cSrcBinBytes = GetBinSize<TFloatSpecific, TUIntSpecific>(bHessian, cScores);
         if(bHessian) {
            constexpr bool bHessianSpecific = true;
            typedef Bin<TFloatSpecific, TUIntSpecific, bHessianSpecific> BinSpecific;

            iSrcSamples = offsetof(BinSpecific, m_cSamples);
            iSrcWeight = offsetof(BinSpecific, m_weight);
            iSrcArray = offsetof(BinSpecific, m_aGradientPairs);
            cSrcArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
            using GradientPairSpecific =
                  typename std::remove_reference<decltype(BinSpecific::m_aGradientPairs[0])>::type;
            iSrcGradient = offsetof(GradientPairSpecific, m_sumGradients);
            iSrcHessian = offsetof(GradientPairSpecific, m_sumHessians);
         } else {
            constexpr bool bHessianSpecific = false;
            typedef Bin<TFloatSpecific, TUIntSpecific, bHessianSpecific> BinSpecific;

            iSrcSamples = offsetof(BinSpecific, m_cSamples);
            iSrcWeight = offsetof(BinSpecific, m_weight);
            iSrcArray = offsetof(BinSpecific, m_aGradientPairs);
            cSrcArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
            using GradientPairSpecific =
                  typename std::remove_reference<decltype(BinSpecific::m_aGradientPairs[0])>::type;
            iSrcGradient = offsetof(GradientPairSpecific, m_sumGradients);
         }
      }
   } else {
      typedef uint32_t TUIntSpecific;
      if(bDoubleSrc) {
         typedef double TFloatSpecific;
         cSrcBinBytes = GetBinSize<TFloatSpecific, TUIntSpecific>(bHessian, cScores);
         if(bHessian) {
            constexpr bool bHessianSpecific = true;
            typedef Bin<TFloatSpecific, TUIntSpecific, bHessianSpecific> BinSpecific;

            iSrcSamples = offsetof(BinSpecific, m_cSamples);
            iSrcWeight = offsetof(BinSpecific, m_weight);
            iSrcArray = offsetof(BinSpecific, m_aGradientPairs);
            cSrcArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
            using GradientPairSpecific =
                  typename std::remove_reference<decltype(BinSpecific::m_aGradientPairs[0])>::type;
            iSrcGradient = offsetof(GradientPairSpecific, m_sumGradients);
            iSrcHessian = offsetof(GradientPairSpecific, m_sumHessians);
         } else {
            constexpr bool bHessianSpecific = false;
            typedef Bin<TFloatSpecific, TUIntSpecific, bHessianSpecific> BinSpecific;

            iSrcSamples = offsetof(BinSpecific, m_cSamples);
            iSrcWeight = offsetof(BinSpecific, m_weight);
            iSrcArray = offsetof(BinSpecific, m_aGradientPairs);
            cSrcArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
            using GradientPairSpecific =
                  typename std::remove_reference<decltype(BinSpecific::m_aGradientPairs[0])>::type;
            iSrcGradient = offsetof(GradientPairSpecific, m_sumGradients);
         }
      } else {
         typedef float TFloatSpecific;
         cSrcBinBytes = GetBinSize<TFloatSpecific, TUIntSpecific>(bHessian, cScores);
         if(bHessian) {
            constexpr bool bHessianSpecific = true;
            typedef Bin<TFloatSpecific, TUIntSpecific, bHessianSpecific> BinSpecific;

            iSrcSamples = offsetof(BinSpecific, m_cSamples);
            iSrcWeight = offsetof(BinSpecific, m_weight);
            iSrcArray = offsetof(BinSpecific, m_aGradientPairs);
            cSrcArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
            using GradientPairSpecific =
                  typename std::remove_reference<decltype(BinSpecific::m_aGradientPairs[0])>::type;
            iSrcGradient = offsetof(GradientPairSpecific, m_sumGradients);
            iSrcHessian = offsetof(GradientPairSpecific, m_sumHessians);
         } else {
            constexpr bool bHessianSpecific = false;
            typedef Bin<TFloatSpecific, TUIntSpecific, bHessianSpecific> BinSpecific;

            iSrcSamples = offsetof(BinSpecific, m_cSamples);
            iSrcWeight = offsetof(BinSpecific, m_weight);
            iSrcArray = offsetof(BinSpecific, m_aGradientPairs);
            cSrcArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
            using GradientPairSpecific =
                  typename std::remove_reference<decltype(BinSpecific::m_aGradientPairs[0])>::type;
            iSrcGradient = offsetof(GradientPairSpecific, m_sumGradients);
         }
      }
   }

   size_t cDestBinBytes;
   ptrdiff_t iDestSamples = -1;
   ptrdiff_t iDestWeight = -1;
   ptrdiff_t iDestArray;
   size_t cDestArrayItemBytes;
   ptrdiff_t iDestGradient;
   ptrdiff_t iDestHessian = -1;

   if(bUInt64Dest) {
      typedef uint64_t TUIntSpecific;
      if(bDoubleDest) {
         typedef double TFloatSpecific;
         cDestBinBytes = GetBinSize<TFloatSpecific, TUIntSpecific>(bHessian, cScores);
         if(bHessian) {
            constexpr bool bHessianSpecific = true;
            typedef Bin<TFloatSpecific, TUIntSpecific, bHessianSpecific> BinSpecific;

            iDestSamples = offsetof(BinSpecific, m_cSamples);
            iDestWeight = offsetof(BinSpecific, m_weight);
            iDestArray = offsetof(BinSpecific, m_aGradientPairs);
            cDestArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
            using GradientPairSpecific =
                  typename std::remove_reference<decltype(BinSpecific::m_aGradientPairs[0])>::type;
            iDestGradient = offsetof(GradientPairSpecific, m_sumGradients);
            iDestHessian = offsetof(GradientPairSpecific, m_sumHessians);
         } else {
            constexpr bool bHessianSpecific = false;
            typedef Bin<TFloatSpecific, TUIntSpecific, bHessianSpecific> BinSpecific;

            iDestSamples = offsetof(BinSpecific, m_cSamples);
            iDestWeight = offsetof(BinSpecific, m_weight);
            iDestArray = offsetof(BinSpecific, m_aGradientPairs);
            cDestArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
            using GradientPairSpecific =
                  typename std::remove_reference<decltype(BinSpecific::m_aGradientPairs[0])>::type;
            iDestGradient = offsetof(GradientPairSpecific, m_sumGradients);
         }
      } else {
         typedef float TFloatSpecific;
         cDestBinBytes = GetBinSize<TFloatSpecific, TUIntSpecific>(bHessian, cScores);
         if(bHessian) {
            constexpr bool bHessianSpecific = true;
            typedef Bin<TFloatSpecific, TUIntSpecific, bHessianSpecific> BinSpecific;

            iDestSamples = offsetof(BinSpecific, m_cSamples);
            iDestWeight = offsetof(BinSpecific, m_weight);
            iDestArray = offsetof(BinSpecific, m_aGradientPairs);
            cDestArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
            using GradientPairSpecific =
                  typename std::remove_reference<decltype(BinSpecific::m_aGradientPairs[0])>::type;
            iDestGradient = offsetof(GradientPairSpecific, m_sumGradients);
            iDestHessian = offsetof(GradientPairSpecific, m_sumHessians);
         } else {
            constexpr bool bHessianSpecific = false;
            typedef Bin<TFloatSpecific, TUIntSpecific, bHessianSpecific> BinSpecific;

            iDestSamples = offsetof(BinSpecific, m_cSamples);
            iDestWeight = offsetof(BinSpecific, m_weight);
            iDestArray = offsetof(BinSpecific, m_aGradientPairs);
            cDestArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
            using GradientPairSpecific =
                  typename std::remove_reference<decltype(BinSpecific::m_aGradientPairs[0])>::type;
            iDestGradient = offsetof(GradientPairSpecific, m_sumGradients);
         }
      }
   } else {
      typedef uint32_t TUIntSpecific;
      if(bDoubleDest) {
         typedef double TFloatSpecific;
         cDestBinBytes = GetBinSize<TFloatSpecific, TUIntSpecific>(bHessian, cScores);
         if(bHessian) {
            constexpr bool bHessianSpecific = true;
            typedef Bin<TFloatSpecific, TUIntSpecific, bHessianSpecific> BinSpecific;

            iDestSamples = offsetof(BinSpecific, m_cSamples);
            iDestWeight = offsetof(BinSpecific, m_weight);
            iDestArray = offsetof(BinSpecific, m_aGradientPairs);
            cDestArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
            using GradientPairSpecific =
                  typename std::remove_reference<decltype(BinSpecific::m_aGradientPairs[0])>::type;
            iDestGradient = offsetof(GradientPairSpecific, m_sumGradients);
            iDestHessian = offsetof(GradientPairSpecific, m_sumHessians);
         } else {
            constexpr bool bHessianSpecific = false;
            typedef Bin<TFloatSpecific, TUIntSpecific, bHessianSpecific> BinSpecific;

            iDestSamples = offsetof(BinSpecific, m_cSamples);
            iDestWeight = offsetof(BinSpecific, m_weight);
            iDestArray = offsetof(BinSpecific, m_aGradientPairs);
            cDestArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
            using GradientPairSpecific =
                  typename std::remove_reference<decltype(BinSpecific::m_aGradientPairs[0])>::type;
            iDestGradient = offsetof(GradientPairSpecific, m_sumGradients);
         }
      } else {
         typedef float TFloatSpecific;
         cDestBinBytes = GetBinSize<TFloatSpecific, TUIntSpecific>(bHessian, cScores);
         if(bHessian) {
            constexpr bool bHessianSpecific = true;
            typedef Bin<TFloatSpecific, TUIntSpecific, bHessianSpecific> BinSpecific;

            iDestSamples = offsetof(BinSpecific, m_cSamples);
            iDestWeight = offsetof(BinSpecific, m_weight);
            iDestArray = offsetof(BinSpecific, m_aGradientPairs);
            cDestArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
            using GradientPairSpecific =
                  typename std::remove_reference<decltype(BinSpecific::m_aGradientPairs[0])>::type;
            iDestGradient = offsetof(GradientPairSpecific, m_sumGradients);
            iDestHessian = offsetof(GradientPairSpecific, m_sumHessians);
         } else {
            constexpr bool bHessianSpecific = false;
            typedef Bin<TFloatSpecific, TUIntSpecific, bHessianSpecific> BinSpecific;

            iDestSamples = offsetof(BinSpecific, m_cSamples);
            iDestWeight = offsetof(BinSpecific, m_weight);
            iDestArray = offsetof(BinSpecific, m_aGradientPairs);
            cDestArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
            using GradientPairSpecific =
                  typename std::remove_reference<decltype(BinSpecific::m_aGradientPairs[0])>::type;
            iDestGradient = offsetof(GradientPairSpecific, m_sumGradients);
         }
      }
   }

   EBM_ASSERT(0 <= iSrcSamples);
   EBM_ASSERT(0 <= iDestSamples);

   EBM_ASSERT(0 <= iSrcWeight);
   EBM_ASSERT(0 <= iDestWeight);

   EBM_ASSERT(0 <= iSrcHessian && 0 <= iDestHessian || iSrcHessian < 0 && iDestHessian < 0);

   const unsigned char* pSrc = reinterpret_cast<const unsigned char*>(aSrc);
   const unsigned char* const pSrcEnd = pSrc + cSrcBinBytes * cBins;
   unsigned char* pAddDest = reinterpret_cast<unsigned char*>(aAddDest);
   const size_t cSrcArrayTotalBytes = cSrcArrayItemBytes * cScores;
   do {
      if(bUInt64Src) {
         const uint64_t src = *reinterpret_cast<const uint64_t*>(pSrc + iSrcSamples);
         if(bUInt64Dest) {
            *reinterpret_cast<uint64_t*>(pAddDest + iDestSamples) += src;
         } else {
            *reinterpret_cast<uint32_t*>(pAddDest + iDestSamples) += static_cast<uint32_t>(src);
         }
      } else {
         const uint32_t src = *reinterpret_cast<const uint32_t*>(pSrc + iSrcSamples);
         if(bUInt64Dest) {
            *reinterpret_cast<uint64_t*>(pAddDest + iDestSamples) += src;
         } else {
            *reinterpret_cast<uint32_t*>(pAddDest + iDestSamples) += src;
         }
      }

      if(bDoubleSrc) {
         const double src = *reinterpret_cast<const double*>(pSrc + iSrcWeight);
         if(bDoubleDest) {
            *reinterpret_cast<double*>(pAddDest + iDestWeight) += static_cast<double>(src);
         } else {
            *reinterpret_cast<float*>(pAddDest + iDestWeight) += static_cast<float>(src);
         }
      } else {
         const float src = *reinterpret_cast<const float*>(pSrc + iSrcWeight);
         if(bDoubleDest) {
            *reinterpret_cast<double*>(pAddDest + iDestWeight) += static_cast<double>(src);
         } else {
            *reinterpret_cast<float*>(pAddDest + iDestWeight) += static_cast<float>(src);
         }
      }

      const unsigned char* pSrcArray = pSrc + iSrcArray;
      const unsigned char* pSrcArrayEnd = pSrcArray + cSrcArrayTotalBytes;
      unsigned char* pDestArray = pAddDest + iDestArray;
      do {
         if(bDoubleSrc) {
            const double src = *reinterpret_cast<const double*>(pSrcArray + iSrcGradient);
            if(bDoubleDest) {
               *reinterpret_cast<double*>(pDestArray + iDestGradient) += static_cast<double>(src);
               if(0 <= iDestHessian) {
                  *reinterpret_cast<double*>(pDestArray + iDestHessian) +=
                        static_cast<double>(*reinterpret_cast<const double*>(pSrcArray + iSrcHessian));
               }
            } else {
               *reinterpret_cast<float*>(pDestArray + iDestGradient) += static_cast<float>(src);
               if(0 <= iDestHessian) {
                  *reinterpret_cast<float*>(pDestArray + iDestHessian) +=
                        static_cast<float>(*reinterpret_cast<const double*>(pSrcArray + iSrcHessian));
               }
            }
         } else {
            const float src = *reinterpret_cast<const float*>(pSrcArray + iSrcGradient);
            if(bDoubleDest) {
               *reinterpret_cast<double*>(pDestArray + iDestGradient) += static_cast<double>(src);
               if(0 <= iDestHessian) {
                  *reinterpret_cast<double*>(pDestArray + iDestHessian) +=
                        static_cast<double>(*reinterpret_cast<const float*>(pSrcArray + iSrcHessian));
               }
            } else {
               *reinterpret_cast<float*>(pDestArray + iDestGradient) += static_cast<float>(src);
               if(0 <= iDestHessian) {
                  *reinterpret_cast<float*>(pDestArray + iDestHessian) +=
                        static_cast<float>(*reinterpret_cast<const float*>(pSrcArray + iSrcHessian));
               }
            }
         }
         pSrcArray += cSrcArrayItemBytes;
         pDestArray += cDestArrayItemBytes;
      } while(pSrcArrayEnd != pSrcArray);

      pSrc += cSrcBinBytes;
      pAddDest += cDestBinBytes;
   } while(pSrcEnd != pSrc);
}

} // namespace DEFINED_ZONE_NAME
