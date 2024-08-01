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
      const bool bCountSrc,
      const bool bWeightSrc,
      const void* const aSrc,
      const UIntMain* const aCounts,
      const FloatPrecomp* const aWeights,
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
         cSrcBinBytes = GetBinSize<TFloatSpecific, TUIntSpecific>(bCountSrc, bWeightSrc, bHessian, cScores);
         if(bCountSrc) {
            if(bWeightSrc) {
               if(bHessian) {
                  typedef Bin<TFloatSpecific, TUIntSpecific, true, true, true> BinSpecific;

                  iSrcSamples =    offsetof    (   BinSpecific,     m_cSamples   );
                  iSrcWeight = offsetof(BinSpecific, m_weight);
                  iSrcArray = offsetof(BinSpecific, m_aGradientPairs);
                  cSrcArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
                  iSrcGradient = BinSpecific::k_offsetGrad;
                  iSrcHessian = BinSpecific::k_offsetHess;
               } else {
                  typedef Bin<TFloatSpecific, TUIntSpecific, true, true, false> BinSpecific;

                  iSrcSamples = offsetof(BinSpecific, m_cSamples);
                  iSrcWeight = offsetof(BinSpecific, m_weight);
                  iSrcArray = offsetof(BinSpecific, m_aGradientPairs);
                  cSrcArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
                  iSrcGradient = BinSpecific::k_offsetGrad;
               }
            } else {
               if(bHessian) {
                  typedef Bin<TFloatSpecific, TUIntSpecific, true, false, true> BinSpecific;

                  iSrcSamples = offsetof(BinSpecific, m_cSamples);
                  iSrcArray = offsetof(BinSpecific, m_aGradientPairs);
                  cSrcArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
                  iSrcGradient = BinSpecific::k_offsetGrad;
                  iSrcHessian = BinSpecific::k_offsetHess;
               } else {
                  typedef Bin<TFloatSpecific, TUIntSpecific, true, false, false> BinSpecific;

                  iSrcSamples = offsetof(BinSpecific, m_cSamples);
                  iSrcArray = offsetof(BinSpecific, m_aGradientPairs);
                  cSrcArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
                  iSrcGradient = BinSpecific::k_offsetGrad;
               }
            }
         } else {
            if(bWeightSrc) {
               if(bHessian) {
                  typedef Bin<TFloatSpecific, TUIntSpecific, false, true, true> BinSpecific;

                  iSrcWeight = offsetof(BinSpecific, m_weight);
                  iSrcArray = offsetof(BinSpecific, m_aGradientPairs);
                  cSrcArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
                  iSrcGradient = BinSpecific::k_offsetGrad;
                  iSrcHessian = BinSpecific::k_offsetHess;
               } else {
                  typedef Bin<TFloatSpecific, TUIntSpecific, false, true, false> BinSpecific;

                  iSrcWeight = offsetof(BinSpecific, m_weight);
                  iSrcArray = offsetof(BinSpecific, m_aGradientPairs);
                  cSrcArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
                  iSrcGradient = BinSpecific::k_offsetGrad;
               }
            } else {
               if(bHessian) {
                  typedef Bin<TFloatSpecific, TUIntSpecific, false, false, true> BinSpecific;

                  iSrcArray = offsetof(BinSpecific, m_aGradientPairs);
                  cSrcArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
                  iSrcGradient = BinSpecific::k_offsetGrad;
                  iSrcHessian = BinSpecific::k_offsetHess;
               } else {
                  typedef Bin<TFloatSpecific, TUIntSpecific, false, false, false> BinSpecific;

                  iSrcArray = offsetof(BinSpecific, m_aGradientPairs);
                  cSrcArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
                  iSrcGradient = BinSpecific::k_offsetGrad;
               }
            }
         }
      } else {
         typedef float TFloatSpecific;
         cSrcBinBytes = GetBinSize<TFloatSpecific, TUIntSpecific>(bCountSrc, bWeightSrc, bHessian, cScores);
         if(bCountSrc) {
            if(bWeightSrc) {
               if(bHessian) {
                  typedef Bin<TFloatSpecific, TUIntSpecific, true, true, true> BinSpecific;

                  iSrcSamples = offsetof(BinSpecific, m_cSamples);
                  iSrcWeight = offsetof(BinSpecific, m_weight);
                  iSrcArray = offsetof(BinSpecific, m_aGradientPairs);
                  cSrcArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
                  iSrcGradient = BinSpecific::k_offsetGrad;
                  iSrcHessian = BinSpecific::k_offsetHess;
               } else {
                  typedef Bin<TFloatSpecific, TUIntSpecific, true, true, false> BinSpecific;

                  iSrcSamples = offsetof(BinSpecific, m_cSamples);
                  iSrcWeight = offsetof(BinSpecific, m_weight);
                  iSrcArray = offsetof(BinSpecific, m_aGradientPairs);
                  cSrcArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
                  iSrcGradient = BinSpecific::k_offsetGrad;
               }
            } else {
               if(bHessian) {
                  typedef Bin<TFloatSpecific, TUIntSpecific, true, false, true> BinSpecific;

                  iSrcSamples = offsetof(BinSpecific, m_cSamples);
                  iSrcArray = offsetof(BinSpecific, m_aGradientPairs);
                  cSrcArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
                  iSrcGradient = BinSpecific::k_offsetGrad;
                  iSrcHessian = BinSpecific::k_offsetHess;
               } else {
                  typedef Bin<TFloatSpecific, TUIntSpecific, true, false, false> BinSpecific;

                  iSrcSamples = offsetof(BinSpecific, m_cSamples);
                  iSrcArray = offsetof(BinSpecific, m_aGradientPairs);
                  cSrcArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
                  iSrcGradient = BinSpecific::k_offsetGrad;
               }
            }
         } else {
            if(bWeightSrc) {
               if(bHessian) {
                  typedef Bin<TFloatSpecific, TUIntSpecific, false, true, true> BinSpecific;

                  iSrcWeight = offsetof(BinSpecific, m_weight);
                  iSrcArray = offsetof(BinSpecific, m_aGradientPairs);
                  cSrcArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
                  iSrcGradient = BinSpecific::k_offsetGrad;
                  iSrcHessian = BinSpecific::k_offsetHess;
               } else {
                  typedef Bin<TFloatSpecific, TUIntSpecific, false, true, false> BinSpecific;

                  iSrcWeight = offsetof(BinSpecific, m_weight);
                  iSrcArray = offsetof(BinSpecific, m_aGradientPairs);
                  cSrcArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
                  iSrcGradient = BinSpecific::k_offsetGrad;
               }
            } else {
               if(bHessian) {
                  typedef Bin<TFloatSpecific, TUIntSpecific, false, false, true> BinSpecific;

                  iSrcArray = offsetof(BinSpecific, m_aGradientPairs);
                  cSrcArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
                  iSrcGradient = BinSpecific::k_offsetGrad;
                  iSrcHessian = BinSpecific::k_offsetHess;
               } else {
                  typedef Bin<TFloatSpecific, TUIntSpecific, false, false, false> BinSpecific;

                  iSrcArray = offsetof(BinSpecific, m_aGradientPairs);
                  cSrcArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
                  iSrcGradient = BinSpecific::k_offsetGrad;
               }
            }
         }
      }
   } else {
      typedef uint32_t TUIntSpecific;
      if(bDoubleSrc) {
         typedef double TFloatSpecific;
         cSrcBinBytes = GetBinSize<TFloatSpecific, TUIntSpecific>(bCountSrc, bWeightSrc, bHessian, cScores);
         if(bCountSrc) {
            if(bWeightSrc) {
               if(bHessian) {
                  typedef Bin<TFloatSpecific, TUIntSpecific, true, true, true> BinSpecific;

                  iSrcSamples = offsetof(BinSpecific, m_cSamples);
                  iSrcWeight = offsetof(BinSpecific, m_weight);
                  iSrcArray = offsetof(BinSpecific, m_aGradientPairs);
                  cSrcArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
                  iSrcGradient = BinSpecific::k_offsetGrad;
                  iSrcHessian = BinSpecific::k_offsetHess;
               } else {
                  typedef Bin<TFloatSpecific, TUIntSpecific, true, true, false> BinSpecific;

                  iSrcSamples = offsetof(BinSpecific, m_cSamples);
                  iSrcWeight = offsetof(BinSpecific, m_weight);
                  iSrcArray = offsetof(BinSpecific, m_aGradientPairs);
                  cSrcArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
                  iSrcGradient = BinSpecific::k_offsetGrad;
               }
            } else {
               if(bHessian) {
                  typedef Bin<TFloatSpecific, TUIntSpecific, true, false, true> BinSpecific;

                  iSrcSamples = offsetof(BinSpecific, m_cSamples);
                  iSrcArray = offsetof(BinSpecific, m_aGradientPairs);
                  cSrcArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
                  iSrcGradient = BinSpecific::k_offsetGrad;
                  iSrcHessian = BinSpecific::k_offsetHess;
               } else {
                  typedef Bin<TFloatSpecific, TUIntSpecific, true, false, false> BinSpecific;

                  iSrcSamples = offsetof(BinSpecific, m_cSamples);
                  iSrcArray = offsetof(BinSpecific, m_aGradientPairs);
                  cSrcArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
                  iSrcGradient = BinSpecific::k_offsetGrad;
               }
            }
         } else {
            if(bWeightSrc) {
               if(bHessian) {
                  typedef Bin<TFloatSpecific, TUIntSpecific, false, true, true> BinSpecific;

                  iSrcWeight = offsetof(BinSpecific, m_weight);
                  iSrcArray = offsetof(BinSpecific, m_aGradientPairs);
                  cSrcArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
                  iSrcGradient = BinSpecific::k_offsetGrad;
                  iSrcHessian = BinSpecific::k_offsetHess;
               } else {
                  typedef Bin<TFloatSpecific, TUIntSpecific, false, true, false> BinSpecific;

                  iSrcWeight = offsetof(BinSpecific, m_weight);
                  iSrcArray = offsetof(BinSpecific, m_aGradientPairs);
                  cSrcArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
                  iSrcGradient = BinSpecific::k_offsetGrad;
               }
            } else {
               if(bHessian) {
                  typedef Bin<TFloatSpecific, TUIntSpecific, false, false, true> BinSpecific;

                  iSrcArray = offsetof(BinSpecific, m_aGradientPairs);
                  cSrcArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
                  iSrcGradient = BinSpecific::k_offsetGrad;
                  iSrcHessian = BinSpecific::k_offsetHess;
               } else {
                  typedef Bin<TFloatSpecific, TUIntSpecific, false, false, false> BinSpecific;

                  iSrcArray = offsetof(BinSpecific, m_aGradientPairs);
                  cSrcArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
                  iSrcGradient = BinSpecific::k_offsetGrad;
               }
            }
         }
      } else {
         typedef float TFloatSpecific;
         cSrcBinBytes = GetBinSize<TFloatSpecific, TUIntSpecific>(bCountSrc, bWeightSrc, bHessian, cScores);
         if(bCountSrc) {
            if(bWeightSrc) {
               if(bHessian) {
                  typedef Bin<TFloatSpecific, TUIntSpecific, true, true, true> BinSpecific;

                  iSrcSamples = offsetof(BinSpecific, m_cSamples);
                  iSrcWeight = offsetof(BinSpecific, m_weight);
                  iSrcArray = offsetof(BinSpecific, m_aGradientPairs);
                  cSrcArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
                  iSrcGradient = BinSpecific::k_offsetGrad;
                  iSrcHessian = BinSpecific::k_offsetHess;
               } else {
                  typedef Bin<TFloatSpecific, TUIntSpecific, true, true, false> BinSpecific;

                  iSrcSamples = offsetof(BinSpecific, m_cSamples);
                  iSrcWeight = offsetof(BinSpecific, m_weight);
                  iSrcArray = offsetof(BinSpecific, m_aGradientPairs);
                  cSrcArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
                  iSrcGradient = BinSpecific::k_offsetGrad;
               }
            } else {
               if(bHessian) {
                  typedef Bin<TFloatSpecific, TUIntSpecific, true, false, true> BinSpecific;

                  iSrcSamples = offsetof(BinSpecific, m_cSamples);
                  iSrcArray = offsetof(BinSpecific, m_aGradientPairs);
                  cSrcArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
                  iSrcGradient = BinSpecific::k_offsetGrad;
                  iSrcHessian = BinSpecific::k_offsetHess;
               } else {
                  typedef Bin<TFloatSpecific, TUIntSpecific, true, false, false> BinSpecific;

                  iSrcSamples = offsetof(BinSpecific, m_cSamples);
                  iSrcArray = offsetof(BinSpecific, m_aGradientPairs);
                  cSrcArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
                  iSrcGradient = BinSpecific::k_offsetGrad;
               }
            }
         } else {
            if(bWeightSrc) {
               if(bHessian) {
                  typedef Bin<TFloatSpecific, TUIntSpecific, false, true, true> BinSpecific;

                  iSrcWeight = offsetof(BinSpecific, m_weight);
                  iSrcArray = offsetof(BinSpecific, m_aGradientPairs);
                  cSrcArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
                  iSrcGradient = BinSpecific::k_offsetGrad;
                  iSrcHessian = BinSpecific::k_offsetHess;
               } else {
                  typedef Bin<TFloatSpecific, TUIntSpecific, false, true, false> BinSpecific;

                  iSrcWeight = offsetof(BinSpecific, m_weight);
                  iSrcArray = offsetof(BinSpecific, m_aGradientPairs);
                  cSrcArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
                  iSrcGradient = BinSpecific::k_offsetGrad;
               }
            } else {
               if(bHessian) {
                  typedef Bin<TFloatSpecific, TUIntSpecific, false, false, true> BinSpecific;

                  iSrcArray = offsetof(BinSpecific, m_aGradientPairs);
                  cSrcArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
                  iSrcGradient = BinSpecific::k_offsetGrad;
                  iSrcHessian = BinSpecific::k_offsetHess;
               } else {
                  typedef Bin<TFloatSpecific, TUIntSpecific, false, false, false> BinSpecific;

                  iSrcArray = offsetof(BinSpecific, m_aGradientPairs);
                  cSrcArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
                  iSrcGradient = BinSpecific::k_offsetGrad;
               }
            }
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
         cDestBinBytes = GetBinSize<TFloatSpecific, TUIntSpecific>(true, true, bHessian, cScores);
         if(bHessian) {
            typedef Bin<TFloatSpecific, TUIntSpecific, true, true, true> BinSpecific;

            iDestSamples = offsetof(BinSpecific, m_cSamples);
            iDestWeight = offsetof(BinSpecific, m_weight);
            iDestArray = offsetof(BinSpecific, m_aGradientPairs);
            cDestArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
            iDestGradient = BinSpecific::k_offsetGrad;
            iDestHessian = BinSpecific::k_offsetHess;
         } else {
            typedef Bin<TFloatSpecific, TUIntSpecific, true, true, false> BinSpecific;

            iDestSamples = offsetof(BinSpecific, m_cSamples);
            iDestWeight = offsetof(BinSpecific, m_weight);
            iDestArray = offsetof(BinSpecific, m_aGradientPairs);
            cDestArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
            iDestGradient = BinSpecific::k_offsetGrad;
         }
      } else {
         typedef float TFloatSpecific;
         cDestBinBytes = GetBinSize<TFloatSpecific, TUIntSpecific>(true, true, bHessian, cScores);
         if(bHessian) {
            typedef Bin<TFloatSpecific, TUIntSpecific, true, true, true> BinSpecific;

            iDestSamples = offsetof(BinSpecific, m_cSamples);
            iDestWeight = offsetof(BinSpecific, m_weight);
            iDestArray = offsetof(BinSpecific, m_aGradientPairs);
            cDestArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
            iDestGradient = BinSpecific::k_offsetGrad;
            iDestHessian = BinSpecific::k_offsetHess;
         } else {
            typedef Bin<TFloatSpecific, TUIntSpecific, true, true, false> BinSpecific;

            iDestSamples = offsetof(BinSpecific, m_cSamples);
            iDestWeight = offsetof(BinSpecific, m_weight);
            iDestArray = offsetof(BinSpecific, m_aGradientPairs);
            cDestArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
            iDestGradient = BinSpecific::k_offsetGrad;
         }
      }
   } else {
      typedef uint32_t TUIntSpecific;
      if(bDoubleDest) {
         typedef double TFloatSpecific;
         cDestBinBytes = GetBinSize<TFloatSpecific, TUIntSpecific>(true, true, bHessian, cScores);
         if(bHessian) {
            typedef Bin<TFloatSpecific, TUIntSpecific, true, true, true> BinSpecific;

            iDestSamples = offsetof(BinSpecific, m_cSamples);
            iDestWeight = offsetof(BinSpecific, m_weight);
            iDestArray = offsetof(BinSpecific, m_aGradientPairs);
            cDestArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
            iDestGradient = BinSpecific::k_offsetGrad;
            iDestHessian = BinSpecific::k_offsetHess;
         } else {
            typedef Bin<TFloatSpecific, TUIntSpecific, true, true, false> BinSpecific;

            iDestSamples = offsetof(BinSpecific, m_cSamples);
            iDestWeight = offsetof(BinSpecific, m_weight);
            iDestArray = offsetof(BinSpecific, m_aGradientPairs);
            cDestArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
            iDestGradient = BinSpecific::k_offsetGrad;
         }
      } else {
         typedef float TFloatSpecific;
         cDestBinBytes = GetBinSize<TFloatSpecific, TUIntSpecific>(true, true, bHessian, cScores);
         if(bHessian) {
            typedef Bin<TFloatSpecific, TUIntSpecific, true, true, true> BinSpecific;

            iDestSamples = offsetof(BinSpecific, m_cSamples);
            iDestWeight = offsetof(BinSpecific, m_weight);
            iDestArray = offsetof(BinSpecific, m_aGradientPairs);
            cDestArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
            iDestGradient = BinSpecific::k_offsetGrad;
            iDestHessian = BinSpecific::k_offsetHess;
         } else {
            typedef Bin<TFloatSpecific, TUIntSpecific, true, true, false> BinSpecific;

            iDestSamples = offsetof(BinSpecific, m_cSamples);
            iDestWeight = offsetof(BinSpecific, m_weight);
            iDestArray = offsetof(BinSpecific, m_aGradientPairs);
            cDestArrayItemBytes = sizeof(BinSpecific::m_aGradientPairs[0]);
            iDestGradient = BinSpecific::k_offsetGrad;
         }
      }
   }

   EBM_ASSERT(0 <= iDestSamples);
   EBM_ASSERT(0 <= iDestWeight);

   // TODO: we should move the application of aCounts and aWeights into a separate function
   // that gets called after this function. Since we have pre-computed final counts, we only
   // need to copy them into the final bins on the final merge, so if there are a lot of
   // bin merges like on a GPU then all of those don't need to care about aCounts and aWeights.
   const UIntMain* pCounts = aCounts;
   const FloatPrecomp* pWeights = aWeights;

   const unsigned char* pSrc = reinterpret_cast<const unsigned char*>(aSrc);
   const unsigned char* const pSrcEnd = pSrc + cSrcBinBytes * cBins;
   unsigned char* pAddDest = reinterpret_cast<unsigned char*>(aAddDest);
   const size_t cSrcArrayTotalBytes = cSrcArrayItemBytes * cScores;
   do {
      if(nullptr != pCounts) {
         EBM_ASSERT(iSrcSamples < 0);
         EBM_ASSERT(0 <= iDestSamples);
         const UIntMain src = *pCounts;
         ++pCounts;
         if(bUInt64Dest) {
            // we should only use aCounts on the last call to ConvertAddBin since it's added
            EBM_ASSERT(0 == *reinterpret_cast<uint64_t*>(pAddDest + iDestSamples));
            // assign instead of add since this should only be called this was on the last iteration
            *reinterpret_cast<uint64_t*>(pAddDest + iDestSamples) = static_cast<uint64_t>(src);
         } else {
            // we should only use aCounts on the last call to ConvertAddBin since it's added
            EBM_ASSERT(0 == *reinterpret_cast<uint32_t*>(pAddDest + iDestSamples));
            // assign instead of add since this should only be called this was on the last iteration
            *reinterpret_cast<uint32_t*>(pAddDest + iDestSamples) = static_cast<uint32_t>(src);
         }
      } else if(0 <= iSrcSamples) {
         EBM_ASSERT(0 <= iDestSamples);
         if(bUInt64Src) {
            const uint64_t src = *reinterpret_cast<const uint64_t*>(pSrc + iSrcSamples);
            if(bUInt64Dest) {
               *reinterpret_cast<uint64_t*>(pAddDest + iDestSamples) += static_cast<uint64_t>(src);
            } else {
               *reinterpret_cast<uint32_t*>(pAddDest + iDestSamples) += static_cast<uint32_t>(src);
            }
         } else {
            const uint32_t src = *reinterpret_cast<const uint32_t*>(pSrc + iSrcSamples);
            if(bUInt64Dest) {
               *reinterpret_cast<uint64_t*>(pAddDest + iDestSamples) += static_cast<uint64_t>(src);
            } else {
               *reinterpret_cast<uint32_t*>(pAddDest + iDestSamples) += static_cast<uint32_t>(src);
            }
         }
      }

      if(nullptr != pWeights) {
         EBM_ASSERT(iSrcWeight < 0);
         EBM_ASSERT(0 <= iDestWeight);
         const FloatPrecomp src = *pWeights;
         ++pWeights;
         if(bDoubleDest) {
            // we should only use aWeights on the last call to ConvertAddBin since it's added
            EBM_ASSERT(0 == *reinterpret_cast<double*>(pAddDest + iDestWeight));
            // assign instead of add since this should only be called this was on the last iteration
            *reinterpret_cast<double*>(pAddDest + iDestWeight) = static_cast<double>(src);
         } else {
            // we should only use aWeights on the last call to ConvertAddBin since it's added
            EBM_ASSERT(0 == *reinterpret_cast<float*>(pAddDest + iDestWeight));
            // assign instead of add since this should only be called this was on the last iteration
            *reinterpret_cast<float*>(pAddDest + iDestWeight) = static_cast<float>(src);
         }
      } else if(0 <= iSrcWeight) {
         EBM_ASSERT(0 <= iDestWeight);
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
                  EBM_ASSERT(0 <= iSrcHessian);
                  *reinterpret_cast<double*>(pDestArray + iDestHessian) +=
                        static_cast<double>(*reinterpret_cast<const double*>(pSrcArray + iSrcHessian));
               }
            } else {
               *reinterpret_cast<float*>(pDestArray + iDestGradient) += static_cast<float>(src);
               if(0 <= iDestHessian) {
                  EBM_ASSERT(0 <= iSrcHessian);
                  *reinterpret_cast<float*>(pDestArray + iDestHessian) +=
                        static_cast<float>(*reinterpret_cast<const double*>(pSrcArray + iSrcHessian));
               }
            }
         } else {
            const float src = *reinterpret_cast<const float*>(pSrcArray + iSrcGradient);
            if(bDoubleDest) {
               *reinterpret_cast<double*>(pDestArray + iDestGradient) += static_cast<double>(src);
               if(0 <= iDestHessian) {
                  EBM_ASSERT(0 <= iSrcHessian);
                  *reinterpret_cast<double*>(pDestArray + iDestHessian) +=
                        static_cast<double>(*reinterpret_cast<const float*>(pSrcArray + iSrcHessian));
               }
            } else {
               *reinterpret_cast<float*>(pDestArray + iDestGradient) += static_cast<float>(src);
               if(0 <= iDestHessian) {
                  EBM_ASSERT(0 <= iSrcHessian);
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
