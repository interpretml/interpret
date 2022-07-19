// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>


#include "precompiled_header_cpp.hpp"

#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "ebm_internal.hpp"

#include "data_set_shared.hpp"
#include "ebm_stats.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

template<ptrdiff_t cCompilerClasses>
class InitializeGradientsAndHessiansInternal final {
public:

   InitializeGradientsAndHessiansInternal() = delete; // this is a static class.  Do not construct

   static ErrorEbmType Func(
      const ptrdiff_t cRuntimeClasses,
      const BagEbmType direction,
      const BagEbmType * const aBag,
      const void * const aTargets,
      const double * const aInitScores,
      const size_t cSetSamples,
      FloatFast * const aGradientAndHessian
   ) {
      static_assert(IsClassification(cCompilerClasses), "must be classification");
      static_assert(!IsBinaryClassification(cCompilerClasses), "must be multiclass");

      LOG_0(TraceLevelInfo, "Entered InitializeGradientsAndHessians");

      EBM_ASSERT(BagEbmType { -1 } == direction || BagEbmType { 1 } == direction);
      EBM_ASSERT(0 < cSetSamples);
      EBM_ASSERT(nullptr != aTargets);
      EBM_ASSERT(nullptr != aGradientAndHessian);

      const ptrdiff_t cClasses = GET_COUNT_CLASSES(
         cCompilerClasses,
         cRuntimeClasses
      );
      const size_t cScores = GetCountScores(cClasses);

      // TODO: we could eliminate this memory by doing the calculation twice below and then this code could return on error value (or would that work with the loss function stuff?)
      FloatFast * const aExps = EbmMalloc<FloatFast>(cScores);
      if(UNLIKELY(nullptr == aExps)) {
         LOG_0(TraceLevelWarning, "WARNING InitializeGradientsAndHessians nullptr == aExps");
         return Error_OutOfMemory;
      }

      const BagEbmType * pBag = aBag;
      const SharedStorageDataType * pTargetData = static_cast<const SharedStorageDataType *>(aTargets);
      const double * pInitScore = aInitScores;
      FloatFast * pGradientAndHessian = aGradientAndHessian;
      const FloatFast * const pGradientAndHessianEnd = aGradientAndHessian + cScores * cSetSamples * 2;
      const bool isLoopTraining = BagEbmType { 0 } < direction;
      do {
         BagEbmType countBagged = 1;
         if(nullptr != pBag) {
            countBagged = *pBag;
            ++pBag;
         }
         if(BagEbmType { 0 } != countBagged) {
            const bool isItemTraining = BagEbmType { 0 } < countBagged;
            if(isLoopTraining == isItemTraining) {
               const SharedStorageDataType targetOriginal = *pTargetData;
               // if we can't fit it, then we should increase our StorageDataType size!
               EBM_ASSERT(!IsConvertError<size_t>(targetOriginal));
               const size_t target = static_cast<size_t>(targetOriginal);
               EBM_ASSERT(target < static_cast<size_t>(cClasses));
               FloatFast * pExp = aExps;

               FloatFast sumExp = 0;

#ifdef ZERO_FIRST_MULTICLASS_LOGIT
               FloatFast zeroLogit = 0;
#endif // ZERO_FIRST_MULTICLASS_LOGIT

               size_t iScore = 0;
               do {
                  FloatFast initScore = 0;
                  if(nullptr != pInitScore) {
                     initScore = SafeConvertFloat<FloatFast>(*pInitScore);
                     ++pInitScore;
                  }
#ifdef ZERO_FIRST_MULTICLASS_LOGIT
                  if(IsMulticlass(cCompilerClasses)) {
                     if(size_t { 0 } == iScore) {
                        zeroLogit = initScore;
                     }
                     initScore -= zeroLogit;
                  }
#endif // ZERO_FIRST_MULTICLASS_LOGIT

                  const FloatFast oneExp = ExpForMulticlass<false>(initScore);
                  *pExp = oneExp;
                  ++pExp;
                  sumExp += oneExp;
                  ++iScore;
               } while(iScore < cScores);

               do {
                  // go back to the start so that we can iterate again
                  pExp -= cScores;

                  iScore = 0;
                  do {
                     FloatFast gradient;
                     FloatFast hessian;
                     EbmStats::InverseLinkFunctionThenCalculateGradientAndHessianMulticlass(sumExp, *pExp, target, iScore, gradient, hessian);
                     ++pExp;

                     EBM_ASSERT(pGradientAndHessian < pGradientAndHessianEnd - 1);

                     *pGradientAndHessian = gradient;
                     *(pGradientAndHessian + 1) = hessian;
                     pGradientAndHessian += 2;
                     ++iScore;
                  } while(iScore < cScores);

                  countBagged -= direction;
               } while(BagEbmType { 0 } != countBagged);
            } else {
               if(nullptr != pInitScore) {
                  pInitScore += cScores;
               }
            }
         }
         ++pTargetData;
      } while(pGradientAndHessianEnd != pGradientAndHessian);

      free(aExps);

      LOG_0(TraceLevelInfo, "Exited InitializeGradientsAndHessians");
      return Error_None;
   }
};

#ifndef EXPAND_BINARY_LOGITS
template<>
class InitializeGradientsAndHessiansInternal<2> final {
public:

   InitializeGradientsAndHessiansInternal() = delete; // this is a static class.  Do not construct

   static ErrorEbmType Func(
      const ptrdiff_t cRuntimeClasses,
      const BagEbmType direction,
      const BagEbmType * const aBag,
      const void * const aTargets,
      const double * const aInitScores,
      const size_t cSetSamples,
      FloatFast * const aGradientAndHessian
   ) {
      UNUSED(cRuntimeClasses);
      LOG_0(TraceLevelInfo, "Entered InitializeGradientsAndHessians");

      // TODO : review this function to see if iZeroLogit was set to a valid index, does that affect the number of items in pInitScore (I assume so), 
      //   and does it affect any calculations below like sumExp += std::exp(initScore) and the equivalent.  Should we use cScores or 
      //   cRuntimeClasses for some of the addition
      // TODO : !!! re-examine the idea of zeroing one of the logits with iZeroLogit after we have the ability to test large numbers of datasets

      EBM_ASSERT(BagEbmType { -1 } == direction || BagEbmType { 1 } == direction);
      EBM_ASSERT(0 < cSetSamples);
      EBM_ASSERT(nullptr != aTargets);
      EBM_ASSERT(nullptr != aGradientAndHessian);

      const BagEbmType * pBag = aBag;
      const SharedStorageDataType * pTargetData = static_cast<const SharedStorageDataType *>(aTargets);
      const double * pInitScore = aInitScores;
      FloatFast * pGradientAndHessian = aGradientAndHessian;
      const FloatFast * const pGradientAndHessianEnd = aGradientAndHessian + cSetSamples * 2;
      const bool isLoopTraining = BagEbmType { 0 } < direction;
      do {
         BagEbmType countBagged = 1;
         if(nullptr != pBag) {
            countBagged = *pBag;
            ++pBag;
         }
         if(BagEbmType { 0 } != countBagged) {
            FloatFast initScore = 0;
            if(nullptr != pInitScore) {
               initScore = SafeConvertFloat<FloatFast>(*pInitScore);
               ++pInitScore;
            }
            const bool isItemTraining = BagEbmType { 0 } < countBagged;
            if(isLoopTraining == isItemTraining) {
               const SharedStorageDataType targetOriginal = *pTargetData;
               // if we can't fit it, then we should increase our StorageDataType size!
               EBM_ASSERT(!IsConvertError<size_t>(targetOriginal));
               const size_t target = static_cast<size_t>(targetOriginal);
               EBM_ASSERT(target < static_cast<size_t>(cRuntimeClasses));

               const FloatFast gradient = EbmStats::InverseLinkFunctionThenCalculateGradientBinaryClassification(initScore, target);
               const FloatFast hessian = EbmStats::CalculateHessianFromGradientBinaryClassification(gradient);

               do {
                  EBM_ASSERT(pGradientAndHessian < pGradientAndHessianEnd - 1);
                  *pGradientAndHessian = gradient;
                  *(pGradientAndHessian + 1) = hessian;
                  pGradientAndHessian += 2;

                  countBagged -= direction;
               } while(BagEbmType { 0 } != countBagged);
            }
         }
         ++pTargetData;
      } while(pGradientAndHessianEnd != pGradientAndHessian);

      LOG_0(TraceLevelInfo, "Exited InitializeGradientsAndHessians");
      return Error_None;
   }
};
#endif // EXPAND_BINARY_LOGITS

template<>
class InitializeGradientsAndHessiansInternal<k_regression> final {
public:

   InitializeGradientsAndHessiansInternal() = delete; // this is a static class.  Do not construct

   static ErrorEbmType Func(
      const ptrdiff_t cRuntimeClasses,
      const BagEbmType direction,
      const BagEbmType * const aBag,
      const void * const aTargets,
      const double * const aInitScores,
      const size_t cSetSamples,
      FloatFast * const aGradientAndHessian
   ) {
      UNUSED(cRuntimeClasses);
      LOG_0(TraceLevelInfo, "Entered InitializeGradientsAndHessians");

      // TODO : review this function to see if iZeroLogit was set to a valid index, does that affect the number of items in pInitScore (I assume so), 
      //   and does it affect any calculations below like sumExp += std::exp(initScore) and the equivalent.  Should we use cScores or 
      //   cRuntimeClasses for some of the addition
      // TODO : !!! re-examine the idea of zeroing one of the logits with iZeroLogit after we have the ability to test large numbers of datasets

      EBM_ASSERT(BagEbmType { -1 } == direction || BagEbmType { 1 } == direction);
      EBM_ASSERT(0 < cSetSamples);
      EBM_ASSERT(nullptr != aTargets);
      EBM_ASSERT(nullptr != aGradientAndHessian);

      const BagEbmType * pBag = aBag;
      const FloatFast * pTargetData = static_cast<const FloatFast *>(aTargets);
      const double * pInitScore = aInitScores;
      FloatFast * pGradientAndHessian = aGradientAndHessian;
      const FloatFast * const pGradientAndHessianEnd = aGradientAndHessian + cSetSamples;
      const bool isLoopTraining = BagEbmType { 0 } < direction;
      do {
         BagEbmType countBagged = 1;
         if(nullptr != pBag) {
            countBagged = *pBag;
            ++pBag;
         }
         if(BagEbmType { 0 } != countBagged) {
            FloatFast initScore = 0;
            if(nullptr != pInitScore) {
               initScore = SafeConvertFloat<FloatFast>(*pInitScore);
               ++pInitScore;
            }
            const bool isItemTraining = BagEbmType { 0 } < countBagged;
            if(isLoopTraining == isItemTraining) {
               // TODO : our caller should handle NaN *pTargetData values, which means that the target is missing, which means we should delete that sample 
               //   from the input data

               // if data is NaN, we pass this along and NaN propagation will ensure that we stop boosting immediately.
               // There is no need to check it here since we already have graceful detection later for other reasons.

               const FloatFast data = *pTargetData;
               // TODO: NaN target values essentially mean missing, so we should be filtering those samples out, but our caller should do that so 
               //   that we don't need to do the work here per outer bag.  Our job in C++ is just not to crash or return inexplicable values.
               const FloatFast gradient = EbmStats::ComputeGradientRegressionMSEInit(initScore, data);
               do {
                  EBM_ASSERT(pGradientAndHessian < pGradientAndHessianEnd);
                  *pGradientAndHessian = gradient;
                  ++pGradientAndHessian;

                  countBagged -= direction;
               } while(BagEbmType { 0 } != countBagged);
            }
         }
         ++pTargetData;
      } while(pGradientAndHessianEnd != pGradientAndHessian);

      LOG_0(TraceLevelInfo, "Exited InitializeGradientsAndHessians");
      return Error_None;
   }
};

extern ErrorEbmType InitializeGradientsAndHessians(
   const unsigned char * const pDataSetShared,
   const BagEbmType direction,
   const BagEbmType * const aBag,
   const double * const aInitScores,
   const size_t cSetSamples,
   FloatFast * const aGradientAndHessian
) {
   ptrdiff_t cRuntimeClasses;
   const void * const aTargets = GetDataSetSharedTarget(
      pDataSetShared,
      0,
      &cRuntimeClasses
   );
   EBM_ASSERT(nullptr != aTargets);

   if(IsClassification(cRuntimeClasses)) {
      if(IsBinaryClassification(cRuntimeClasses)) {
         return InitializeGradientsAndHessiansInternal<2>::Func(
            cRuntimeClasses,
            direction,
            aBag,
            aTargets,
            aInitScores,
            cSetSamples,
            aGradientAndHessian
         );
      } else {
         return InitializeGradientsAndHessiansInternal<k_dynamicClassification>::Func(
            cRuntimeClasses,
            direction,
            aBag,
            aTargets,
            aInitScores,
            cSetSamples,
            aGradientAndHessian
         );
      }
   } else {
      EBM_ASSERT(IsRegression(cRuntimeClasses));
      return InitializeGradientsAndHessiansInternal<k_regression>::Func(
         cRuntimeClasses,
         direction,
         aBag,
         aTargets,
         aInitScores,
         cSetSamples,
         aGradientAndHessian
      );
   }
}

} // DEFINED_ZONE_NAME
