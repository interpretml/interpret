// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>


#include "precompiled_header_cpp.hpp"

#include <stddef.h> // size_t, ptrdiff_t

#include "libebm.h" // ErrorEbm
#include "logging.h" // EBM_ASSERT
#include "common_c.h" // FloatFast
#include "zones.h"

#include "ebm_internal.hpp"

#include "ebm_stats.hpp"
#include "dataset_shared.hpp" // GetDataSetSharedTarget
#include "DataSetBoosting.hpp"
#include "DataSetInteraction.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

extern void InitializeRmseGradientsAndHessiansBoosting(
   const unsigned char * const pDataSetShared,
   const BagEbm direction,
   const BagEbm * const aBag,
   const double * const aInitScores,
   DataSubsetBoosting * const pDataSet
) {
   // RMSE regression is super super special in that we do not need to keep the scores and we can just use gradients

   LOG_0(Trace_Info, "Entered InitializeRmseGradientsAndHessiansBoosting");

   ptrdiff_t cRuntimeClasses;
   const void * const aTargets = GetDataSetSharedTarget(pDataSetShared, 0, &cRuntimeClasses);
   EBM_ASSERT(nullptr != aTargets); // we previously called GetDataSetSharedTarget and got back non-null result
   EBM_ASSERT(IsRegression(cRuntimeClasses));

   const size_t cSetSamples = pDataSet->GetCountSamples();
   FloatFast * const aGradientAndHessian = pDataSet->GetGradientsAndHessiansPointer();

   EBM_ASSERT(BagEbm { -1 } == direction || BagEbm { 1 } == direction);
   EBM_ASSERT(1 <= cSetSamples);
   EBM_ASSERT(nullptr != aGradientAndHessian);

   const BagEbm * pSampleReplication = aBag;
   const bool isLoopTraining = BagEbm { 0 } < direction;
   EBM_ASSERT(nullptr != aBag || isLoopTraining); // if pSampleReplication is nullptr then we have no validation samples

   const FloatFast * pTargetData = static_cast<const FloatFast *>(aTargets);
   const double * pInitScore = aInitScores;
   FloatFast * pGradientAndHessian = aGradientAndHessian;
   const FloatFast * const pGradientAndHessianEnd = aGradientAndHessian + cSetSamples;
   do {
      BagEbm replication = 1;
      size_t cInitAdvances = 1;
      if(nullptr != pSampleReplication) {
         bool isItemTraining;
         do {
            do {
               replication = *pSampleReplication;
               ++pSampleReplication;
               ++pTargetData;
            } while(BagEbm { 0 } == replication);
            isItemTraining = BagEbm { 0 } < replication;
            ++cInitAdvances;
         } while(isLoopTraining != isItemTraining);
         --pTargetData;
         --cInitAdvances;
      }
      const FloatFast data = *pTargetData;
      ++pTargetData;

      FloatFast initScore = 0;
      if(nullptr != pInitScore) {
         pInitScore += cInitAdvances;
         initScore = SafeConvertFloat<FloatFast>(*(pInitScore - 1));
      }

      // TODO : our caller should handle NaN *pTargetData values, which means that the target is missing, which means we should delete that sample 
      //   from the input data

      // if data is NaN, we pass this along and NaN propagation will ensure that we stop boosting immediately.
      // There is no need to check it here since we already have graceful detection later for other reasons.

      // TODO: NaN target values essentially mean missing, so we should be filtering those samples out, but our caller should do that so 
      //   that we don't need to do the work here per outer bag.  Our job in C++ is just not to crash or return inexplicable values.
      FloatFast gradient = EbmStats::ComputeGradientRegressionRmseInit(initScore, data);
      do {
         EBM_ASSERT(pGradientAndHessian < pGradientAndHessianEnd);
         *pGradientAndHessian = gradient;
         ++pGradientAndHessian;

         replication -= direction;
      } while(BagEbm { 0 } != replication);
   } while(pGradientAndHessianEnd != pGradientAndHessian);

   LOG_0(Trace_Info, "Exited InitializeRmseGradientsAndHessiansBoosting");
}

extern void InitializeRmseGradientsAndHessiansInteraction(
   const unsigned char * const pDataSetShared,
   const BagEbm * const aBag,
   const double * const aInitScores,
   DataSubsetInteraction * const pDataSet
) {
   // RMSE regression is super super special in that we do not need to keep the scores and we can just use gradients

   LOG_0(Trace_Info, "Entered InitializeRmseGradientsAndHessiansInteraction");

   ptrdiff_t cRuntimeClasses;
   const void * const aTargets = GetDataSetSharedTarget(pDataSetShared, 0, &cRuntimeClasses);
   EBM_ASSERT(nullptr != aTargets); // we previously called GetDataSetSharedTarget and got back non-null result
   EBM_ASSERT(IsRegression(cRuntimeClasses));

   const size_t cSetSamples = pDataSet->GetCountSamples();
   FloatFast * const aGradientAndHessian = pDataSet->GetGradientsAndHessiansPointer();
   const FloatFast * const aWeight = pDataSet->GetWeights();

   EBM_ASSERT(1 <= cSetSamples);
   EBM_ASSERT(nullptr != aGradientAndHessian);

   const FloatFast * pWeight = aWeight; // has been expanded to match the length of our output (aGradientAndHessian)

   const BagEbm * pSampleReplication = aBag;

   const FloatFast * pTargetData = static_cast<const FloatFast *>(aTargets);
   const double * pInitScore = aInitScores;
   FloatFast * pGradientAndHessian = aGradientAndHessian;
   const FloatFast * const pGradientAndHessianEnd = aGradientAndHessian + cSetSamples;
   do {
      BagEbm replication = 1;
      size_t cInitAdvances = 1;
      if(nullptr != pSampleReplication) {
         do {
            do {
               replication = *pSampleReplication;
               ++pSampleReplication;
               ++pTargetData;
            } while(BagEbm { 0 } == replication);
            ++cInitAdvances;
         } while(replication < BagEbm { 0 });
         --pTargetData;
         --cInitAdvances;
      }
      const FloatFast data = *pTargetData;
      ++pTargetData;

      FloatFast initScore = 0;
      if(nullptr != pInitScore) {
         pInitScore += cInitAdvances;
         initScore = SafeConvertFloat<FloatFast>(*(pInitScore - 1));
      }

      // TODO : our caller should handle NaN *pTargetData values, which means that the target is missing, which means we should delete that sample 
      //   from the input data

      // if data is NaN, we pass this along and NaN propagation will ensure that we stop boosting immediately.
      // There is no need to check it here since we already have graceful detection later for other reasons.

      // TODO: NaN target values essentially mean missing, so we should be filtering those samples out, but our caller should do that so 
      //   that we don't need to do the work here per outer bag.  Our job in C++ is just not to crash or return inexplicable values.
      FloatFast gradient = EbmStats::ComputeGradientRegressionRmseInit(initScore, data);

      if(nullptr != pWeight) {
         // This is only used during the initialization of interaction detection. For boosting
         // we currently multiply by the weight during bin summation instead since we use the weight
         // there to include the inner bagging counts of occurences.
         // Whether this multiplication happens or not is controlled by the caller by passing in the
         // weight array or not.
         gradient *= *pWeight;
         pWeight += EbmAbs(replication);
      }
      do {
         EBM_ASSERT(pGradientAndHessian < pGradientAndHessianEnd);
         *pGradientAndHessian = gradient;
         ++pGradientAndHessian;

         --replication;
      } while(BagEbm { 0 } != replication);
   } while(pGradientAndHessianEnd != pGradientAndHessian);

   LOG_0(Trace_Info, "Exited InitializeRmseGradientsAndHessiansInteraction");
}

} // DEFINED_ZONE_NAME
