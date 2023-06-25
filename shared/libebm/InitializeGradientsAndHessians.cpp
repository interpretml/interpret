// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stddef.h> // size_t, ptrdiff_t

#include "libebm.h" // ErrorEbm
#include "logging.h" // EBM_ASSERT
#include "common_c.h" // FloatFast
#include "zones.h"

#include "ebm_internal.hpp" //SafeConvertFloat

#include "ebm_stats.hpp"
#include "dataset_shared.hpp" // GetDataSetSharedTarget
#include "DataSetBoosting.hpp"
#include "DataSetInteraction.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

WARNING_PUSH
WARNING_DISABLE_UNINITIALIZED_LOCAL_VARIABLE
extern void InitializeRmseGradientsAndHessiansBoosting(
   const unsigned char * const pDataSetShared,
   const BagEbm direction,
   const BagEbm * const aBag,
   const double * const aInitScores,
   DataSetBoosting * const pDataSet
) {
   // RMSE regression is super super special in that we do not need to keep the scores and we can just use gradients

   LOG_0(Trace_Info, "Entered InitializeRmseGradientsAndHessiansBoosting");

   EBM_ASSERT(nullptr != pDataSetShared);
   EBM_ASSERT(BagEbm { -1 } == direction || BagEbm { 1 } == direction);
   EBM_ASSERT(nullptr != pDataSet);

   if(size_t { 0 } != pDataSet->GetCountSamples()) {
      ptrdiff_t cRuntimeClasses;
      const FloatFast * pTargetData =
         static_cast<const FloatFast *>(GetDataSetSharedTarget(pDataSetShared, 0, &cRuntimeClasses));
      EBM_ASSERT(nullptr != pTargetData); // we previously called GetDataSetSharedTarget and got back non-null result
      EBM_ASSERT(IsRegression(cRuntimeClasses));

      const BagEbm * pSampleReplication = aBag;
      const double * pInitScore = aInitScores;

      const bool isLoopValidation = direction < BagEbm { 0 };
      EBM_ASSERT(nullptr != aBag || !isLoopValidation); // if pSampleReplication is nullptr then we have no validation samples

      EBM_ASSERT(1 <= pDataSet->GetCountSamples());
      EBM_ASSERT(1 <= pDataSet->GetCountSubsets());
      DataSubsetBoosting * pSubset = pDataSet->GetSubsets();
      EBM_ASSERT(nullptr != pSubset);
      const DataSubsetBoosting * const pSubsetsEnd = pSubset + pDataSet->GetCountSubsets();

      BagEbm replication = 0;
      FloatFast gradient;
      do {
         EBM_ASSERT(1 <= pSubset->GetCountSamples());
         FloatFast * pGradHess = pSubset->GetGradHess();
         EBM_ASSERT(nullptr != pGradHess);
         const FloatFast * const pGradHessEnd = pGradHess + pSubset->GetCountSamples();

         do {
            if(BagEbm { 0 } == replication) {
               replication = 1;
               size_t cInitAdvances = 1;
               if(nullptr != pSampleReplication) {
                  cInitAdvances = 0;
                  bool isItemValidation;
                  do {
                     do {
                        replication = *pSampleReplication;
                        ++pSampleReplication;
                        ++pTargetData;
                     } while(BagEbm { 0 } == replication);
                     isItemValidation = replication < BagEbm { 0 };
                     ++cInitAdvances;
                  } while(isLoopValidation != isItemValidation);
                  --pTargetData;
               }
               const FloatFast data = *pTargetData;
               ++pTargetData;

               FloatFast initScore = 0;
               if(nullptr != pInitScore) {
                  pInitScore += cInitAdvances;
                  initScore = SafeConvertFloat<FloatFast>(pInitScore[-1]);
               }

               // TODO : our caller should handle NaN *pTargetData values, which means that the target is missing, which means we should delete that sample 
               //   from the input data

               // if data is NaN, we pass this along and NaN propagation will ensure that we stop boosting immediately.
               // There is no need to check it here since we already have graceful detection later for other reasons.

               // TODO: NaN target values essentially mean missing, so we should be filtering those samples out, but our caller should do that so 
               //   that we don't need to do the work here per outer bag.  Our job in C++ is just not to crash or return inexplicable values.
               gradient = EbmStats::ComputeGradientRegressionRmseInit(initScore, data);
            }

            *pGradHess = gradient;
            ++pGradHess;

            replication -= direction;
         } while(pGradHessEnd != pGradHess);

         ++pSubset;
      } while(pSubsetsEnd != pSubset);
      EBM_ASSERT(0 == replication);
   }
   LOG_0(Trace_Info, "Exited InitializeRmseGradientsAndHessiansBoosting");
}
WARNING_POP

WARNING_PUSH
WARNING_DISABLE_UNINITIALIZED_LOCAL_VARIABLE
extern void InitializeRmseGradientsAndHessiansInteraction(
   const unsigned char * const pDataSetShared,
   const BagEbm * const aBag,
   const double * const aInitScores,
   DataSetInteraction * const pDataSet
) {
   // RMSE regression is super super special in that we do not need to keep the scores and we can just use gradients

   LOG_0(Trace_Info, "Entered InitializeRmseGradientsAndHessiansInteraction");

   EBM_ASSERT(nullptr != pDataSetShared);
   EBM_ASSERT(nullptr != pDataSet);

   size_t cIncludedSamples = pDataSet->GetCountSamples();
   if(size_t { 0 } != cIncludedSamples) {
      ptrdiff_t cRuntimeClasses;
      const FloatFast * pTargetData =
         static_cast<const FloatFast *>(GetDataSetSharedTarget(pDataSetShared, 0, &cRuntimeClasses));
      EBM_ASSERT(nullptr != pTargetData); // we previously called GetDataSetSharedTarget and got back non-null result
      EBM_ASSERT(IsRegression(cRuntimeClasses));

      EBM_ASSERT(1 <= pDataSet->GetCountSamples());

      DataSubsetInteraction * pSubset = pDataSet->GetSubsets();
      EBM_ASSERT(nullptr != pSubset);

      const FloatFast * pWeight = nullptr;
      // check the first subset just to see if weights were specified
      if(nullptr != pSubset->GetWeights()) {
         pWeight = GetDataSetSharedWeight(pDataSetShared, 0);
         EBM_ASSERT(nullptr != pWeight);
      }

      EBM_ASSERT(1 <= pDataSet->GetCountSubsets());
      const DataSubsetInteraction * const pSubsetsEnd = pSubset + pDataSet->GetCountSubsets();

      const BagEbm * pSampleReplication = aBag;
      const double * pInitScore = aInitScores;

      BagEbm replication = 0;
      FloatFast gradient;
      do {
         EBM_ASSERT(1 <= pSubset->GetCountSamples());
         FloatFast * pGradHess = pSubset->GetGradHess();
         EBM_ASSERT(nullptr != pGradHess);
         const FloatFast * const pGradHessEnd = pGradHess + pSubset->GetCountSamples();

         EBM_ASSERT(nullptr == pWeight && nullptr == pSubset->GetWeights() ||
            nullptr != pWeight && nullptr != pSubset->GetWeights());
         do {
            if(BagEbm { 0 } == replication) {
               replication = 1;
               size_t cInitAdvances = 1;
               size_t cSharedAdvances = 1;
               if(nullptr != pSampleReplication) {
                  cInitAdvances = 0;
                  cSharedAdvances = 0;
                  do {
                     do {
                        replication = pSampleReplication[cSharedAdvances];
                        ++cSharedAdvances;
                     } while(BagEbm { 0 } == replication);
                     ++cInitAdvances;
                  } while(replication < BagEbm { 0 });
                  pSampleReplication += cSharedAdvances;
               }
               pTargetData += cSharedAdvances;
               const FloatFast data = pTargetData[-1];

               FloatFast initScore = 0;
               if(nullptr != pInitScore) {
                  pInitScore += cInitAdvances;
                  initScore = SafeConvertFloat<FloatFast>(pInitScore[-1]);
               }

               // TODO : our caller should handle NaN *pTargetData values, which means that the target is missing, which means we should delete that sample 
               //   from the input data

               // if data is NaN, we pass this along and NaN propagation will ensure that we stop boosting immediately.
               // There is no need to check it here since we already have graceful detection later for other reasons.

               // TODO: NaN target values essentially mean missing, so we should be filtering those samples out, but our caller should do that so 
               //   that we don't need to do the work here per outer bag.  Our job in C++ is just not to crash or return inexplicable values.
               gradient = EbmStats::ComputeGradientRegressionRmseInit(initScore, data);

               if(nullptr != pWeight) {
                  // This is only used during the initialization of interaction detection. For boosting
                  // we currently multiply by the weight during bin summation instead since we use the weight
                  // there to include the inner bagging counts of occurences.
                  // Whether this multiplication happens or not is controlled by the caller by passing in the
                  // weight array or not.
                  pWeight += cSharedAdvances;
                  gradient *= pWeight[-1];
               }
            }

            *pGradHess = gradient;
            ++pGradHess;

            --replication;
         } while(pGradHessEnd != pGradHess);

         ++pSubset;
      } while(pSubsetsEnd != pSubset);
      EBM_ASSERT(0 == replication);
   }
   LOG_0(Trace_Info, "Exited InitializeRmseGradientsAndHessiansInteraction");
}
WARNING_POP

} // DEFINED_ZONE_NAME
