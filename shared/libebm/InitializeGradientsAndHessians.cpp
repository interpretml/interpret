// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "pch.hpp"

#include <stddef.h> // size_t, ptrdiff_t

#include "libebm.h" // ErrorEbm
#include "logging.h" // EBM_ASSERT
#include "unzoned.h"

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
      const FloatShared * pTargetData =
         static_cast<const FloatShared *>(GetDataSetSharedTarget(pDataSetShared, 0, &cRuntimeClasses));
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

      double initScore = 0;
      BagEbm replication = 0;
      double gradient;
      do {
         EBM_ASSERT(1 <= pSubset->GetCountSamples());
         void * pGradHess = pSubset->GetGradHess();
         EBM_ASSERT(nullptr != pGradHess);
         const void * const pGradHessEnd = IndexByte(pGradHess, pSubset->GetObjectiveWrapper()->m_cFloatBytes * pSubset->GetCountSamples());

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
               const FloatShared data = *pTargetData;
               ++pTargetData;

               if(nullptr != pInitScore) {
                  pInitScore += cInitAdvances;
                  initScore = pInitScore[-1];
               }

               // TODO : our caller should handle NaN *pTargetData values, which means that the target is missing, which means we should delete that sample 
               //   from the input data

               // if data is NaN, we pass this along and NaN propagation will ensure that we stop boosting immediately.
               // There is no need to check it here since we already have graceful detection later for other reasons.

               // TODO: NaN target values essentially mean missing, so we should be filtering those samples out, but our caller should do that so 
               //   that we don't need to do the work here per outer bag.  Our job in C++ is just not to crash or return inexplicable values.


               // for RMSE regression, the gradient is the residual, and we can calculate it once at init and we don't need
               // to keep the original scores when computing the gradient updates.

               gradient = initScore - static_cast<double>(data);
            }

            if(sizeof(FloatBig) == pSubset->GetObjectiveWrapper()->m_cFloatBytes) {
               *reinterpret_cast<FloatBig *>(pGradHess) = static_cast<FloatBig>(gradient);
            } else {
               EBM_ASSERT(sizeof(FloatSmall) == pSubset->GetObjectiveWrapper()->m_cFloatBytes);
               *reinterpret_cast<FloatSmall *>(pGradHess) = static_cast<FloatSmall>(gradient);
            }
            pGradHess = IndexByte(pGradHess, pSubset->GetObjectiveWrapper()->m_cFloatBytes);

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
   const size_t cWeights,
   const BagEbm * const aBag,
   const double * const aInitScores,
   DataSetInteraction * const pDataSet
) {
   // RMSE regression is super super special in that we do not need to keep the scores and we can just use gradients

   LOG_0(Trace_Info, "Entered InitializeRmseGradientsAndHessiansInteraction");

   EBM_ASSERT(nullptr != pDataSetShared);
   EBM_ASSERT(nullptr != pDataSet);

   if(size_t { 0 } != pDataSet->GetCountSamples()) {
      ptrdiff_t cRuntimeClasses;
      const FloatShared * pTargetData =
         static_cast<const FloatShared *>(GetDataSetSharedTarget(pDataSetShared, 0, &cRuntimeClasses));
      EBM_ASSERT(nullptr != pTargetData); // we previously called GetDataSetSharedTarget and got back non-null result
      EBM_ASSERT(IsRegression(cRuntimeClasses));

      DataSubsetInteraction * pSubset = pDataSet->GetSubsets();
      EBM_ASSERT(nullptr != pSubset);

      const FloatShared * pWeight = nullptr;
      if(size_t { 0 } != cWeights) {
         pWeight = GetDataSetSharedWeight(pDataSetShared, 0);
         EBM_ASSERT(nullptr != pWeight);
      }

      EBM_ASSERT(1 <= pDataSet->GetCountSubsets());
      const DataSubsetInteraction * const pSubsetsEnd = pSubset + pDataSet->GetCountSubsets();

      const BagEbm * pSampleReplication = aBag;
      const double * pInitScore = aInitScores;

      double initScore = 0;
      BagEbm replication = 0;
      double gradient;
      do {
         EBM_ASSERT(1 <= pSubset->GetCountSamples());
         void * pGradHess = pSubset->GetGradHess();
         EBM_ASSERT(nullptr != pGradHess);
         const void * const pGradHessEnd = IndexByte(pGradHess, pSubset->GetObjectiveWrapper()->m_cFloatBytes * pSubset->GetCountSamples());

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
               const FloatShared data = pTargetData[-1];

               if(nullptr != pInitScore) {
                  pInitScore += cInitAdvances;
                  initScore = pInitScore[-1];
               }

               // TODO : our caller should handle NaN *pTargetData values, which means that the target is missing, which means we should delete that sample 
               //   from the input data

               // if data is NaN, we pass this along and NaN propagation will ensure that we stop boosting immediately.
               // There is no need to check it here since we already have graceful detection later for other reasons.

               // TODO: NaN target values essentially mean missing, so we should be filtering those samples out, but our caller should do that so 
               //   that we don't need to do the work here per outer bag.  Our job in C++ is just not to crash or return inexplicable values.


               // for RMSE regression, the gradient is the residual, and we can calculate it once at init and we don't need
               // to keep the original scores when computing the gradient updates.

               gradient = initScore - static_cast<double>(data);

               if(nullptr != pWeight) {
                  // This is only used during the initialization of interaction detection. For boosting
                  // we currently multiply by the weight during bin summation instead since we use the weight
                  // there to include the inner bagging counts of occurences.
                  // Whether this multiplication happens or not is controlled by the caller by passing in the
                  // weight array or not.
                  pWeight += cSharedAdvances;
                  gradient *= static_cast<double>(pWeight[-1]);
               }
            }

            if(sizeof(FloatBig) == pSubset->GetObjectiveWrapper()->m_cFloatBytes) {
               *reinterpret_cast<FloatBig *>(pGradHess) = static_cast<FloatBig>(gradient);
            } else {
               EBM_ASSERT(sizeof(FloatSmall) == pSubset->GetObjectiveWrapper()->m_cFloatBytes);
               *reinterpret_cast<FloatSmall *>(pGradHess) = static_cast<FloatSmall>(gradient);
            }
            pGradHess = IndexByte(pGradHess, pSubset->GetObjectiveWrapper()->m_cFloatBytes);

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
