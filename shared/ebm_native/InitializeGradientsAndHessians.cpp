// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>


#include "precompiled_header_cpp.hpp"

#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h" // ErrorEbm
#include "logging.h" // EBM_ASSERT
#include "common_c.h" // FloatFast
#include "zones.h"

#include "ebm_stats.hpp"
#include "dataset_shared.hpp" // GetDataSetSharedTarget

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

extern void InitializeMSEGradientsAndHessians(
   const unsigned char * const pDataSetShared,
   const BagEbm direction,
   const BagEbm * const aBag,
   const double * const aInitScores,
   const size_t cSetSamples,
   FloatFast * const aGradientAndHessian
) {
   // MSE regression is super super special in that we do not need to keep the scores and we can just use gradients

   ptrdiff_t cRuntimeClasses;
   const void * const aTargets = GetDataSetSharedTarget(pDataSetShared, 0, &cRuntimeClasses);
   EBM_ASSERT(nullptr != aTargets);
   EBM_ASSERT(IsRegression(cRuntimeClasses));

   LOG_0(Trace_Info, "Entered InitializeMSEGradientsAndHessians");

   EBM_ASSERT(BagEbm { -1 } == direction || BagEbm { 1 } == direction);
   EBM_ASSERT(1 <= cSetSamples);
   EBM_ASSERT(nullptr != aGradientAndHessian);

   const BagEbm * pSampleReplication = aBag;
   const FloatFast * pTargetData = static_cast<const FloatFast *>(aTargets);
   const double * pInitScore = aInitScores;
   FloatFast * pGradientAndHessian = aGradientAndHessian;
   const FloatFast * const pGradientAndHessianEnd = aGradientAndHessian + cSetSamples;
   const bool isLoopTraining = BagEbm { 0 } < direction;
   do {
      BagEbm replication = 1;
      if(nullptr != pSampleReplication) {
         replication = *pSampleReplication;
         ++pSampleReplication;
      }
      if(BagEbm { 0 } != replication) {
         FloatFast initScore = 0;
         if(nullptr != pInitScore) {
            initScore = SafeConvertFloat<FloatFast>(*pInitScore);
            ++pInitScore;
         }
         const bool isItemTraining = BagEbm { 0 } < replication;
         if(isLoopTraining == isItemTraining) {
            // TODO : our caller should handle NaN *pTargetData values, which means that the target is missing, which means we should delete that sample 
            //   from the input data

            // if data is NaN, we pass this along and NaN propagation will ensure that we stop boosting immediately.
            // There is no need to check it here since we already have graceful detection later for other reasons.

            const FloatFast data = *pTargetData; // TODO: is this faster if we always fetch data thus making the load more predictable
            // TODO: NaN target values essentially mean missing, so we should be filtering those samples out, but our caller should do that so 
            //   that we don't need to do the work here per outer bag.  Our job in C++ is just not to crash or return inexplicable values.
            const FloatFast gradient = EbmStats::ComputeGradientRegressionMSEInit(initScore, data);
            do {
               EBM_ASSERT(pGradientAndHessian < pGradientAndHessianEnd);
               *pGradientAndHessian = gradient;
               ++pGradientAndHessian;

               replication -= direction;
            } while(BagEbm { 0 } != replication);
         }
      }
      ++pTargetData;
   } while(pGradientAndHessianEnd != pGradientAndHessian);

   LOG_0(Trace_Info, "Exited InitializeMSEGradientsAndHessians");
}

} // DEFINED_ZONE_NAME
