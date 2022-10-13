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
   EBM_ASSERT(nullptr != aBag || isLoopTraining); // if pSampleReplication is nullptr then we have no validation samples
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
      const FloatFast gradient = EbmStats::ComputeGradientRegressionMSEInit(initScore, data);
      do {
         EBM_ASSERT(pGradientAndHessian < pGradientAndHessianEnd);
         *pGradientAndHessian = gradient;
         ++pGradientAndHessian;

         replication -= direction;
      } while(BagEbm { 0 } != replication);
   } while(pGradientAndHessianEnd != pGradientAndHessian);

   LOG_0(Trace_Info, "Exited InitializeMSEGradientsAndHessians");
}

} // DEFINED_ZONE_NAME
