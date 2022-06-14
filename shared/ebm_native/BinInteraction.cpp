// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "ebm_internal.hpp"

#include "ebm_stats.hpp"

#include "Feature.hpp"
#include "FeatureGroup.hpp"
#include "DataSetInteraction.hpp"

#include "InteractionCore.hpp"

#include "HistogramTargetEntry.hpp"
#include "HistogramBucket.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses, size_t cCompilerDimensions>
class BinInteractionInternal final {
public:

   BinInteractionInternal() = delete; // this is a static class.  Do not construct

   static void Func(
      InteractionCore * const pInteractionCore,
      const FeatureGroup * const pFeatureGroup,
      HistogramBucketBase * const aHistogramBucketBase
#ifndef NDEBUG
      , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
   ) {
      constexpr bool bClassification = IsClassification(compilerLearningTypeOrCountTargetClasses);

      LOG_0(TraceLevelVerbose, "Entered BinInteractionInternal");

      auto * const aHistogramBuckets = aHistogramBucketBase->GetHistogramBucket<FloatEbmType, bClassification>();

      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pInteractionCore->GetRuntimeLearningTypeOrCountTargetClasses();

      const ptrdiff_t learningTypeOrCountTargetClasses = GET_LEARNING_TYPE_OR_COUNT_TARGET_CLASSES(
         compilerLearningTypeOrCountTargetClasses,
         runtimeLearningTypeOrCountTargetClasses
      );
      const size_t cVectorLength = GetVectorLength(learningTypeOrCountTargetClasses);
      EBM_ASSERT(!GetHistogramBucketSizeOverflow<FloatEbmType>(bClassification, cVectorLength)); // we're accessing allocated memory
      const size_t cBytesPerHistogramBucket = GetHistogramBucketSize<FloatEbmType>(bClassification, cVectorLength);

      const DataSetInteraction * const pDataSet = pInteractionCore->GetDataSetInteraction();
      const FloatEbmType * pGradientAndHessian = pDataSet->GetGradientsAndHessiansPointer();
      const FloatEbmType * const pGradientsAndHessiansEnd = pGradientAndHessian + (bClassification ? 2 : 1) * cVectorLength * pDataSet->GetCountSamples();

      const FloatEbmType * pWeight = pDataSet->GetWeights();

      EBM_ASSERT(pFeatureGroup->GetCountDimensions() == pFeatureGroup->GetCountSignificantDimensions()); // for interactions, we just return 0 for interactions with zero features
      const size_t cDimensions = GET_DIMENSIONS(cCompilerDimensions, pFeatureGroup->GetCountSignificantDimensions());
      EBM_ASSERT(1 <= cDimensions); // for interactions, we just return 0 for interactions with zero features

#ifndef NDEBUG
      FloatEbmType weightTotalDebug = 0;
#endif // NDEBUG

      for(size_t iSample = 0; pGradientsAndHessiansEnd != pGradientAndHessian; ++iSample) {
         // this loop gets about twice as slow if you add a single unpredictable branching if statement based on count, even if you still access all the memory
         // in complete sequential order, so we'll probably want to use non-branching instructions for any solution like conditional selection or multiplication
         // this loop gets about 3 times slower if you use a bad pseudo random number generator like rand(), although it might be better if you inlined rand().
         // this loop gets about 10 times slower if you use a proper pseudo random number generator like std::default_random_engine
         // taking all the above together, it seems unlikley we'll use a method of separating sets via single pass randomized set splitting.  Even if count is 
         // stored in memory if shouldn't increase the time spent fetching it by 2 times, unless our bottleneck when threading is overwhelmingly memory pressure 
         // related, and even then we could store the count for a single bit aleviating the memory pressure greatly, if we use the right sampling method 

         // TODO : try using a sampling method with non-repeating samples, and put the count into a bit.  Then unwind that loop either at the byte level 
         //   (8 times) or the uint64_t level.  This can be done without branching and doesn't require random number generators

         // TODO : we can elminate the inner vector loop for regression at least, and also if we add a templated bool for binary class.  Propegate this change 
         //   to all places that we loop on the vector

         size_t cBuckets = 1;
         size_t iBucket = 0;
         size_t iDimension = 0;
         do {
            const Feature * const pInputFeature = pFeatureGroup->GetFeatureGroupEntries()[iDimension].m_pFeature;
            const size_t cBins = pInputFeature->GetCountBins();
            // interactions return interaction score of zero earlier on any useless dimensions
            // we strip dimensions from the tensors with 1 bin, so if 1 bin was accepted here, we'd need to strip
            // the bin too
            EBM_ASSERT(size_t { 2 } <= cBins);
            const StorageDataType * pInputData = pDataSet->GetInputDataPointer(pInputFeature);
            pInputData += iSample;
            StorageDataType iBinOriginal = *pInputData;
            EBM_ASSERT(!IsConvertError<size_t>(iBinOriginal));
            size_t iBin = static_cast<size_t>(iBinOriginal);
            EBM_ASSERT(iBin < cBins);
            iBucket += cBuckets * iBin;
            cBuckets *= cBins;
            ++iDimension;
         } while(iDimension < cDimensions);

         auto * pHistogramBucketEntry = 
            GetHistogramBucketByIndex(cBytesPerHistogramBucket, aHistogramBuckets, iBucket);
         ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, pHistogramBucketEntry, aHistogramBucketsEndDebug);
         pHistogramBucketEntry->SetCountSamplesInBucket(pHistogramBucketEntry->GetCountSamplesInBucket() + 1);
         FloatEbmType weight = 1;
         if(nullptr != pWeight) {
            weight = *pWeight;
            ++pWeight;
#ifndef NDEBUG
            weightTotalDebug += weight;
#endif // NDEBUG
         }
         pHistogramBucketEntry->SetWeightInBucket(pHistogramBucketEntry->GetWeightInBucket() + weight);

         auto * const pHistogramTargetEntry = pHistogramBucketEntry->GetHistogramTargetEntry();

         for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
            const FloatEbmType gradient = *pGradientAndHessian;
            // gradient could be NaN
            // for classification, gradient can be anything from -1 to +1 (it cannot be infinity!)
            // for regression, gradient can be anything from +infinity or -infinity
            pHistogramTargetEntry[iVector].m_sumGradients += gradient * weight;
            // m_sumGradients could be NaN, or anything from +infinity or -infinity in the case of regression
            if(bClassification) {
               EBM_ASSERT(
                  std::isnan(gradient) ||
                  !std::isinf(gradient) && 
                  FloatEbmType { -1 } - k_epsilonGradient <= gradient && gradient <= FloatEbmType { 1 }
                  );

               // TODO : this code gets executed for each SamplingSet set.  I could probably execute it once and then all the SamplingSet
               //   sets would have this value, but I would need to store the computation in a new memory place, and it might make more sense to calculate this 
               //   values in the CPU rather than put more pressure on memory.  I think controlling this should be done in a MACRO and we should use a class to 
               //   hold the gradient and this computation from that value and then comment out the computation if not necssary and access it through an 
               //   accessor so that we can make the change entirely via macro
               const FloatEbmType hessian = *(pGradientAndHessian + 1);
               EBM_ASSERT(
                  std::isnan(hessian) ||
                  !std::isinf(hessian) && -k_epsilonGradient <= hessian && hessian <= FloatEbmType { 0.25 }
               ); // since any one hessian is limited to 0 <= hessian <= 0.25, the sum must be representable by a 64 bit number, 

               const FloatEbmType oldHessian = pHistogramTargetEntry[iVector].GetSumHessians();
               // since any one hessian is limited to 0 <= gradient <= 0.25, the sum must be representable by a 64 bit number, 
               EBM_ASSERT(std::isnan(oldHessian) || !std::isinf(oldHessian) && -k_epsilonGradient <= oldHessian);
               const FloatEbmType newHessian = oldHessian + hessian * weight;
               // since any one hessian is limited to 0 <= hessian <= 0.25, the sum must be representable by a 64 bit number, 
               EBM_ASSERT(std::isnan(newHessian) || !std::isinf(newHessian) && -k_epsilonGradient <= newHessian);
               // which will always be representable by a float or double, so we can't overflow to inifinity or -infinity
               pHistogramTargetEntry[iVector].SetSumHessians(newHessian);
            }
            pGradientAndHessian += bClassification ? 2 : 1;
         }
      }
      EBM_ASSERT(FloatEbmType { 0 } < pDataSet->GetWeightTotal());
      EBM_ASSERT(nullptr == pWeight || weightTotalDebug * 0.999 <= pDataSet->GetWeightTotal() && 
         pDataSet->GetWeightTotal() <= 1.001 * weightTotalDebug);
      EBM_ASSERT(nullptr != pWeight || 
         static_cast<FloatEbmType>(pDataSet->GetCountSamples()) == pDataSet->GetWeightTotal());

      LOG_0(TraceLevelVerbose, "Exited BinInteractionInternal");
   }
};

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses, size_t cCompilerDimensionsPossible>
class BinInteractionDimensions final {
public:

   BinInteractionDimensions() = delete; // this is a static class.  Do not construct

   INLINE_ALWAYS static void Func(
      InteractionCore * const pInteractionCore,
      const FeatureGroup * const pFeatureGroup,
      HistogramBucketBase * const aHistogramBuckets
#ifndef NDEBUG
      , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
   ) {
      static_assert(1 <= cCompilerDimensionsPossible, "can't have less than 1 dimension for interactions");
      static_assert(cCompilerDimensionsPossible <= k_cDimensionsMax, "can't have more than the max dimensions");

      const size_t cRuntimeDimensions = pFeatureGroup->GetCountSignificantDimensions();

      EBM_ASSERT(1 <= cRuntimeDimensions);
      EBM_ASSERT(cRuntimeDimensions <= k_cDimensionsMax);
      if(cCompilerDimensionsPossible == cRuntimeDimensions) {
         BinInteractionInternal<compilerLearningTypeOrCountTargetClasses, cCompilerDimensionsPossible>::Func(
            pInteractionCore,
            pFeatureGroup,
            aHistogramBuckets
#ifndef NDEBUG
            , aHistogramBucketsEndDebug
#endif // NDEBUG
         );
      } else {
         BinInteractionDimensions<compilerLearningTypeOrCountTargetClasses, cCompilerDimensionsPossible + 1>::Func(
            pInteractionCore,
            pFeatureGroup,
            aHistogramBuckets
#ifndef NDEBUG
            , aHistogramBucketsEndDebug
#endif // NDEBUG
         );
      }
   }
};

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
class BinInteractionDimensions<compilerLearningTypeOrCountTargetClasses, k_cCompilerOptimizedCountDimensionsMax + 1> final {
public:

   BinInteractionDimensions() = delete; // this is a static class.  Do not construct

   INLINE_ALWAYS static void Func(
      InteractionCore * const pInteractionCore,
      const FeatureGroup * const pFeatureGroup,
      HistogramBucketBase * const aHistogramBuckets
#ifndef NDEBUG
      , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
   ) {
      EBM_ASSERT(1 <= pFeatureGroup->GetCountSignificantDimensions());
      EBM_ASSERT(pFeatureGroup->GetCountSignificantDimensions() <= k_cDimensionsMax);
      BinInteractionInternal<compilerLearningTypeOrCountTargetClasses, k_dynamicDimensions>::Func(
         pInteractionCore,
         pFeatureGroup,
         aHistogramBuckets
#ifndef NDEBUG
         , aHistogramBucketsEndDebug
#endif // NDEBUG
      );
   }
};

template<ptrdiff_t compilerLearningTypeOrCountTargetClassesPossible>
class BinInteractionTarget final {
public:

   BinInteractionTarget() = delete; // this is a static class.  Do not construct

   INLINE_ALWAYS static void Func(
      InteractionCore * const pInteractionCore,
      const FeatureGroup * const pFeatureGroup,
      HistogramBucketBase * const aHistogramBuckets
#ifndef NDEBUG
      , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
   ) {
      static_assert(IsClassification(compilerLearningTypeOrCountTargetClassesPossible), "compilerLearningTypeOrCountTargetClassesPossible needs to be a classification");
      static_assert(compilerLearningTypeOrCountTargetClassesPossible <= k_cCompilerOptimizedTargetClassesMax, "We can't have this many items in a data pack.");

      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pInteractionCore->GetRuntimeLearningTypeOrCountTargetClasses();
      EBM_ASSERT(IsClassification(runtimeLearningTypeOrCountTargetClasses));
      EBM_ASSERT(runtimeLearningTypeOrCountTargetClasses <= k_cCompilerOptimizedTargetClassesMax);

      if(compilerLearningTypeOrCountTargetClassesPossible == runtimeLearningTypeOrCountTargetClasses) {
         BinInteractionDimensions<compilerLearningTypeOrCountTargetClassesPossible, 2>::Func(
            pInteractionCore,
            pFeatureGroup,
            aHistogramBuckets
#ifndef NDEBUG
            , aHistogramBucketsEndDebug
#endif // NDEBUG
         );
      } else {
         BinInteractionTarget<compilerLearningTypeOrCountTargetClassesPossible + 1>::Func(
            pInteractionCore,
            pFeatureGroup,
            aHistogramBuckets
#ifndef NDEBUG
            , aHistogramBucketsEndDebug
#endif // NDEBUG
         );
      }
   }
};

template<>
class BinInteractionTarget<k_cCompilerOptimizedTargetClassesMax + 1> final {
public:

   BinInteractionTarget() = delete; // this is a static class.  Do not construct

   INLINE_ALWAYS static void Func(
      InteractionCore * const pInteractionCore,
      const FeatureGroup * const pFeatureGroup,
      HistogramBucketBase * const aHistogramBuckets
#ifndef NDEBUG
      , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
   ) {
      static_assert(IsClassification(k_cCompilerOptimizedTargetClassesMax), "k_cCompilerOptimizedTargetClassesMax needs to be a classification");

      EBM_ASSERT(IsClassification(pInteractionCore->GetRuntimeLearningTypeOrCountTargetClasses()));
      EBM_ASSERT(k_cCompilerOptimizedTargetClassesMax < pInteractionCore->GetRuntimeLearningTypeOrCountTargetClasses());

      BinInteractionDimensions<k_dynamicClassification, 2>::Func(
         pInteractionCore,
         pFeatureGroup,
         aHistogramBuckets
#ifndef NDEBUG
         , aHistogramBucketsEndDebug
#endif // NDEBUG
      );
   }
};

extern void BinInteraction(
   InteractionCore * const pInteractionCore,
   const FeatureGroup * const pFeatureGroup,
   HistogramBucketBase * const aHistogramBuckets
#ifndef NDEBUG
   , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
) {
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pInteractionCore->GetRuntimeLearningTypeOrCountTargetClasses();

   if(IsClassification(runtimeLearningTypeOrCountTargetClasses)) {
      BinInteractionTarget<2>::Func(
         pInteractionCore,
         pFeatureGroup,
         aHistogramBuckets
#ifndef NDEBUG
         , aHistogramBucketsEndDebug
#endif // NDEBUG
      );
   } else {
      EBM_ASSERT(IsRegression(runtimeLearningTypeOrCountTargetClasses));
      BinInteractionDimensions<k_regression, 2>::Func(
         pInteractionCore,
         pFeatureGroup,
         aHistogramBuckets
#ifndef NDEBUG
         , aHistogramBucketsEndDebug
#endif // NDEBUG
      );
   }
}

} // DEFINED_ZONE_NAME
