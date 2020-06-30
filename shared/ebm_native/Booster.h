// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <ebm@koch.ninja>

#ifndef EBM_BOOSTING_STATE_H
#define EBM_BOOSTING_STATE_H

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t
#include <limits> // numeric_limits

#include "ebm_native.h"
#include "EbmInternal.h"
// very independent includes
#include "Logging.h" // EBM_ASSERT & LOG
#include "RandomStream.h"
#include "SegmentedTensor.h"
// this depends on TreeNode pointers, but doesn't require the full definition of TreeNode
#include "CachedThreadResourcesBoosting.h"
// feature includes
#include "FeatureAtomic.h"
// FeatureCombination.h depends on FeatureInternal.h
#include "FeatureGroup.h"
// dataset depends on features
#include "DataSetBoosting.h"
// samples is somewhat independent from datasets, but relies on an indirect coupling with them
#include "SamplingSet.h"

class EbmBoostingState final {
   ptrdiff_t m_runtimeLearningTypeOrCountTargetClasses;

   size_t m_cFeatures;
   Feature * m_aFeatures;

   size_t m_cFeatureCombinations;
   FeatureCombination ** m_apFeatureCombinations;

   DataSetByFeatureCombination m_trainingSet;
   DataSetByFeatureCombination m_validationSet;

   size_t m_cSamplingSets;
   SamplingSet ** m_apSamplingSets;

   SegmentedTensor ** m_apCurrentModel;
   SegmentedTensor ** m_apBestModel;

   FloatEbmType m_bestModelMetric;

   // m_pSmallChangeToModelOverwriteSingleSamplingSet, m_pSmallChangeToModelAccumulatedFromSamplingSets and m_aEquivalentSplits should eventually move into 
   // the per-chunk class and we'll need a per-chunk m_randomStream that is initialized with it's own predictable seed 
   SegmentedTensor * m_pSmallChangeToModelOverwriteSingleSamplingSet;
   SegmentedTensor * m_pSmallChangeToModelAccumulatedFromSamplingSets;

   CachedBoostingThreadResources * m_pCachedThreadResources;

   RandomStream m_randomStream;

   static void DeleteSegmentedTensors(const size_t cFeatureCombinations, SegmentedTensor ** const apSegmentedTensors);

   static SegmentedTensor ** InitializeSegmentedTensors(
      const size_t cFeatureCombinations,
      const FeatureCombination * const * const apFeatureCombinations,
      const size_t cVectorLength
   );

public:

   EbmBoostingState() = default; // preserve our POD status
   ~EbmBoostingState() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   EBM_INLINE void InitializeZero() {
      m_runtimeLearningTypeOrCountTargetClasses = 0;

      m_cFeatures = 0;
      m_aFeatures = nullptr;

      m_cFeatureCombinations = 0;
      m_apFeatureCombinations = nullptr;

      m_trainingSet.InitializeZero();
      m_validationSet.InitializeZero();

      m_cSamplingSets = 0;
      m_apSamplingSets = nullptr;

      m_apCurrentModel = nullptr;
      m_apBestModel = nullptr;

      m_bestModelMetric = FloatEbmType { 0 };

      m_pSmallChangeToModelOverwriteSingleSamplingSet = nullptr;
      m_pSmallChangeToModelAccumulatedFromSamplingSets = nullptr;

      m_pCachedThreadResources = nullptr;
   }

   EBM_INLINE ptrdiff_t GetRuntimeLearningTypeOrCountTargetClasses() const {
      return m_runtimeLearningTypeOrCountTargetClasses;
   }

   EBM_INLINE size_t GetCountFeatureCombinations() const {
      return m_cFeatureCombinations;
   }

   EBM_INLINE FeatureCombination * const * GetFeatureCombinations() const {
      return m_apFeatureCombinations;
   }

   EBM_INLINE DataSetByFeatureCombination * GetTrainingSet() {
      return &m_trainingSet;
   }

   EBM_INLINE DataSetByFeatureCombination * GetValidationSet() {
      return &m_validationSet;
   }

   EBM_INLINE size_t GetCountSamplingSets() const {
      return m_cSamplingSets;
   }

   EBM_INLINE const SamplingSet * const * GetSamplingSets() const {
      return m_apSamplingSets;
   }

   EBM_INLINE SegmentedTensor * const * GetCurrentModel() const {
      return m_apCurrentModel;
   }

   EBM_INLINE SegmentedTensor * const * GetBestModel() const {
      return m_apBestModel;
   }

   EBM_INLINE FloatEbmType GetBestModelMetric() const {
      return m_bestModelMetric;
   }

   EBM_INLINE void SetBestModelMetric(const FloatEbmType bestModelMetric) {
      m_bestModelMetric = bestModelMetric;
   }

   EBM_INLINE SegmentedTensor * GetSmallChangeToModelOverwriteSingleSamplingSet() {
      return m_pSmallChangeToModelOverwriteSingleSamplingSet;
   }

   EBM_INLINE SegmentedTensor * GetSmallChangeToModelAccumulatedFromSamplingSets() {
      return m_pSmallChangeToModelAccumulatedFromSamplingSets;
   }

   EBM_INLINE CachedBoostingThreadResources * GetCachedThreadResources() const {
      return m_pCachedThreadResources;
   }

   EBM_INLINE RandomStream * GetRandomStream() {
      return &m_randomStream;
   }

   static void Free(EbmBoostingState * const pBoostingState);

   static EbmBoostingState * Allocate(
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
      const size_t cFeatures,
      const size_t cFeatureCombinations,
      const size_t cSamplingSets,
      const FloatEbmType * const optionalTempParams,
      const EbmNativeFeature * const aFeatures,
      const EbmNativeFeatureCombination * const aFeatureCombinations, 
      const IntEbmType * featureCombinationIndexes, 
      const size_t cTrainingInstances, 
      const void * const aTrainingTargets, 
      const IntEbmType * const aTrainingBinnedData, 
      const FloatEbmType * const aTrainingPredictorScores, 
      const size_t cValidationInstances, 
      const void * const aValidationTargets, 
      const IntEbmType * const aValidationBinnedData, 
      const FloatEbmType * const aValidationPredictorScores,
      const IntEbmType randomSeed
   );
};
static_assert(std::is_standard_layout<EbmBoostingState>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<EbmBoostingState>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<EbmBoostingState>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

#endif // EBM_BOOSTING_STATE_H
