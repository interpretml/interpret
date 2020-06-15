// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <ebm@koch.ninja>

#ifndef EBM_BOOSTING_STATE_H
#define EBM_BOOSTING_STATE_H

#include <stdlib.h> // malloc, realloc, free
#include <stddef.h> // size_t, ptrdiff_t
#include <limits> // numeric_limits

#include "ebm_native.h"
#include "EbmInternal.h"
// very independent includes
#include "Logging.h" // EBM_ASSERT & LOG
#include "RandomStream.h"
#include "SegmentedTensor.h"
// this depends on TreeNode pointers, but doesn't require the full definition of TreeNode
#include "CachedBoostingThreadResources.h"
// feature includes
#include "Feature.h"
// FeatureCombination.h depends on FeatureInternal.h
#include "FeatureCombination.h"
// dataset depends on features
#include "DataSetByFeatureCombination.h"
// samples is somewhat independent from datasets, but relies on an indirect coupling with them
#include "SamplingSet.h"

class EbmBoostingState {
   const ptrdiff_t m_runtimeLearningTypeOrCountTargetClasses;

   const size_t m_cFeatures;
   Feature * const m_aFeatures;

   const size_t m_cFeatureCombinations;
   FeatureCombination ** const m_apFeatureCombinations;

   DataSetByFeatureCombination m_trainingSet;
   DataSetByFeatureCombination m_validationSet;

   const size_t m_cSamplingSets;
   SamplingSet ** m_apSamplingSets;

   SegmentedTensor ** m_apCurrentModel;
   SegmentedTensor ** m_apBestModel;

   FloatEbmType m_bestModelMetric;

   // m_pSmallChangeToModelOverwriteSingleSamplingSet, m_pSmallChangeToModelAccumulatedFromSamplingSets and m_aEquivalentSplits should eventually move into 
   // the per-chunk class and we'll need a per-chunk m_randomStream that is initialized with it's own predictable seed 
   SegmentedTensor * const m_pSmallChangeToModelOverwriteSingleSamplingSet;
   SegmentedTensor * const m_pSmallChangeToModelAccumulatedFromSamplingSets;

   CachedBoostingThreadResources * m_pCachedThreadResources;

   RandomStream m_randomStream;

public:

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

   EBM_INLINE CachedBoostingThreadResources * GetCachedThreadResources() {
      return m_pCachedThreadResources;
   }

   EBM_INLINE RandomStream * GetRandomStream() {
      return &m_randomStream;
   }

   EBM_INLINE EbmBoostingState(
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses, 
      const size_t cFeatures, 
      const size_t cFeatureCombinations, 
      const size_t cSamplingSets, 
      const FloatEbmType * const optionalTempParams
   )
      : m_runtimeLearningTypeOrCountTargetClasses(runtimeLearningTypeOrCountTargetClasses)
      , m_cFeatures(cFeatures)
      , m_aFeatures(0 == cFeatures || IsMultiplyError(sizeof(Feature), cFeatures) ? nullptr : static_cast<Feature *>(malloc(sizeof(Feature) * cFeatures)))
      , m_cFeatureCombinations(cFeatureCombinations)
      , m_apFeatureCombinations(0 == cFeatureCombinations ? nullptr : FeatureCombination::AllocateFeatureCombinations(cFeatureCombinations))
      , m_trainingSet()
      , m_validationSet()
      , m_cSamplingSets(cSamplingSets)
      , m_apSamplingSets(nullptr)
      , m_apCurrentModel(nullptr)
      , m_apBestModel(nullptr)
      , m_bestModelMetric(FloatEbmType { std::numeric_limits<FloatEbmType>::max() })
      , m_pSmallChangeToModelOverwriteSingleSamplingSet(
         SegmentedTensor::Allocate(k_cDimensionsMax, GetVectorLength(runtimeLearningTypeOrCountTargetClasses)))
      , m_pSmallChangeToModelAccumulatedFromSamplingSets(
         SegmentedTensor::Allocate(k_cDimensionsMax, GetVectorLength(runtimeLearningTypeOrCountTargetClasses)))
      , m_pCachedThreadResources(nullptr)
      // we catch any errors in the constructor, so this should not be able to throw
      , m_randomStream()
   {
      // optionalTempParams isn't used by default.  It's meant to provide an easy way for python or other higher
      // level languages to pass EXPERIMENTAL temporary parameters easily to the C++ code.
      UNUSED(optionalTempParams);
   }

   EBM_INLINE ~EbmBoostingState() {
      LOG_0(TraceLevelInfo, "Entered ~EbmBoostingState");

      if(nullptr != m_pCachedThreadResources) {
         m_pCachedThreadResources->Free();
      }

      SamplingSet::FreeSamplingSets(m_cSamplingSets, m_apSamplingSets);

      FeatureCombination::FreeFeatureCombinations(m_cFeatureCombinations, m_apFeatureCombinations);

      free(m_aFeatures);

      DeleteSegmentedTensors(m_cFeatureCombinations, m_apCurrentModel);
      DeleteSegmentedTensors(m_cFeatureCombinations, m_apBestModel);
      SegmentedTensor::Free(m_pSmallChangeToModelOverwriteSingleSamplingSet);
      SegmentedTensor::Free(m_pSmallChangeToModelAccumulatedFromSamplingSets);

      LOG_0(TraceLevelInfo, "Exited ~EbmBoostingState");
   }

   static void DeleteSegmentedTensors(const size_t cFeatureCombinations, SegmentedTensor ** const apSegmentedTensors);
   static SegmentedTensor ** InitializeSegmentedTensors(
      const size_t cFeatureCombinations, 
      const FeatureCombination * const * const apFeatureCombinations, 
      const size_t cVectorLength
   );
   bool Initialize(
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

#endif // EBM_BOOSTING_STATE_H
