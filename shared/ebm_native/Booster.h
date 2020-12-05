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
// FeatureGroup.h depends on FeatureInternal.h
#include "FeatureGroup.h"
// dataset depends on features
#include "DataSetBoosting.h"
// samples is somewhat independent from datasets, but relies on an indirect coupling with them
#include "SamplingSet.h"

class Booster final {
   ptrdiff_t m_runtimeLearningTypeOrCountTargetClasses;

   size_t m_cFeatures;
   Feature * m_aFeatures;

   size_t m_cFeatureGroups;
   FeatureGroup ** m_apFeatureGroups;

   DataSetByFeatureGroup m_trainingSet;
   DataSetByFeatureGroup m_validationSet;

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

   static void DeleteSegmentedTensors(const size_t cFeatureGroups, SegmentedTensor ** const apSegmentedTensors);

   static SegmentedTensor ** InitializeSegmentedTensors(
      const size_t cFeatureGroups,
      const FeatureGroup * const * const apFeatureGroups,
      const size_t cVectorLength
   );

public:

   Booster() = default; // preserve our POD status
   ~Booster() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   INLINE_ALWAYS void InitializeZero() {
      m_runtimeLearningTypeOrCountTargetClasses = 0;

      m_cFeatures = 0;
      m_aFeatures = nullptr;

      m_cFeatureGroups = 0;
      m_apFeatureGroups = nullptr;

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

   INLINE_ALWAYS ptrdiff_t GetRuntimeLearningTypeOrCountTargetClasses() const {
      return m_runtimeLearningTypeOrCountTargetClasses;
   }

   INLINE_ALWAYS size_t GetCountFeatureGroups() const {
      return m_cFeatureGroups;
   }

   INLINE_ALWAYS FeatureGroup * const * GetFeatureGroups() const {
      return m_apFeatureGroups;
   }

   INLINE_ALWAYS DataSetByFeatureGroup * GetTrainingSet() {
      return &m_trainingSet;
   }

   INLINE_ALWAYS DataSetByFeatureGroup * GetValidationSet() {
      return &m_validationSet;
   }

   INLINE_ALWAYS size_t GetCountSamplingSets() const {
      return m_cSamplingSets;
   }

   INLINE_ALWAYS const SamplingSet * const * GetSamplingSets() const {
      return m_apSamplingSets;
   }

   INLINE_ALWAYS SegmentedTensor * const * GetCurrentModel() const {
      return m_apCurrentModel;
   }

   INLINE_ALWAYS SegmentedTensor * const * GetBestModel() const {
      return m_apBestModel;
   }

   INLINE_ALWAYS FloatEbmType GetBestModelMetric() const {
      return m_bestModelMetric;
   }

   INLINE_ALWAYS void SetBestModelMetric(const FloatEbmType bestModelMetric) {
      m_bestModelMetric = bestModelMetric;
   }

   INLINE_ALWAYS SegmentedTensor * GetSmallChangeToModelOverwriteSingleSamplingSet() {
      return m_pSmallChangeToModelOverwriteSingleSamplingSet;
   }

   INLINE_ALWAYS SegmentedTensor * GetSmallChangeToModelAccumulatedFromSamplingSets() {
      return m_pSmallChangeToModelAccumulatedFromSamplingSets;
   }

   INLINE_ALWAYS CachedBoostingThreadResources * GetCachedThreadResources() const {
      return m_pCachedThreadResources;
   }

   INLINE_ALWAYS RandomStream * GetRandomStream() {
      return &m_randomStream;
   }

   static void Free(Booster * const pBooster);

   static Booster * Allocate(
      const SeedEbmType randomSeed,
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
      const size_t cFeatures,
      const size_t cFeatureGroups,
      const size_t cSamplingSets,
      const FloatEbmType * const optionalTempParams,
      const EbmNativeFeature * const aFeatures,
      const EbmNativeFeatureGroup * const aFeatureGroups, 
      const IntEbmType * featureGroupIndexes, 
      const size_t cTrainingSamples, 
      const void * const aTrainingTargets, 
      const IntEbmType * const aTrainingBinnedData, 
      const FloatEbmType * const aTrainingPredictorScores, 
      const size_t cValidationSamples, 
      const void * const aValidationTargets, 
      const IntEbmType * const aValidationBinnedData, 
      const FloatEbmType * const aValidationPredictorScores
   );
};
static_assert(std::is_standard_layout<Booster>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<Booster>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<Booster>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

#endif // EBM_BOOSTING_STATE_H
