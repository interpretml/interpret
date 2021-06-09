// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <ebm@koch.ninja>

#ifndef BOOSTER_H
#define BOOSTER_H

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t
#include <limits> // numeric_limits

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "ebm_internal.hpp"

#include "RandomStream.hpp"
#include "SegmentedTensor.hpp"
// feature includes
#include "Feature.hpp"
// FeatureGroup.h depends on FeatureInternal.h
#include "FeatureGroup.hpp"
// dataset depends on features
#include "DataSetBoosting.hpp"
// samples is somewhat independent from datasets, but relies on an indirect coupling with them
#include "SamplingSet.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

class Booster final {
   ptrdiff_t m_runtimeLearningTypeOrCountTargetClasses;

   size_t m_cFeatureAtomics;
   FeatureAtomic * m_aFeatureAtomics;

   size_t m_cFeatureGroups;
   FeatureGroup ** m_apFeatureGroups;

   DataFrameBoosting m_trainingSet;
   DataFrameBoosting m_validationSet;

   size_t m_cSamplingSets;
   SamplingSet ** m_apSamplingSets;
   FloatEbmType m_validationWeightTotal;
   FloatEbmType * m_aValidationWeights;

   SegmentedTensor ** m_apCurrentModel;
   SegmentedTensor ** m_apBestModel;

   FloatEbmType m_bestModelMetric;

   size_t m_cBytesArrayEquivalentSplitMax;

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

      m_cFeatureAtomics = 0;
      m_aFeatureAtomics = nullptr;

      m_cFeatureGroups = 0;
      m_apFeatureGroups = nullptr;

      m_trainingSet.InitializeZero();
      m_validationSet.InitializeZero();

      m_cSamplingSets = 0;
      m_apSamplingSets = nullptr;
      m_validationWeightTotal = 0;
      m_aValidationWeights = nullptr;

      m_apCurrentModel = nullptr;
      m_apBestModel = nullptr;

      m_bestModelMetric = FloatEbmType { 0 };

      m_cBytesArrayEquivalentSplitMax = size_t { 0 };
   }

   INLINE_ALWAYS ptrdiff_t GetRuntimeLearningTypeOrCountTargetClasses() const {
      return m_runtimeLearningTypeOrCountTargetClasses;
   }

   INLINE_ALWAYS size_t GetCountBytesArrayEquivalentSplitMax() const {
      return m_cBytesArrayEquivalentSplitMax;
   }

   INLINE_ALWAYS size_t GetCountFeatureGroups() const {
      return m_cFeatureGroups;
   }

   INLINE_ALWAYS FeatureGroup * const * GetFeatureGroups() const {
      return m_apFeatureGroups;
   }

   INLINE_ALWAYS DataFrameBoosting * GetTrainingSet() {
      return &m_trainingSet;
   }

   INLINE_ALWAYS DataFrameBoosting * GetValidationSet() {
      return &m_validationSet;
   }

   INLINE_ALWAYS size_t GetCountSamplingSets() const {
      return m_cSamplingSets;
   }

   INLINE_ALWAYS const SamplingSet * const * GetSamplingSets() const {
      return m_apSamplingSets;
   }

   INLINE_ALWAYS FloatEbmType GetValidationWeightTotal() const {
      return m_validationWeightTotal;
   }

   INLINE_ALWAYS const FloatEbmType * GetValidationWeights() const {
      return m_aValidationWeights;
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
      const BoolEbmType * const aFeatureAtomicsCategorical,
      const IntEbmType * const aFeatureAtomicsBinCount,
      const IntEbmType * const aFeatureGroupsDimensionCounts,
      const IntEbmType * const aFeatureGroupsFeatureAtomicIndexes, 
      const size_t cTrainingSamples, 
      const void * const aTrainingTargets, 
      const IntEbmType * const aTrainingBinnedData, 
      const FloatEbmType * const aTrainingWeights,
      const FloatEbmType * const aTrainingPredictorScores,
      const size_t cValidationSamples, 
      const void * const aValidationTargets, 
      const IntEbmType * const aValidationBinnedData, 
      const FloatEbmType * const aValidationWeights,
      const FloatEbmType * const aValidationPredictorScores
   );
};
static_assert(std::is_standard_layout<Booster>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<Booster>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<Booster>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

} // DEFINED_ZONE_NAME

#endif // BOOSTER_H
