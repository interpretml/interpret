// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <ebm@koch.ninja>

#ifndef BOOSTER_CORE_HPP
#define BOOSTER_CORE_HPP

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t
#include <limits> // numeric_limits
#include <atomic>

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

class BoosterShell;

class BoosterCore final {

   // std::atomic_size_t used to be standard layout and trivial, but the C++ standard comitee judged that an error
   // and revoked the trivial nature of the class.  So, this means our BoosterCore class needs to have a constructor 
   // and destructor
   // https://stackoverflow.com/questions/48794325/why-stdatomic-is-not-trivial-type-in-only-visual-c
   // https://stackoverflow.com/questions/41308372/stdatomic-for-built-in-types-non-lock-free-vs-trivial-destructor
   std::atomic_size_t m_REFERENCE_COUNT;

   ptrdiff_t m_runtimeLearningTypeOrCountTargetClasses;

   size_t m_cFeatures;
   Feature * m_aFeatures;

   size_t m_cFeatureGroups;
   FeatureGroup ** m_apFeatureGroups;

   size_t m_cSamplingSets;
   SamplingSet ** m_apSamplingSets;
   FloatEbmType m_validationWeightTotal;
   FloatEbmType * m_aValidationWeights;

   CompressibleTensor ** m_apCurrentModel;
   CompressibleTensor ** m_apBestModel;

   FloatEbmType m_bestModelMetric;

   size_t m_cBytesArrayEquivalentSplitMax;

   RandomStream m_randomStream;

   DataSetBoosting m_trainingSet;
   DataSetBoosting m_validationSet;

   static void DeleteCompressibleTensors(const size_t cFeatureGroups, CompressibleTensor ** const apCompressibleTensors);

   static ErrorEbmType InitializeCompressibleTensors(
      const size_t cFeatureGroups,
      const FeatureGroup * const * const apFeatureGroups,
      const size_t cVectorLength,
      CompressibleTensor *** papCompressibleTensorsOut
   );

   INLINE_ALWAYS ~BoosterCore() {
      // this only gets called after our reference count has been decremented to zero

      m_trainingSet.Destruct();
      m_validationSet.Destruct();

      SamplingSet::FreeSamplingSets(m_cSamplingSets, m_apSamplingSets);
      free(m_aValidationWeights);

      FeatureGroup::FreeFeatureGroups(m_cFeatureGroups, m_apFeatureGroups);

      free(m_aFeatures);

      DeleteCompressibleTensors(m_cFeatureGroups, m_apCurrentModel);
      DeleteCompressibleTensors(m_cFeatureGroups, m_apBestModel);
   };

   WARNING_PUSH
   ATTRIBUTE_WARNING_DISABLE_UNINITIALIZED_MEMBER
   INLINE_ALWAYS BoosterCore() noexcept :
      m_REFERENCE_COUNT(1), // we're not visible on any other thread yet, so no synchronization required
      m_runtimeLearningTypeOrCountTargetClasses(0),
      m_cFeatures(0),
      m_aFeatures(nullptr),
      m_cFeatureGroups(0),
      m_apFeatureGroups(nullptr),
      m_cSamplingSets(0),
      m_apSamplingSets(nullptr),
      m_validationWeightTotal(0),
      m_aValidationWeights(nullptr),
      m_apCurrentModel(nullptr),
      m_apBestModel(nullptr),
      m_bestModelMetric(0),
      m_cBytesArrayEquivalentSplitMax(0)
   {
      m_trainingSet.InitializeZero();
      m_validationSet.InitializeZero();
   }
   WARNING_POP

public:

   INLINE_ALWAYS void AddReferenceCount() {
      // incrementing reference counts can be relaxed memory order since we're guaranteed to be above 1, 
      // so no result will change our behavior below
      // https://www.boost.org/doc/libs/1_59_0/doc/html/atomic/usage_examples.html
      m_REFERENCE_COUNT.fetch_add(1, std::memory_order_relaxed);
   };

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

   INLINE_ALWAYS DataSetBoosting * GetTrainingSet() {
      return &m_trainingSet;
   }

   INLINE_ALWAYS DataSetBoosting * GetValidationSet() {
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

   INLINE_ALWAYS CompressibleTensor * const * GetCurrentModel() const {
      return m_apCurrentModel;
   }

   INLINE_ALWAYS CompressibleTensor * const * GetBestModel() const {
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

   static void Free(BoosterCore * const pBoosterCore);

   static ErrorEbmType Create(
      BoosterShell * const pBoosterShell,
      const SeedEbmType randomSeed,
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
      const size_t cFeatures,
      const size_t cFeatureGroups,
      const size_t cSamplingSets,
      const FloatEbmType * const optionalTempParams,
      const BoolEbmType * const aFeaturesCategorical,
      const IntEbmType * const aFeaturesBinCount,
      const IntEbmType * const aFeatureGroupsDimensionCounts,
      const IntEbmType * const aFeatureGroupsFeatureIndexes, 
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

} // DEFINED_ZONE_NAME

#endif // BOOSTER_CORE_HPP
