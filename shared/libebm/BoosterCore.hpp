// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <ebm@koch.ninja>

#ifndef BOOSTER_CORE_HPP
#define BOOSTER_CORE_HPP

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t
#include <limits> // numeric_limits
#include <atomic>

#include "libebm.h" // ErrorEbm
#include "unzoned.h"
#include "bridge.h" // ObjectiveWrapper
#include "zones.h"

#include "ebm_internal.hpp" // FloatMain
#include "DataSetBoosting.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

class RandomDeterministic;
class FeatureBoosting;
class Term;
struct InnerBag;
class Tensor;

class BoosterCore final {

   // std::atomic_size_t used to be standard layout and trivial, but the C++ standard comitee judged that an error
   // and revoked the trivial nature of the class.  So, this means our BoosterCore class needs to have a constructor 
   // and destructor
   // https://stackoverflow.com/questions/48794325/why-stdatomic-is-not-trivial-type-in-only-visual-c
   // https://stackoverflow.com/questions/41308372/stdatomic-for-built-in-types-non-lock-free-vs-trivial-destructor
   std::atomic_size_t m_REFERENCE_COUNT;

   ptrdiff_t m_cClasses;

   size_t m_cFeatures;
   FeatureBoosting * m_aFeatures;

   size_t m_cTerms;
   Term ** m_apTerms;

   size_t m_cInnerBags;

   Tensor ** m_apCurrentTermTensors;
   Tensor ** m_apBestTermTensors;

   double m_bestModelMetric;

   size_t m_cBytesFastBins;
   size_t m_cBytesMainBins;

   size_t m_cBytesSplitPositions;
   size_t m_cBytesTreeNodes;

   DataSetBoosting m_trainingSet;
   DataSetBoosting m_validationSet;

   ObjectiveWrapper m_objectiveCpu;
   ObjectiveWrapper m_objectiveSIMD;

   static void DeleteTensors(const size_t cTerms, Tensor ** const apTensors);

   static ErrorEbm InitializeTensors(
      const size_t cTerms,
      const Term * const * const apTerms,
      const size_t cScores,
      Tensor *** papTensorsOut
   );

   ~BoosterCore();

   inline BoosterCore() noexcept :
      m_REFERENCE_COUNT(1), // we're not visible on any other thread yet, so no synchronization required
      m_cClasses(0),
      m_cFeatures(0),
      m_aFeatures(nullptr),
      m_cTerms(0),
      m_apTerms(nullptr),
      m_cInnerBags(0),
      m_apCurrentTermTensors(nullptr),
      m_apBestTermTensors(nullptr),
      m_bestModelMetric(std::numeric_limits<double>::infinity()),
      m_cBytesFastBins(0),
      m_cBytesMainBins(0),
      m_cBytesSplitPositions(0),
      m_cBytesTreeNodes(0)
   {
      m_trainingSet.SafeInitDataSetBoosting();
      m_validationSet.SafeInitDataSetBoosting();
      InitializeObjectiveWrapperUnfailing(&m_objectiveCpu);
      InitializeObjectiveWrapperUnfailing(&m_objectiveSIMD);
   }

public:

   inline void AddReferenceCount() {
      // incrementing reference counts can be relaxed memory order since we're guaranteed to be above 1, 
      // so no result will change our behavior below
      // https://www.boost.org/doc/libs/1_59_0/doc/html/atomic/usage_examples.html
      m_REFERENCE_COUNT.fetch_add(1, std::memory_order_relaxed);
   };

   inline ptrdiff_t GetCountClasses() const {
      return m_cClasses;
   }

   inline size_t GetCountBytesFastBins() const {
      return m_cBytesFastBins;
   }

   inline size_t GetCountBytesMainBins() const {
      return m_cBytesMainBins;
   }

   inline size_t GetCountBytesSplitPositions() const {
      return m_cBytesSplitPositions;
   }

   inline size_t GetCountBytesTreeNodes() const {
      return m_cBytesTreeNodes;
   }

   inline size_t GetCountTerms() const {
      return m_cTerms;
   }

   inline Term * const * GetTerms() const {
      return m_apTerms;
   }

   inline DataSetBoosting * GetTrainingSet() {
      return &m_trainingSet;
   }

   inline DataSetBoosting * GetValidationSet() {
      return &m_validationSet;
   }

   inline size_t GetCountInnerBags() const {
      return m_cInnerBags;
   }

   inline Tensor * const * GetCurrentModel() const {
      return m_apCurrentTermTensors;
   }

   inline Tensor * const * GetBestModel() const {
      return m_apBestTermTensors;
   }

   inline double GetBestModelMetric() const {
      return m_bestModelMetric;
   }

   inline void SetBestModelMetric(const double bestModelMetric) {
      m_bestModelMetric = bestModelMetric;
   }

   static void Free(BoosterCore * const pBoosterCore);

   static ErrorEbm Create(
      void * const rng,
      const size_t cTerms,
      const size_t cInnerBags,
      const double * const experimentalParams,
      const IntEbm * const acTermDimensions,
      const IntEbm * const aiTermFeatures,
      const unsigned char * const pDataSetShared,
      const BagEbm * const aBag,
      const double * const aInitScores,
      const CreateBoosterFlags flags,
      const char * const sObjective,
      BoosterCore ** const ppBoosterCoreOut
   );

   ErrorEbm InitializeBoosterGradientsAndHessians(
      void * const aMulticlassMidwayTemp,
      FloatScore * const aUpdateScores
   );

   inline double FinishMetric(const double metricSum) {
      EBM_ASSERT(nullptr != m_objectiveCpu.m_pObjective);
      return FinishMetricC(&m_objectiveCpu, metricSum);
   }

   inline BoolEbm CheckTargets(const size_t c, const void * const aTargets) const noexcept {
      EBM_ASSERT(nullptr != aTargets);
      EBM_ASSERT(nullptr != m_objectiveCpu.m_pObjective);
      return CheckTargetsC(&m_objectiveCpu, c, aTargets);
   }

   inline bool IsRmse() {
      EBM_ASSERT(nullptr != m_objectiveCpu.m_pObjective);
      return EBM_FALSE != m_objectiveCpu.m_bRmse;
   }

   inline bool IsHessian() {
      EBM_ASSERT(nullptr != m_objectiveCpu.m_pObjective);
      return EBM_FALSE != m_objectiveCpu.m_bObjectiveHasHessian;
   }

   inline double LearningRateAdjustmentDifferentialPrivacy() const noexcept {
      EBM_ASSERT(nullptr != m_objectiveCpu.m_pObjective);
      return m_objectiveCpu.m_learningRateAdjustmentDifferentialPrivacy;
   }

   inline double LearningRateAdjustmentGradientBoosting() const noexcept {
      EBM_ASSERT(nullptr != m_objectiveCpu.m_pObjective);
      return m_objectiveCpu.m_learningRateAdjustmentGradientBoosting;
   }

   inline double LearningRateAdjustmentHessianBoosting() const noexcept {
      EBM_ASSERT(nullptr != m_objectiveCpu.m_pObjective);
      return m_objectiveCpu.m_learningRateAdjustmentHessianBoosting;
   }

   inline double GainAdjustmentGradientBoosting() const noexcept {
      EBM_ASSERT(nullptr != m_objectiveCpu.m_pObjective);
      return m_objectiveCpu.m_gainAdjustmentGradientBoosting;
   }

   inline double GainAdjustmentHessianBoosting() const noexcept {
      EBM_ASSERT(nullptr != m_objectiveCpu.m_pObjective);
      return m_objectiveCpu.m_gainAdjustmentHessianBoosting;
   }

   inline double GradientConstant() {
      EBM_ASSERT(nullptr != m_objectiveCpu.m_pObjective);
      return m_objectiveCpu.m_gradientConstant;
   }

   inline double HessianConstant() {
      EBM_ASSERT(nullptr != m_objectiveCpu.m_pObjective);
      return m_objectiveCpu.m_hessianConstant;
   }

   inline BoolEbm MaximizeMetric() {
      EBM_ASSERT(nullptr != m_objectiveCpu.m_pObjective);
      return m_objectiveCpu.m_bMaximizeMetric;
   }
};

} // DEFINED_ZONE_NAME

#endif // BOOSTER_CORE_HPP
