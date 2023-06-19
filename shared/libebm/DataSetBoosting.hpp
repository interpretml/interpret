// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef DATA_SET_BOOSTING_HPP
#define DATA_SET_BOOSTING_HPP

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t

#include "libebm.h" // ErrorEbm
#include "logging.h" // EBM_ASSERT
#include "common_c.h" // FloatFast
#include "bridge_c.h" // StorageDataType
#include "zones.h"

#include "InnerBag.hpp" // InnerBag

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

class Term;
struct DataSetBoosting;

struct DataSubsetBoosting final {
   friend DataSetBoosting;

   DataSubsetBoosting() = default; // preserve our POD status
   ~DataSubsetBoosting() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   inline void InitializeUnfailing() {
      m_cSamples = 0;
      m_aGradientsAndHessians = nullptr;
      m_aSampleScores = nullptr;
      m_aTargetData = nullptr;
      m_aaInputData = nullptr;
      m_aInnerBags = nullptr;
   }

   void Destruct(const size_t cTerms, const size_t cInnerBags);

   inline size_t GetCountSamples() const {
      return m_cSamples;
   }
   inline FloatFast * GetGradientsAndHessiansPointer() {
      return m_aGradientsAndHessians;
   }
   inline const FloatFast * GetGradientsAndHessiansPointer() const {
      EBM_ASSERT(nullptr != m_aGradientsAndHessians);
      return m_aGradientsAndHessians;
   }
   inline FloatFast * GetSampleScores() {
      return m_aSampleScores;
   }
   inline const void * GetTargetDataPointer() const {
      return m_aTargetData;
   }
   inline const StorageDataType * GetInputDataPointer(const size_t iTerm) const {
      EBM_ASSERT(nullptr != m_aaInputData);
      return m_aaInputData[iTerm];
   }
   inline const InnerBag * GetInnerBag(const size_t iBag) const {
      EBM_ASSERT(nullptr != m_aInnerBags);
      return &m_aInnerBags[iBag];
   }

private:
   size_t m_cSamples;
   FloatFast * m_aGradientsAndHessians;
   FloatFast * m_aSampleScores;
   void * m_aTargetData;
   StorageDataType ** m_aaInputData;
   InnerBag * m_aInnerBags;
};
static_assert(std::is_standard_layout<DataSubsetBoosting>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<DataSubsetBoosting>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<DataSubsetBoosting>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

struct DataSetBoosting final {
   DataSetBoosting() = default; // preserve our POD status
   ~DataSetBoosting() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   inline void InitializeUnfailing() {
      m_cSamples = 0;
      m_cSubsets = 0;
      m_aSubsets = nullptr;
      m_aBagWeightTotals = nullptr;
   }

   void Destruct(const size_t cTerms, const size_t cInnerBags);

   ErrorEbm Initialize(
      const ObjectiveWrapper * const pObjective,
      const size_t cSubsetItemsMax,
      const size_t cScores,
      const bool bAllocateGradients,
      const bool bAllocateHessians,
      const bool bAllocateSampleScores,
      const bool bAllocateTargetData,
      const unsigned char * const pDataSetShared,
      const size_t cSharedSamples,
      const BagEbm direction,
      const BagEbm * const aBag,
      const double * const aInitScores,
      const size_t cSetSamples,
      void * const rng,
      const size_t cInnerBags,
      const size_t cWeights,
      const IntEbm * const aiTermFeatures,
      const size_t cTerms,
      const Term * const * const apTerms
   );

   inline size_t GetCountSamples() const {
      return m_cSamples;
   }
   inline size_t GetCountSubsets() const {
      return m_cSubsets;
   }
   inline DataSubsetBoosting * GetSubsets() {
      EBM_ASSERT(nullptr != m_aSubsets);
      return m_aSubsets;
   }
   inline double GetBagWeightTotal(const size_t iBag) const {
      EBM_ASSERT(nullptr != m_aBagWeightTotals);
      return m_aBagWeightTotals[iBag];
   }

private:

   ErrorEbm InitializeGradientsAndHessians(
      const ObjectiveWrapper * const pObjective, 
      const size_t cScores,
      const bool bAllocateHessians
   );

   ErrorEbm InitializeSampleScores(
      const size_t cScores,
      const BagEbm direction,
      const BagEbm * const aBag,
      const double * const aInitScores
   );

   ErrorEbm InitializeTargetData(
      const unsigned char * const pDataSetShared,
      const BagEbm direction,
      const BagEbm * const aBag
   );

   ErrorEbm InitializeInputData(
      const unsigned char * const pDataSetShared,
      const size_t cSharedSamples,
      const BagEbm direction,
      const BagEbm * const aBag,
      const IntEbm * const aiTermFeatures,
      const size_t cTerms,
      const Term * const * const apTerms
   );

   ErrorEbm InitializeBags(
      const unsigned char * const pDataSetShared,
      const BagEbm direction,
      const BagEbm * const aBag,
      void * const rng,
      const size_t cInnerBags,
      const size_t cWeights
   );

   size_t m_cSamples;
   size_t m_cSubsets;
   DataSubsetBoosting * m_aSubsets;
   double * m_aBagWeightTotals;
};
static_assert(std::is_standard_layout<DataSetBoosting>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<DataSetBoosting>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<DataSetBoosting>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

} // DEFINED_ZONE_NAME

#endif // DATA_SET_BOOSTING_HPP
