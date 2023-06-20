// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef DATA_SET_BOOSTING_HPP
#define DATA_SET_BOOSTING_HPP

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

   inline void SafeInitDataSubsetBoosting() {
      m_cSamples = 0;
      m_aGradHess = nullptr;
      m_aSampleScores = nullptr;
      m_aTargetData = nullptr;
      m_aaTermData = nullptr;
      m_aInnerBags = nullptr;
   }

   void DestructDataSubsetBoosting(const size_t cTerms, const size_t cInnerBags);

   inline size_t GetCountSamples() const {
      return m_cSamples;
   }
   inline FloatFast * GetGradHess() {
      return m_aGradHess;
   }
   inline FloatFast * GetSampleScores() {
      return m_aSampleScores;
   }
   inline const void * GetTargetData() const {
      return m_aTargetData;
   }
   inline const StorageDataType * GetTermData(const size_t iTerm) const {
      EBM_ASSERT(nullptr != m_aaTermData);
      return m_aaTermData[iTerm];
   }
   inline const InnerBag * GetInnerBag(const size_t iBag) const {
      EBM_ASSERT(nullptr != m_aInnerBags);
      return &m_aInnerBags[iBag];
   }

private:

   size_t m_cSamples;
   FloatFast * m_aGradHess;
   FloatFast * m_aSampleScores;
   void * m_aTargetData;
   StorageDataType ** m_aaTermData;
   InnerBag * m_aInnerBags;
};
static_assert(std::is_standard_layout<DataSubsetBoosting>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<DataSubsetBoosting>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");

struct DataSetBoosting final {
   DataSetBoosting() = default; // preserve our POD status
   ~DataSetBoosting() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   inline void SafeInitDataSetBoosting() {
      m_cSamples = 0;
      m_cSubsets = 0;
      m_aSubsets = nullptr;
      m_aBagWeightTotals = nullptr;
   }

   ErrorEbm InitDataSetBoosting(
      const bool bAllocateGradients,
      const bool bAllocateHessians,
      const bool bAllocateSampleScores,
      const bool bAllocateTargetData,
      void * const rng,
      const size_t cScores,
      const size_t cSubsetItemsMax,
      const ObjectiveWrapper * const pObjective,
      const unsigned char * const pDataSetShared,
      const BagEbm direction,
      const size_t cSharedSamples,
      const BagEbm * const aBag,
      const double * const aInitScores,
      const size_t cIncludedSamples,
      const size_t cInnerBags,
      const size_t cWeights,
      const size_t cTerms,
      const Term * const * const apTerms,
      const IntEbm * const aiTermFeatures
   );

   void DestructDataSetBoosting(const size_t cTerms, const size_t cInnerBags);

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

   ErrorEbm InitGradHess(
      const bool bAllocateHessians,
      const size_t cScores,
      const ObjectiveWrapper * const pObjective
   );

   ErrorEbm InitSampleScores(
      const size_t cScores,
      const BagEbm direction,
      const BagEbm * const aBag,
      const double * const aInitScores
   );

   ErrorEbm InitTargetData(
      const unsigned char * const pDataSetShared,
      const BagEbm direction,
      const BagEbm * const aBag
   );

   ErrorEbm InitTermData(
      const unsigned char * const pDataSetShared,
      const BagEbm direction,
      const size_t cSharedSamples,
      const BagEbm * const aBag,
      const size_t cTerms,
      const Term * const * const apTerms,
      const IntEbm * const aiTermFeatures
   );

   ErrorEbm InitBags(
      void * const rng,
      const unsigned char * const pDataSetShared,
      const BagEbm direction,
      const BagEbm * const aBag,
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

} // DEFINED_ZONE_NAME

#endif // DATA_SET_BOOSTING_HPP
