// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef DATA_SET_BOOSTING_HPP
#define DATA_SET_BOOSTING_HPP

#include <stddef.h> // size_t, ptrdiff_t

#include "libebm.h" // ErrorEbm
#include "logging.h" // EBM_ASSERT
#include "unzoned.h"

#include "bridge.h" // UIntMain

#include "DataSetInnerBag.hpp" // DataSetInnerBag
#include "SubsetInnerBag.hpp" // SubsetInnerBag
#include "TermInnerBag.hpp" // TermInnerBag

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
   void* operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete(void*) = delete; // we only use malloc/free in this library

   inline void SafeInitDataSubsetBoosting() {
      m_cSamples = 0;
      m_pObjective = nullptr;
      m_aGradHess = nullptr;
      m_aSampleScores = nullptr;
      m_aTargetData = nullptr;
      m_aaTermData = nullptr;
      m_aSubsetInnerBags = nullptr;
   }

   void DestructDataSubsetBoosting(const size_t cTerms, const size_t cInnerBags);

   inline size_t GetCountSamples() const { return m_cSamples; }

   inline const ObjectiveWrapper* GetObjectiveWrapper() const {
      EBM_ASSERT(nullptr != m_pObjective);
      return m_pObjective;
   }

   inline ErrorEbm ObjectiveApplyUpdate(ApplyUpdateBridge* const pData) {
      EBM_ASSERT(nullptr != pData);
      EBM_ASSERT(nullptr != m_pObjective);
      EBM_ASSERT(nullptr != m_pObjective->m_pApplyUpdateC);
      EBM_ASSERT(0 == m_cSamples % m_pObjective->m_cSIMDPack);
      return (*m_pObjective->m_pApplyUpdateC)(m_pObjective, pData);
   }

   inline ErrorEbm BinSumsBoosting(BinSumsBoostingBridge* const pParams) {
      EBM_ASSERT(nullptr != pParams);
      EBM_ASSERT(nullptr != m_pObjective);
      EBM_ASSERT(nullptr != m_pObjective->m_pBinSumsBoostingC);
      EBM_ASSERT(0 == m_cSamples % m_pObjective->m_cSIMDPack);
      return (*m_pObjective->m_pBinSumsBoostingC)(m_pObjective, pParams);
   }

   inline void* GetGradHess() { return m_aGradHess; }

   inline void* GetSampleScores() { return m_aSampleScores; }

   inline const void* GetTargetData() const { return m_aTargetData; }

   inline const void* GetTermData(const size_t iTerm) const {
      EBM_ASSERT(nullptr != m_aaTermData);
      return m_aaTermData[iTerm];
   }

   inline const SubsetInnerBag* GetSubsetInnerBag(const size_t iBag) const {
      EBM_ASSERT(nullptr != m_aSubsetInnerBags);
      return &m_aSubsetInnerBags[iBag];
   }

 private:
   size_t m_cSamples;
   const ObjectiveWrapper* m_pObjective;
   void* m_aGradHess;
   void* m_aSampleScores;
   void* m_aTargetData;
   void** m_aaTermData;
   SubsetInnerBag* m_aSubsetInnerBags;
};
static_assert(std::is_standard_layout<DataSubsetBoosting>::value,
      "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<DataSubsetBoosting>::value,
      "We use memcpy in several places, so disallow non-trivial types in general");

struct DataSetBoosting final {
   DataSetBoosting() = default; // preserve our POD status
   ~DataSetBoosting() = default; // preserve our POD status
   void* operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete(void*) = delete; // we only use malloc/free in this library

   inline void SafeInitDataSetBoosting() {
      m_cSamples = 0;
      m_cSubsets = 0;
      m_aSubsets = nullptr;
      m_aDataSetInnerBags = nullptr;
      m_aOriginalWeights = nullptr;
   }

   ErrorEbm InitDataSetBoosting(const bool bAllocateGradients,
         const bool bAllocateHessians,
         const bool bAllocateSampleScores,
         const bool bAllocateTargetData,
         const bool bAllocateCachedTensors,
         void* const rng,
         const size_t cScores,
         const size_t cSubsetItemsMax,
         const ObjectiveWrapper* const pObjectiveCpu,
         const ObjectiveWrapper* const pObjectiveSIMD,
         const unsigned char* const pDataSetShared,
         const double* const aIntercept,
         const BagEbm direction,
         const size_t cSharedSamples,
         const BagEbm* const aBag,
         const double* const aInitScores,
         const size_t cIncludedSamples,
         const size_t cInnerBags,
         const size_t cWeights,
         const size_t cTerms,
         const Term* const* const apTerms,
         const IntEbm* const aiTermFeatures);

   void DestructDataSetBoosting(const size_t cTerms, const size_t cInnerBags);

   inline size_t GetCountSamples() const { return m_cSamples; }
   inline size_t GetCountSubsets() const { return m_cSubsets; }
   inline DataSubsetBoosting* GetSubsets() {
      EBM_ASSERT(nullptr != m_aSubsets);
      return m_aSubsets;
   }
   inline double GetBagWeightTotal(const size_t iBag) const {
      EBM_ASSERT(nullptr != m_aDataSetInnerBags);
      return static_cast<double>(*m_aDataSetInnerBags[iBag].GetTotalWeight());
   }
   inline size_t GetBagCountTotal(const size_t iBag) const {
      EBM_ASSERT(nullptr != m_aDataSetInnerBags);
      return static_cast<size_t>(*m_aDataSetInnerBags[iBag].GetTotalCount());
   }
   inline const DataSetInnerBag* GetDataSetInnerBag() {
      EBM_ASSERT(nullptr != m_aDataSetInnerBags);
      return m_aDataSetInnerBags;
   }

 private:
   ErrorEbm InitGradHess(const bool bAllocateHessians, const size_t cScores);

   ErrorEbm InitSampleScores(const size_t cScores,
         const double* const aIntercept,
         const BagEbm direction,
         const BagEbm* const aBag,
         const double* const aInitScores);

   ErrorEbm InitTargetData(const unsigned char* const pDataSetShared, const BagEbm direction, const BagEbm* const aBag);

   ErrorEbm InitTermData(const unsigned char* const pDataSetShared,
         const BagEbm direction,
         const size_t cSharedSamples,
         const BagEbm* const aBag,
         const size_t cTerms,
         const Term* const* const apTerms,
         const IntEbm* const aiTermFeatures);

   ErrorEbm CopyWeights(const unsigned char* const pDataSetShared, const BagEbm direction, const BagEbm* const aBag);

   ErrorEbm InitBags(const bool bAllocateCachedTensors,
         void* const rng,
         const size_t cInnerBags,
         const size_t cTerms,
         const Term* const* const apTerms);

   size_t m_cSamples;
   size_t m_cSubsets;
   DataSubsetBoosting* m_aSubsets;
   DataSetInnerBag* m_aDataSetInnerBags;
   FloatShared* m_aOriginalWeights;
};
static_assert(std::is_standard_layout<DataSetBoosting>::value,
      "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<DataSetBoosting>::value,
      "We use memcpy in several places, so disallow non-trivial types in general");

} // namespace DEFINED_ZONE_NAME

#endif // DATA_SET_BOOSTING_HPP
