// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef DATA_FRAME_INTERACTION_H
#define DATA_FRAME_INTERACTION_H

#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h" // FloatEbmType
#include "EbmInternal.h" // INLINE_ALWAYS
#include "Logging.h" // EBM_ASSERT & LOG
#include "FeatureAtomic.h"

class DataFrameInteraction final {
   FloatEbmType * m_aResidualErrors;
   StorageDataType * * m_aaInputData;
   size_t m_cSamples;
   size_t m_cFeatureAtomics;

public:

   DataFrameInteraction() = default; // preserve our POD status
   ~DataFrameInteraction() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   void Destruct();

   INLINE_ALWAYS void InitializeZero() {
      m_aResidualErrors = nullptr;
      m_aaInputData = nullptr;
      m_cSamples = 0;
      m_cFeatureAtomics = 0;
   }

   bool Initialize(
      const size_t cFeatureAtomics, 
      const FeatureAtomic * const aFeatureAtomics, 
      const size_t cSamples, 
      const IntEbmType * const aInputDataFrom, 
      const void * const aTargetData, 
      const FloatEbmType * const aPredictorScores, 
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses
   );

   INLINE_ALWAYS const FloatEbmType * GetResidualPointer() const {
      EBM_ASSERT(nullptr != m_aResidualErrors);
      return m_aResidualErrors;
   }
   // TODO: we can change this to take the m_iFeatureData value directly, which we get from a loop index
   INLINE_ALWAYS const StorageDataType * GetInputDataPointer(const FeatureAtomic * const pFeatureAtomic) const {
      EBM_ASSERT(nullptr != pFeatureAtomic);
      EBM_ASSERT(pFeatureAtomic->GetIndexFeatureAtomicData() < m_cFeatureAtomics);
      EBM_ASSERT(nullptr != m_aaInputData);
      return m_aaInputData[pFeatureAtomic->GetIndexFeatureAtomicData()];
   }
   INLINE_ALWAYS size_t GetCountSamples() const {
      return m_cSamples;
   }
   INLINE_ALWAYS size_t GetCountFeatureAtomics() const {
      return m_cFeatureAtomics;
   }
};
static_assert(std::is_standard_layout<DataFrameInteraction>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<DataFrameInteraction>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<DataFrameInteraction>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

#endif // DATA_FRAME_INTERACTION_H
