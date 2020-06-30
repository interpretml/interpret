// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <ebm@koch.ninja>

#ifndef EBM_INTERACTION_STATE_H
#define EBM_INTERACTION_STATE_H

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t
#include <limits> // numeric_limits

#include "ebm_native.h"
#include "EbmInternal.h"
#include "Logging.h" // EBM_ASSERT & LOG
// feature includes
#include "FeatureAtomic.h"
// dataset depends on features
#include "DataSetInteraction.h"

class EbmInteractionState final {
   ptrdiff_t m_runtimeLearningTypeOrCountTargetClasses;

   size_t m_cFeatures;
   Feature * m_aFeatures;

   DataSetByFeature m_dataSet;

   unsigned int m_cLogEnterMessages;
   unsigned int m_cLogExitMessages;

public:

   EbmInteractionState() = default; // preserve our POD status
   ~EbmInteractionState() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   EBM_INLINE void InitializeZero() {
      m_runtimeLearningTypeOrCountTargetClasses = 0;

      m_cFeatures = 0;
      m_aFeatures = nullptr;

      m_dataSet.InitializeZero();

      m_cLogEnterMessages = 0;
      m_cLogExitMessages = 0;
   }

   EBM_INLINE ptrdiff_t GetRuntimeLearningTypeOrCountTargetClasses() {
      return m_runtimeLearningTypeOrCountTargetClasses;
   }

   EBM_INLINE unsigned int * GetPointerCountLogEnterMessages() {
      return &m_cLogEnterMessages;
   }

   EBM_INLINE unsigned int * GetPointerCountLogExitMessages() {
      return &m_cLogExitMessages;
   }

   EBM_INLINE const DataSetByFeature * GetDataSetByFeature() const {
      return &m_dataSet;
   }

   EBM_INLINE const Feature * GetFeatures() const {
      return m_aFeatures;
   }

   EBM_INLINE size_t GetCountFeatures() const {
      return m_cFeatures;
   }

   static void Free(EbmInteractionState * const pInteractionDetection);
   static EbmInteractionState * Allocate(
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
      const size_t cFeatures,
      const FloatEbmType * const optionalTempParams,
      const EbmNativeFeature * const aNativeFeatures,
      const size_t cInstances,
      const void * const aTargets,
      const IntEbmType * const aBinnedData,
      const FloatEbmType * const aPredictorScores
   );
};
static_assert(std::is_standard_layout<EbmInteractionState>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<EbmInteractionState>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<EbmInteractionState>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

#endif // EBM_INTERACTION_STATE_H
