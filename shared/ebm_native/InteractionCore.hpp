// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <ebm@koch.ninja>

#ifndef INTERACTION_CORE_HPP
#define INTERACTION_CORE_HPP

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t
#include <limits> // numeric_limits

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "ebm_internal.hpp"

// feature includes
#include "Feature.hpp"
// dataset depends on features
#include "DataSetInteraction.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

class InteractionCore final {
   ptrdiff_t m_runtimeLearningTypeOrCountTargetClasses;

   size_t m_cFeatures;
   Feature * m_aFeatures;

   DataSetInteraction m_dataFrame;

   int m_cLogEnterMessages;
   int m_cLogExitMessages;

public:

   InteractionCore() = default; // preserve our POD status
   ~InteractionCore() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   INLINE_ALWAYS void InitializeZero() {
      m_runtimeLearningTypeOrCountTargetClasses = 0;

      m_cFeatures = 0;
      m_aFeatures = nullptr;

      m_dataFrame.InitializeZero();

      m_cLogEnterMessages = 0;
      m_cLogExitMessages = 0;
   }

   INLINE_ALWAYS ptrdiff_t GetRuntimeLearningTypeOrCountTargetClasses() {
      return m_runtimeLearningTypeOrCountTargetClasses;
   }

   INLINE_ALWAYS int * GetPointerCountLogEnterMessages() {
      return &m_cLogEnterMessages;
   }

   INLINE_ALWAYS int * GetPointerCountLogExitMessages() {
      return &m_cLogExitMessages;
   }

   INLINE_ALWAYS const DataSetInteraction * GetDataSetInteraction() const {
      return &m_dataFrame;
   }

   INLINE_ALWAYS const Feature * GetFeatures() const {
      return m_aFeatures;
   }

   INLINE_ALWAYS size_t GetCountFeatures() const {
      return m_cFeatures;
   }

   static void Free(InteractionCore * const pInteractionCore);
   static InteractionCore * Allocate(
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
      const size_t cFeatures,
      const FloatEbmType * const optionalTempParams,
      const BoolEbmType * const aFeaturesCategorical,
      const IntEbmType * const aFeaturesBinCount,
      const size_t cSamples,
      const void * const aTargets,
      const IntEbmType * const aBinnedData,
      const FloatEbmType * const aWeights,
      const FloatEbmType * const aPredictorScores
   );
};
static_assert(std::is_standard_layout<InteractionCore>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<InteractionCore>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<InteractionCore>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

} // DEFINED_ZONE_NAME

#endif // INTERACTION_CORE_HPP
