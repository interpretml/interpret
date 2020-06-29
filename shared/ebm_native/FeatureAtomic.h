// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef FEATURE_H
#define FEATURE_H

#include <stddef.h> // size_t, ptrdiff_t

#include "EbmInternal.h" // EBM_INLINE

enum class FeatureType;

class Feature final {
   size_t m_cBins;
   size_t m_iFeatureData;
   FeatureType m_featureType;
   bool m_bMissing;

public:

   EBM_INLINE void Initialize(const size_t cBins, const size_t iFeatureData, const FeatureType featureType, const bool bMissing) {
      m_cBins = cBins;
      m_iFeatureData = iFeatureData;
      m_featureType = featureType;
      m_bMissing = bMissing;
   }

   EBM_INLINE size_t GetCountBins() const {
      StopClangAnalysis();
      return m_cBins;
   }

   EBM_INLINE size_t GetIndexFeatureData() const {
      return m_iFeatureData;
   }

   EBM_INLINE FeatureType GetFeatureType() const {
      return m_featureType;
   }

   EBM_INLINE bool GetIsMissing() const {
      return m_bMissing;
   }
};
static_assert(std::is_standard_layout<Feature>::value,
   "we use malloc to allocate this, so it needs to be standard layout");

#endif // FEATURE_H
