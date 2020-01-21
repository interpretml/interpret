// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef FEATURE_H
#define FEATURE_H

#include <stddef.h> // size_t, ptrdiff_t

#include "EbmInternal.h" // EBM_INLINE

enum class FeatureType;

class Feature final {
public:
   const size_t m_cBins;
   const size_t m_iFeatureData;
   const FeatureType m_featureType;
   const bool m_bMissing;

   EBM_INLINE Feature(const size_t cBins, const size_t iFeatureData, const FeatureType featureType, const bool bMissing)
      : m_cBins(cBins)
      , m_iFeatureData(iFeatureData)
      , m_featureType(featureType)
      , m_bMissing(bMissing) {
   }
};

#endif // FEATURE_H
