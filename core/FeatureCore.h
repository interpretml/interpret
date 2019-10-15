// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef FEATURE_CORE_H
#define FEATURE_CORE_H

#include <stddef.h> // size_t, ptrdiff_t

#include "EbmInternal.h" // EBM_INLINE

enum class FeatureTypeCore;

class FeatureCore final {
public:
   const size_t m_cBins;
   const size_t m_iFeatureData;
   // TODO : implement feature to handle m_featureType
   const FeatureTypeCore m_featureType;
   // TODO : implement feature to handle m_bMissing
   const bool m_bMissing;

   EBM_INLINE FeatureCore(const size_t cBins, const size_t iFeatureData, const FeatureTypeCore featureType, const bool bMissing)
      : m_cBins(cBins)
      , m_iFeatureData(iFeatureData)
      , m_featureType(featureType)
      , m_bMissing(bMissing) {
   }
};

#endif // FEATURE_CORE_H
