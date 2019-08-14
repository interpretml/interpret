// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef ATTRIBUTE_INTERNAL_H
#define ATTRIBUTE_INTERNAL_H

#include <stddef.h> // size_t, ptrdiff_t

#include "EbmInternal.h" // TML_INLINE

enum class FeatureTypeCore;

// FeatureInternal is a class internal to our library.  Our public interface will not have a "Feature" POD that we can use for C interop since everything will be a specific type of feature like OrdinalFeature (POD)
class FeatureInternalCore final {
public:
   const size_t m_cStates;
   const size_t m_iFeatureData;
   // TODO : implement feature to handle m_featureType
   const FeatureTypeCore m_featureType;
   // TODO : implement feature to handle m_bMissing
   const bool m_bMissing;

   TML_INLINE FeatureInternalCore(const size_t cStates, const size_t iFeatureData, const FeatureTypeCore featureType, const bool bMissing)
      : m_cStates(cStates)
      , m_iFeatureData(iFeatureData)
      , m_featureType(featureType)
      , m_bMissing(bMissing) {
   }
};

#endif // ATTRIBUTE_INTERNAL_H
