// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef ATTRIBUTE_SET_INTERNAL_H
#define ATTRIBUTE_SET_INTERNAL_H

#include <vector>
#include <stddef.h> // size_t, ptrdiff_t

#include "EbmInternal.h" // TML_INLINE

class AttributeInternalCore;

class AttributeSetInternalCore final {
public:
   // TODO: turn this data protected

   std::vector<AttributeInternalCore *> m_inputAttributes;

   TML_INLINE AttributeSetInternalCore() {
   }

   ~AttributeSetInternalCore();

   TML_INLINE size_t GetCountAttributes() const {
      return m_inputAttributes.size();
   }

   AttributeInternalCore * AddAttribute(const size_t cStates, const size_t iAttributeData, const AttributeTypeCore attributeType, const bool bMissing);
};

#endif // ATTRIBUTE_SET_INTERNAL_H
