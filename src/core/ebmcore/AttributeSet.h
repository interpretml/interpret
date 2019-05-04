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

   // TODO: can m_cTargetStates be moved into TmlState?
   const size_t m_cTargetStates;
   std::vector<AttributeInternalCore *> m_inputAttributes;

   TML_INLINE AttributeSetInternalCore(size_t cTargetStates)
      : m_cTargetStates(cTargetStates) {
   }

   ~AttributeSetInternalCore();

   TML_INLINE size_t GetCountAttributes() const {
      return m_inputAttributes.size();
   }

   AttributeInternalCore * AddAttribute(const size_t cStates, const size_t iAttributeData, const AttributeTypeCore attributeType, const bool bMissing);
};

#endif // ATTRIBUTE_SET_INTERNAL_H
