// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <new> // std::nothrow
#include <stddef.h> // size_t, ptrdiff_t

#include "AttributeInternal.h"
#include "AttributeSet.h"

AttributeSetInternalCore::~AttributeSetInternalCore() {
   for(AttributeInternalCore * pInputAttribute : m_inputAttributes) {
      delete pInputAttribute;
   }
}

AttributeInternalCore * AttributeSetInternalCore::AddAttribute(const size_t cStates, const size_t iAttributeData, const AttributeTypeCore attributeType, const bool bMissing) {
   AttributeInternalCore * const pAttribute = new (std::nothrow) AttributeInternalCore(cStates, iAttributeData, attributeType, bMissing);
   if(nullptr == pAttribute) {
      return nullptr;
   }
   try {
      m_inputAttributes.push_back(pAttribute);
   } catch(...) {
      delete pAttribute;
      return nullptr;
   }
   return pAttribute;
}

