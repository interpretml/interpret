// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef CONFIG_H
#define CONFIG_H

#include "EbmInternal.h"

class Config final {

   const size_t m_cOutputs;

public:
   INLINE_ALWAYS Config(const size_t cOutputs) : m_cOutputs(cOutputs) {
   }

   INLINE_ALWAYS size_t GetCountOutputs() const noexcept {
      return m_cOutputs;
   }
};

#endif // CONFIG_H