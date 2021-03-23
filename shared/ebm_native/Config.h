// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef CONFIG_H
#define CONFIG_H

#include "ebm_native.h"
#include "logging.h"
#include "bridge_c.h"
#include "common_c.h"
#include "zones.h"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

class Config final {

   const size_t m_cOutputs;

public:
   INLINE_ALWAYS Config(const size_t cOutputs) : m_cOutputs(cOutputs) {
   }

   INLINE_ALWAYS size_t GetCountOutputs() const noexcept {
      return m_cOutputs;
   }
};

} // DEFINED_ZONE_NAME

#endif // CONFIG_H