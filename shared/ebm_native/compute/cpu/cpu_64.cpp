// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <cmath>

#include "ebm_native.h"
#include "logging.h"
#include "common_c.h"
#include "bridge_c.h"
#include "zones.h"

#include "common_cpp.hpp"
#include "bridge_cpp.hpp"

#include "Registrable.hpp"
#include "Registration.hpp"
#include "Loss.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

struct Cpu_64_Operators final {
   typedef double Unpacked;
   typedef double Packed;

private:

   Packed m_data;

public:

   ATTRIBUTE_WARNING_DISABLE_UNINITIALIZED_MEMBER
   INLINE_ALWAYS Cpu_64_Operators() noexcept {
   }

   INLINE_ALWAYS Cpu_64_Operators(const Cpu_64_Operators & data) noexcept : m_data(data.m_data) {
   }

   INLINE_ALWAYS Cpu_64_Operators(const float data) noexcept : m_data(static_cast<Unpacked>(data)) {
   }

   INLINE_ALWAYS Cpu_64_Operators(const double data) noexcept : m_data(static_cast<Unpacked>(data)) {
   }

   INLINE_ALWAYS Cpu_64_Operators(const int data) noexcept : m_data(static_cast<Unpacked>(data)) {
   }

   INLINE_ALWAYS Cpu_64_Operators operator+ (const Cpu_64_Operators & other) const noexcept {
      return Cpu_64_Operators(m_data + other.m_data);
   }

   INLINE_ALWAYS Cpu_64_Operators operator- (const Cpu_64_Operators & other) const noexcept {
      return Cpu_64_Operators(m_data - other.m_data);
   }

   INLINE_ALWAYS Cpu_64_Operators operator* (const Cpu_64_Operators & other) const noexcept {
      return Cpu_64_Operators(m_data * other.m_data);
   }

   INLINE_ALWAYS Cpu_64_Operators operator/ (const Cpu_64_Operators & other) const noexcept {
      return Cpu_64_Operators(m_data / other.m_data);
   }

   INLINE_ALWAYS bool IsAnyEqual(const Cpu_64_Operators & other) const noexcept {
      return m_data == other.m_data;
   }

   INLINE_ALWAYS bool IsAnyInf() const noexcept {
      return std::isinf(m_data);
   }

   INLINE_ALWAYS bool IsAnyNaN() const noexcept {
      return std::isnan(m_data);
   }

   INLINE_ALWAYS Cpu_64_Operators Sqrt() const noexcept {
      return Cpu_64_Operators(std::sqrt(m_data));
   }
};

// FIRST, define the RegisterLoss function that we'll be calling from our registrations.  This is a static 
// function, so we can have duplicate named functions in other files and they'll refer to different functions
template<template <typename> class TRegistrable, typename... Args>
static INLINE_ALWAYS std::shared_ptr<const Registration> RegisterLoss(const char * const sRegistrationName, const Args...args) {
   return Register<TRegistrable, Cpu_64_Operators>(sRegistrationName, args...);
}

// now include all our special loss registrations which will use the RegisterLoss function we defined above!
#include "loss_registrations.hpp"

INTERNAL_IMPORT_EXPORT_BODY ErrorEbmType CreateLoss_Cpu_64(
   const size_t cOutputs,
   const char * const sLoss,
   const void ** const ppLossOut
) {
   return Loss::CreateLoss(&RegisterLosses, cOutputs, sLoss, ppLossOut);
}

INTERNAL_IMPORT_EXPORT_BODY ErrorEbmType CreateMetric_Cpu_64(
   const size_t cOutputs,
   const char * const sLoss,
   const char * const sLossEnd,
   const void ** const ppLossOut
) {
   UNUSED(cOutputs);
   UNUSED(sLoss);
   UNUSED(sLossEnd);
   UNUSED(ppLossOut);

   return Error_UnknownInternalError;
}

} // DEFINED_ZONE_NAME
