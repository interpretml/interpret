// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "EbmInternal.h"

#include "Registration.h"

#include "Registrable.h"
#include "Loss.h"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

struct Simd64None final {
   typedef double Unpacked;
   typedef double Packed;

private:

   Packed m_data;

public:

   INLINE_ALWAYS Simd64None() noexcept {
   }

   INLINE_ALWAYS Simd64None(const Simd64None & data) noexcept : m_data(data.m_data) {
   }

   INLINE_ALWAYS Simd64None(const float data) noexcept : m_data(static_cast<Unpacked>(data)) {
   }

   INLINE_ALWAYS Simd64None(const double data) noexcept : m_data(static_cast<Unpacked>(data)) {
   }

   INLINE_ALWAYS Simd64None(const int data) noexcept : m_data(static_cast<Unpacked>(data)) {
   }

   INLINE_ALWAYS Simd64None operator+ (const Simd64None & other) const noexcept {
      return Simd64None(m_data + other.m_data);
   }

   INLINE_ALWAYS Simd64None operator- (const Simd64None & other) const noexcept {
      return Simd64None(m_data - other.m_data);
   }

   INLINE_ALWAYS Simd64None operator* (const Simd64None & other) const noexcept {
      return Simd64None(m_data * other.m_data);
   }

   INLINE_ALWAYS Simd64None operator/ (const Simd64None & other) const noexcept {
      return Simd64None(m_data / other.m_data);
   }

   INLINE_ALWAYS bool IsAnyEqual(const Simd64None & other) const noexcept {
      return m_data == other.m_data;
   }

   INLINE_ALWAYS bool IsAnyInf() const noexcept {
      return std::isinf(m_data);
   }

   INLINE_ALWAYS bool IsAnyNaN() const noexcept {
      return std::isnan(m_data);
   }

   INLINE_ALWAYS Simd64None Sqrt() const noexcept {
      return Simd64None(std::sqrt(m_data));
   }
};

// !!! IMPORTANT: This file is compiled with the default instruction set enabled !!!

// FIRST, define the RegisterLoss function that we'll be calling from our registrations.  This is a static 
// function, so we can have duplicate named functions in other files and they'll refer to different functions
template<template <typename> class TRegistrable, typename... Args>
static INLINE_ALWAYS std::shared_ptr<const Registration> RegisterLoss(const char * const sRegistrationName, const Args...args) {
   return Register<TRegistrable, Simd64None>(sRegistrationName, args...);
}

// now include all our special loss registrations which will use the RegisterLoss function we defined above!
#include "LossRegistrations.h"

extern const std::vector<std::shared_ptr<const Registration>> RegisterLosses64None() {
   return RegisterLosses();
}

} // DEFINED_ZONE_NAME
