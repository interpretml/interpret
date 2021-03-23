// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <stddef.h> // size_t, ptrdiff_t

// TODO: put EVERYTHING in this zone into a single anonymous namespace (including *.cpp files which we'll include!)
//namespace {

// !!!! VERY VERY IMPORTANT !!!!
// Having translation units (*.cpp files) with different compiler settings is VERY VERY dangerous due to the
// possibility of an ODR violation.  Unfortunately, we really want different SIMD options that have different
// compiler settings, so we take the care here to separate our code into completely different sections.  This
// anonymous namespace helps us separate classes/template/functions/etc that could share definitions.  Each
// SIMD type is under it's own anonymous namespace to avoid cross polution.  We use exclusively POD data types
// to communicate from this translation unit to annother
// Unfortunately, having non POD types causes that are used between translation units not compiled with the same
// settings is an ODR violation since they 
// Further Info:
// https://stackoverflow.com/questions/20470369/c-automatically-implemented-functions-and-the-odr
// https://en.wikipedia.org/wiki/One_Definition_Rule
//
// C++ classes fail the rule of "For a given entity, each definition must have the same sequence of tokens." since
// different compiler flags are not guaranteed to generate the same class definition alignment unless the types
// are POD structures.

#include "ebm_native.h"
#include "EbmInternal.h"
   // very independent includes
#include "logging.h" // EBM_ASSERT & LOG
#include "Registration.h"

#include "Registrable.h"
#include "Loss.h"

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

//} // namespace