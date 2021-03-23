// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef EBM_EXCEPTION_H
#define EBM_EXCEPTION_H

#include <exception>

#include "ebm_native.h"
#include "zones.h"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

class EbmException : public std::exception {
   const ErrorEbmType m_error;

public:
   EbmException(const ErrorEbmType error) : m_error(error) {
   }

   ErrorEbmType GetError() const noexcept {
      return m_error;
   }
};

} // DEFINED_ZONE_NAME

#endif // EBM_EXCEPTION_H
