// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef EBM_EXCEPTION_H
#define EBM_EXCEPTION_H

#include <exception>

#include "ebm_native.h"

class EbmException : public std::exception {
   const ErrorEbmType m_error;

public:
   EbmException(const ErrorEbmType error) : m_error(error) {
   }

   ErrorEbmType GetError() const noexcept {
      return m_error;
   }
};

#endif // EBM_EXCEPTION_H
