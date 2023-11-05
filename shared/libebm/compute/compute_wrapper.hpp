// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef COMPUTE_WRAPPER_HPP
#define COMPUTE_WRAPPER_HPP

#include "libebm.h" // ErrorEbm
#include "zones.h"

#include "BinSumsBoosting.hpp"
#include "BinSumsInteraction.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

template<typename TFloat>
struct ComputeWrapper final {
   static ErrorEbm StaticBinSumsBoosting(BinSumsBoostingBridge * const pParams) {
      return BinSumsBoosting<TFloat>(pParams);
   }

   static ErrorEbm StaticBinSumsInteraction(BinSumsInteractionBridge * const pParams) {
      return BinSumsInteraction<TFloat>(pParams);
   }

   INLINE_RELEASE_TEMPLATED static ErrorEbm FillWrapper(ObjectiveWrapper * const pObjectiveWrapperOut) noexcept {
      EBM_ASSERT(nullptr != pObjectiveWrapperOut);

      FunctionPointersCpp * const pFunctionPointersCpp = 
         reinterpret_cast<FunctionPointersCpp *>(malloc(sizeof(FunctionPointersCpp)));
      if(nullptr == pFunctionPointersCpp) {
         return Error_OutOfMemory;
      }
      pObjectiveWrapperOut->m_pFunctionPointersCpp = pFunctionPointersCpp;

      pFunctionPointersCpp->m_pBinSumsBoostingCpp = StaticBinSumsBoosting;
      pFunctionPointersCpp->m_pBinSumsInteractionCpp = StaticBinSumsInteraction;

      pObjectiveWrapperOut->m_cSIMDPack = static_cast<size_t>(TFloat::k_cSIMDPack);

      static_assert(std::is_unsigned<typename TFloat::TInt::T>::value,
         "TFloat::TInt::T must be an unsigned integer type");
      static_assert(std::is_same<UIntBig, typename TFloat::TInt::T>::value || std::is_same<UIntSmall, typename TFloat::TInt::T>::value,
         "TFloat::TInt::T must be either UIntBig or UIntSmall");
      static_assert(std::is_same<FloatBig, typename TFloat::T>::value || std::is_same<FloatSmall, typename TFloat::T>::value,
         "TFloat::T must be either FloatBig or FloatSmall");

      pObjectiveWrapperOut->m_cFloatBytes = sizeof(typename TFloat::T);
      pObjectiveWrapperOut->m_cUIntBytes = sizeof(typename TFloat::TInt::T);

      return Error_None;
   }
};

} // DEFINED_ZONE_NAME

#endif // COMPUTE_WRAPPER_HPP
