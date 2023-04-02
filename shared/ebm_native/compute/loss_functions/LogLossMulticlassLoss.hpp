// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// !! To add a new loss/objective function in C++ follow the steps at the top of the "loss_registrations.hpp" file !!
// Do not use this file as a reference for other loss functions. LogLoss is special.

#include "Loss.hpp"

#include "approximate_math.hpp"
#include "ebm_stats.hpp"

template<typename TFloat>
struct LogLossMulticlassLoss final : public MulticlassLoss {
   static constexpr bool k_bMse = false;
   LOSS_CLASS_CONSTANTS_BOILERPLATE(true)
   LOSS_CLASS_VIRTUAL_BOILERPLATE(LogLossMulticlassLoss)

   inline LogLossMulticlassLoss(const Config & config) {
      if(1 == config.cOutputs) {
         // we share the tag "log_loss" with binary classification
         throw SkipRegistrationException();
      }

      if(config.cOutputs <= 0) {
         throw ParamMismatchWithConfigException();
      }
   }

   inline double GetFinalMultiplier() const noexcept {
      return 1.0;
   }

   template<ptrdiff_t cCompilerScores, ptrdiff_t cCompilerPack, bool bHessian, bool bKeepGradHess, bool bCalcMetric, bool bWeight>
   GPU_DEVICE void InteriorApplyUpdateTemplated(ApplyUpdateBridge * const pData) const {
      Loss::InteriorApplyUpdate<typename std::remove_pointer<decltype(this)>::type, TFloat,
         cCompilerScores, cCompilerPack, bHessian, bKeepGradHess, bCalcMetric, bWeight>(pData);
   }
};
