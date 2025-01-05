// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// !!! NOTE: To add a new objective in C++, follow the steps listed at the top of the "objective_registrations.hpp" file
// !!!

#ifndef OBJECTIVE_HPP
#define OBJECTIVE_HPP

#include <stddef.h> // size_t, ptrdiff_t
#include <memory> // shared_ptr, unique_ptr
#include <type_traits> // is_same
#include <vector>

#include "libebm.h" // ErrorEbm
#include "logging.h" // EBM_ASSERT
#include "unzoned.h" // INLINE_ALWAYS

#include "bridge.hpp" // IdentifyTask

#include "zoned_bridge_cpp_functions.hpp" // FunctionPointersCpp
#include "compute.hpp" // GPU_GLOBAL
#include "registration_exceptions.hpp"
#include "Registration.hpp"

struct ApplyUpdateBridge;

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

template<typename TFloat> static const std::vector<std::shared_ptr<const Registration>> RegisterObjectives();

struct SingletaskObjective;
struct BinaryObjective;
struct MulticlassObjective;
struct RegressionObjective;

struct MultitaskObjective;
struct BinaryMultitaskObjective;
struct MulticlassMultitaskObjective;
struct RegressionMultitaskObjective;

template<typename TFloat> class alignas(alignof(TFloat)) GradientHessian {
   TFloat m_gradient;
   TFloat m_hessian;

 public:
   template<typename T1, typename T2>
   GPU_DEVICE inline GradientHessian(const T1& gradient, const T2& hessian) :
         m_gradient(TFloat(gradient)), m_hessian(TFloat(hessian)) {}

   GPU_DEVICE inline TFloat GetGradient() const noexcept { return m_gradient; }
   GPU_DEVICE inline TFloat GetHessian() const noexcept { return m_hessian; }
};
template<typename TFloat>
GPU_DEVICE inline GradientHessian<TFloat> MakeGradientHessian(const TFloat& gradient, const TFloat& hessian) {
   return GradientHessian<TFloat>(gradient, hessian);
}
template<typename TFloat>
GPU_DEVICE inline GradientHessian<TFloat> MakeGradientHessian(const TFloat& gradient, const double hessian) {
   return GradientHessian<TFloat>(gradient, hessian);
}
template<typename TFloat>
GPU_DEVICE inline GradientHessian<TFloat> MakeGradientHessian(const double gradient, const TFloat& hessian) {
   return GradientHessian<TFloat>(gradient, hessian);
}
template<typename TFloat>
GPU_DEVICE inline GradientHessian<TFloat> MakeGradientHessian(const double gradient, const double hessian) {
   return GradientHessian<TFloat>(gradient, hessian);
}

template<typename TObjective,
      bool bCollapsed,
      bool bValidation,
      bool bWeight,
      bool bHessian,
      bool bUseApprox,
      size_t cCompilerScores,
      int cCompilerPack>
GPU_DEVICE INLINE_RELEASE_TEMPLATED static void DoneBitpacking(
      const Objective* const pObjective, ApplyUpdateBridge* const pData) {
   const TObjective* const pObjectiveSpecific = static_cast<const TObjective*>(pObjective);
   pObjectiveSpecific->template InjectedApplyUpdate<bCollapsed,
         bValidation,
         bWeight,
         bHessian,
         bUseApprox,
         cCompilerScores,
         cCompilerPack>(pData);
}

// in our current format cCompilerScores will always be 1, but just in case we change our code to allow
// for special casing multiclass with compile time unrolling of the compiler pack, leave cCompilerScores here
template<typename TObjective,
      bool bCollapsed,
      bool bValidation,
      bool bWeight,
      bool bHessian,
      bool bUseApprox,
      size_t cCompilerScores,
      int cCompilerPack>
struct BitPackObjective final {
   GPU_DEVICE INLINE_RELEASE_TEMPLATED static void Func(
         const Objective* const pObjective, ApplyUpdateBridge* const pData) {

      static_assert(!bCollapsed, "Cannot be bCollapsed since there would be no bitpacking");
      static_assert(cCompilerPack <= COUNT_BITS(typename TObjective::TFloatInternal::TInt::T),
            "cCompilerPack must fit into the bitpack");

      if(cCompilerPack == pData->m_cPack) {
         size_t cSamples = pData->m_cSamples;
         const size_t cRemnants =
               cSamples % static_cast<size_t>(cCompilerPack * TObjective::TFloatInternal::k_cSIMDPack);
         if(0 != cRemnants) {
            pData->m_cSamples = cRemnants;

            DoneBitpacking<TObjective,
                  bCollapsed,
                  bValidation,
                  bWeight,
                  bHessian,
                  bUseApprox,
                  cCompilerScores,
                  k_cItemsPerBitPackUndefined>(pObjective, pData);

            cSamples -= cRemnants;
            if(0 == cSamples) {
               return;
            }
            pData->m_cSamples = cSamples;

            if(bWeight) {
               EBM_ASSERT(nullptr != pData->m_aWeights);
               pData->m_aWeights =
                     IndexByte(pData->m_aWeights, sizeof(typename TObjective::TFloatInternal::T) * cRemnants);
            } else {
               EBM_ASSERT(nullptr == pData->m_aWeights);
            }

            const size_t cScores = GET_COUNT_SCORES(cCompilerScores, pData->m_cScores);

            constexpr bool bRmse = Objective_Rmse == TObjective::k_objective;
            constexpr bool bGradientsAndHessians = bRmse || !bValidation;
            if(bGradientsAndHessians) {
               EBM_ASSERT(nullptr != pData->m_aGradientsAndHessians);
               pData->m_aGradientsAndHessians = IndexByte(pData->m_aGradientsAndHessians,
                     sizeof(typename TObjective::TFloatInternal::T) * (bHessian ? size_t{2} : size_t{1}) * cScores *
                           cRemnants);
            } else {
               EBM_ASSERT(nullptr == pData->m_aGradientsAndHessians);
            }

            if(!bRmse) {
               EBM_ASSERT(nullptr != pData->m_aTargets);
               EBM_ASSERT(nullptr != pData->m_aSampleScores);

               constexpr bool bClassification = Task_GeneralClassification == IdentifyTask(TObjective::k_linkFunction);
               if(bClassification) {
                  // targets are integers
                  pData->m_aTargets =
                        IndexByte(pData->m_aTargets, sizeof(typename TObjective::TFloatInternal::TInt::T) * cRemnants);
               } else {
                  // targets are floats
                  pData->m_aTargets =
                        IndexByte(pData->m_aTargets, sizeof(typename TObjective::TFloatInternal::T) * cRemnants);
               }
               pData->m_aSampleScores = IndexByte(
                     pData->m_aSampleScores, sizeof(typename TObjective::TFloatInternal::T) * cScores * cRemnants);
            } else {
               EBM_ASSERT(nullptr == pData->m_aTargets);
               EBM_ASSERT(nullptr == pData->m_aSampleScores);
            }
         }
         DoneBitpacking<TObjective,
               bCollapsed,
               bValidation,
               bWeight,
               bHessian,
               bUseApprox,
               cCompilerScores,
               cCompilerPack>(pObjective, pData);
      } else {
         BitPackObjective<TObjective,
               bCollapsed,
               bValidation,
               bWeight,
               bHessian,
               bUseApprox,
               cCompilerScores,
               GetNextBitPack<typename TObjective::TFloatInternal::TInt::T>(
                     cCompilerPack, TObjective::k_cItemsPerBitPackMin)>::Func(pObjective, pData);
      }
   }
};
template<typename TObjective,
      bool bCollapsed,
      bool bValidation,
      bool bWeight,
      bool bHessian,
      bool bUseApprox,
      size_t cCompilerScores>
struct BitPackObjective<TObjective,
      bCollapsed,
      bValidation,
      bWeight,
      bHessian,
      bUseApprox,
      cCompilerScores,
      k_cItemsPerBitPackUndefined>
      final {
   GPU_DEVICE INLINE_RELEASE_TEMPLATED static void Func(
         const Objective* const pObjective, ApplyUpdateBridge* const pData) {

      static_assert(!bCollapsed, "Cannot be bCollapsed since there would be no bitpacking");

      DoneBitpacking<TObjective,
            bCollapsed,
            bValidation,
            bWeight,
            bHessian,
            bUseApprox,
            cCompilerScores,
            k_cItemsPerBitPackUndefined>(pObjective, pData);
   }
};

template<typename TObjective,
      bool bCollapsed,
      bool bValidation,
      bool bWeight,
      bool bHessian,
      bool bUseApprox,
      size_t cCompilerScores,
      typename std::enable_if<!(bCollapsed || 1 != cCompilerScores || bUseApprox ||
                                    AccelerationFlags_NONE == TObjective::TFloatInternal::k_zone),
            int>::type = 0>
GPU_DEVICE INLINE_RELEASE_TEMPLATED static void ApplyBitpacking(
      const Objective* const pObjective, ApplyUpdateBridge* const pData) {
   BitPackObjective<TObjective,
         bCollapsed,
         bValidation,
         bWeight,
         bHessian,
         bUseApprox,
         cCompilerScores,
         GetFirstBitPack<typename TObjective::TFloatInternal::TInt::T>(
               TObjective::k_cItemsPerBitPackMax, TObjective::k_cItemsPerBitPackMin)>::Func(pObjective, pData);
}
template<typename TObjective,
      bool bCollapsed,
      bool bValidation,
      bool bWeight,
      bool bHessian,
      bool bUseApprox,
      size_t cCompilerScores,
      typename std::enable_if<bCollapsed || 1 != cCompilerScores || bUseApprox ||
                  AccelerationFlags_NONE == TObjective::TFloatInternal::k_zone,
            int>::type = 0>
GPU_DEVICE INLINE_RELEASE_TEMPLATED static void ApplyBitpacking(
      const Objective* const pObjective, ApplyUpdateBridge* const pData) {
   DoneBitpacking<TObjective,
         bCollapsed,
         bValidation,
         bWeight,
         bHessian,
         bUseApprox,
         cCompilerScores,
         k_cItemsPerBitPackUndefined>(pObjective, pData);
}

template<typename TObjective,
      bool bCollapsed,
      bool bValidation,
      bool bWeight,
      bool bHessian,
      bool bUseApprox,
      size_t cCompilerScores>
GPU_GLOBAL static void RemoteApplyUpdate(const Objective* const pObjective, ApplyUpdateBridge* const pData) {
   ApplyBitpacking<TObjective, bCollapsed, bValidation, bWeight, bHessian, bUseApprox, cCompilerScores>(
         pObjective, pData);
}

struct Registrable {
   // TODO: move this into its own file once we create Metric classes that are also Registrable
 protected:
   Registrable() = default;
   ~Registrable() = default;
};

struct Objective : public Registrable {
 private:
   // Welcome to the demented hall of mirrors.. a prison for your mind
   // And no, I did not make this to purposely torment you

   template<typename TObjective> constexpr static bool IsEdgeObjective() {
      return std::is_base_of<BinaryObjective, TObjective>::value ||
            std::is_base_of<MulticlassObjective, TObjective>::value ||
            std::is_base_of<RegressionObjective, TObjective>::value ||
            std::is_base_of<BinaryMultitaskObjective, TObjective>::value ||
            std::is_base_of<MulticlassMultitaskObjective, TObjective>::value ||
            std::is_base_of<RegressionMultitaskObjective, TObjective>::value;
   }

   template<typename TObjective,
         bool bCollapsed,
         typename std::enable_if<Objective_Rmse != TObjective::k_objective, int>::type = 0>
   INLINE_RELEASE_TEMPLATED ErrorEbm OptionsApplyUpdate(ApplyUpdateBridge* const pData) const {
      if(EBM_FALSE != pData->m_bValidation) {
         static constexpr bool bValidation = true;

         // the validation set will have no gradients or hessians
         EBM_ASSERT(nullptr == pData->m_aGradientsAndHessians);

         // the hessian is only used for gradient/hessian calculation, not for metric calculations
         EBM_ASSERT(EBM_FALSE == pData->m_bHessianNeeded);
         static constexpr bool bHessian = false;

         if(nullptr != pData->m_aWeights) {
            static constexpr bool bWeight = true;
            return ApproxApplyUpdate<TObjective, bCollapsed, bValidation, bWeight, bHessian>(pData);
         } else {
            static constexpr bool bWeight = false;
            return ApproxApplyUpdate<TObjective, bCollapsed, bValidation, bWeight, bHessian>(pData);
         }
      } else {
         static constexpr bool bValidation = false;

         EBM_ASSERT(nullptr != pData->m_aGradientsAndHessians);

         // we only use weights for calculating the metric. Weights get applied
         // in BinSumsBoosting or during initialization for interactions
         EBM_ASSERT(nullptr == pData->m_aWeights);
         static constexpr bool bWeight = false;

         return HessianApplyUpdate<TObjective, bCollapsed, bValidation, bWeight>(pData);
      }
   }
   template<typename TObjective,
         bool bCollapsed,
         typename std::enable_if<Objective_Rmse == TObjective::k_objective, int>::type = 0>
   INLINE_RELEASE_TEMPLATED ErrorEbm OptionsApplyUpdate(ApplyUpdateBridge* const pData) const {
      EBM_ASSERT(nullptr != pData->m_aGradientsAndHessians); // we always keep gradients for regression

      EBM_ASSERT(EBM_FALSE == pData->m_bHessianNeeded);
      static constexpr bool bHessian = false;

      if(EBM_FALSE != pData->m_bValidation) {
         static constexpr bool bValidation = true;

         if(nullptr != pData->m_aWeights) {
            static constexpr bool bWeight = true;
            return ApproxApplyUpdate<TObjective, bCollapsed, bValidation, bWeight, bHessian>(pData);
         } else {
            static constexpr bool bWeight = false;
            return ApproxApplyUpdate<TObjective, bCollapsed, bValidation, bWeight, bHessian>(pData);
         }
      } else {
         static constexpr bool bValidation = false;

         // we only use weights for calculating the metric. Weights get applied
         // in BinSumsBoosting or during initialization for interactions
         EBM_ASSERT(nullptr == pData->m_aWeights);
         static constexpr bool bWeight = false; // if we are not calculating the metric then we never need the weights

         return ApproxApplyUpdate<TObjective, bCollapsed, bValidation, bWeight, bHessian>(pData);
      }
   }

   template<typename TObjective,
         bool bCollapsed,
         bool bValidation,
         bool bWeight,
         typename std::enable_if<TObjective::k_bHessian, int>::type = 0>
   INLINE_RELEASE_TEMPLATED ErrorEbm HessianApplyUpdate(ApplyUpdateBridge* const pData) const {
      if(pData->m_bHessianNeeded) {
         return ApproxApplyUpdate<TObjective, bCollapsed, bValidation, bWeight, true>(pData);
      } else {
         return ApproxApplyUpdate<TObjective, bCollapsed, bValidation, bWeight, false>(pData);
      }
   }
   template<typename TObjective,
         bool bCollapsed,
         bool bValidation,
         bool bWeight,
         typename std::enable_if<!TObjective::k_bHessian, int>::type = 0>
   INLINE_RELEASE_TEMPLATED ErrorEbm HessianApplyUpdate(ApplyUpdateBridge* const pData) const {
      EBM_ASSERT(!pData->m_bHessianNeeded);
      return ApproxApplyUpdate<TObjective, bCollapsed, bValidation, bWeight, false>(pData);
   }

   template<typename TObjective,
         bool bCollapsed,
         bool bValidation,
         bool bWeight,
         bool bHessian,
         typename std::enable_if<TObjective::k_bHasApprox, int>::type = 0>
   INLINE_RELEASE_TEMPLATED ErrorEbm ApproxApplyUpdate(ApplyUpdateBridge* const pData) const {
      if(EBM_FALSE != pData->m_bUseApprox) {
         return CountApplyUpdate<TObjective, bCollapsed, bValidation, bWeight, bHessian, true>(pData);
      } else {
         return CountApplyUpdate<TObjective, bCollapsed, bValidation, bWeight, bHessian, false>(pData);
      }
   }
   template<typename TObjective,
         bool bCollapsed,
         bool bValidation,
         bool bWeight,
         bool bHessian,
         typename std::enable_if<!TObjective::k_bHasApprox, int>::type = 0>
   INLINE_RELEASE_TEMPLATED ErrorEbm ApproxApplyUpdate(ApplyUpdateBridge* const pData) const {
      return CountApplyUpdate<TObjective, bCollapsed, bValidation, bWeight, bHessian, false>(pData);
   }

   // if we have multiple scores AND multiple bitpacks, then we have two nested loops in our final function
   // and the compiler will only unroll the inner loop.  That inner loop will be for the scores, so there
   // is not much benefit in generating hard coded loop counts for the bitpacks, so short circut the
   // bit packing to use the dynamic value if we don't have the single bin case.  This also solves
   // part of our template blowup issue of having N * M starting point templates where N is the number
   // of scores and M is the number of bit packs.  If we use 8 * 16 that's already 128 copies of the
   // templated function at this point and more later.  Reducing this to just 16 is very very helpful.
   template<typename TObjective,
         bool bCollapsed,
         bool bValidation,
         bool bWeight,
         bool bHessian,
         bool bUseApprox,
         typename std::enable_if<!TObjective::IsMultiScore, int>::type = 0>
   INLINE_RELEASE_TEMPLATED ErrorEbm CountApplyUpdate(ApplyUpdateBridge* const pData) const {
      return OperatorApplyUpdate<TObjective, bCollapsed, bValidation, bWeight, bHessian, bUseApprox, k_oneScore>(pData);
   }
   template<typename TObjective,
         bool bCollapsed,
         bool bValidation,
         bool bWeight,
         bool bHessian,
         bool bUseApprox,
         typename std::enable_if<TObjective::IsMultiScore &&
                     (std::is_base_of<MulticlassMultitaskObjective, TObjective>::value || !bHessian || bUseApprox ||
                           bCollapsed),
               int>::type = 0>
   INLINE_RELEASE_TEMPLATED ErrorEbm CountApplyUpdate(ApplyUpdateBridge* const pData) const {
      // multiclass multitask is going to need some really special handling, so use dynamic scores
      // don't blow up our complexity if we have only 1 bin or during init. Just use dynamic for the count of scores
      return OperatorApplyUpdate<TObjective, bCollapsed, bValidation, bWeight, bHessian, bUseApprox, k_dynamicScores>(
            pData);
   }
   template<typename TObjective,
         bool bCollapsed,
         bool bValidation,
         bool bWeight,
         bool bHessian,
         bool bUseApprox,
         typename std::enable_if<TObjective::IsMultiScore &&
                     !(std::is_base_of<MulticlassMultitaskObjective, TObjective>::value || !bHessian || bUseApprox ||
                           bCollapsed),
               int>::type = 0>
   INLINE_RELEASE_TEMPLATED ErrorEbm CountApplyUpdate(ApplyUpdateBridge* const pData) const {
      // don't blow up our complexity if we have only 1 bin or during init. Just use dynamic for the count of scores
      return CountScores<TObjective,
            bCollapsed,
            bValidation,
            bWeight,
            bHessian,
            bUseApprox,
            (k_cCompilerScoresMax < k_cCompilerScoresStart ? k_dynamicScores : k_cCompilerScoresStart)>::Func(this,
            pData);
   }

   template<typename TObjective,
         bool bCollapsed,
         bool bValidation,
         bool bWeight,
         bool bHessian,
         bool bUseApprox,
         size_t cCompilerScores>
   struct CountScores final {
      INLINE_ALWAYS static ErrorEbm Func(const Objective* const pObjective, ApplyUpdateBridge* const pData) {
         if(cCompilerScores == pData->m_cScores) {
            return pObjective->OperatorApplyUpdate<TObjective,
                  bCollapsed,
                  bValidation,
                  bWeight,
                  bHessian,
                  bUseApprox,
                  cCompilerScores>(pData);
         } else {
            return CountScores < TObjective, bCollapsed, bValidation, bWeight, bHessian, bUseApprox,
                   k_cCompilerScoresMax == cCompilerScores ? k_dynamicScores :
                                                             cCompilerScores + 1 > ::Func(pObjective, pData);
         }
      }
   };
   template<typename TObjective, bool bCollapsed, bool bValidation, bool bWeight, bool bHessian, bool bUseApprox>
   struct CountScores<TObjective, bCollapsed, bValidation, bWeight, bHessian, bUseApprox, k_dynamicScores> final {
      INLINE_ALWAYS static ErrorEbm Func(const Objective* const pObjective, ApplyUpdateBridge* const pData) {
         return pObjective->OperatorApplyUpdate<TObjective,
               bCollapsed,
               bValidation,
               bWeight,
               bHessian,
               bUseApprox,
               k_dynamicScores>(pData);
      }
   };

   template<typename TObjective,
         bool bCollapsed,
         bool bValidation,
         bool bWeight,
         bool bHessian,
         bool bUseApprox,
         size_t cCompilerScores>
   INLINE_RELEASE_TEMPLATED ErrorEbm OperatorApplyUpdate(ApplyUpdateBridge* const pData) const {
      return TObjective::TFloatInternal::template OperatorApplyUpdate<TObjective,
            bCollapsed,
            bValidation,
            bWeight,
            bHessian,
            bUseApprox,
            cCompilerScores>(this, pData);
   }

 protected:
   template<typename TObjective, typename TFloat, bool bHessian, typename std::enable_if<bHessian, int>::type = 0>
   GPU_DEVICE INLINE_ALWAYS typename TFloat::T* HandleGradHess(typename TFloat::T* const pGradientAndHessian,
         const TFloat& sampleScore,
         const TFloat& target) const noexcept {
      const TObjective* const pObjective = static_cast<const TObjective*>(this);
      const GradientHessian<TFloat> gradientHessian = pObjective->CalcGradientHessian(sampleScore, target);
      const TFloat gradient = gradientHessian.GetGradient();
      const TFloat hessian = gradientHessian.GetHessian();
      gradient.Store(pGradientAndHessian);
      hessian.Store(pGradientAndHessian + TFloat::k_cSIMDPack);
      return pGradientAndHessian + (TFloat::k_cSIMDPack + TFloat::k_cSIMDPack);
   }
   template<typename TObjective, typename TFloat, bool bHessian, typename std::enable_if<!bHessian, int>::type = 0>
   GPU_DEVICE INLINE_ALWAYS typename TFloat::T* HandleGradHess(typename TFloat::T* const pGradientAndHessian,
         const TFloat& sampleScore,
         const TFloat& target) const noexcept {
      const TObjective* const pObjective = static_cast<const TObjective*>(this);
      const TFloat gradient = pObjective->CalcGradient(sampleScore, target);
      gradient.Store(pGradientAndHessian);
      return pGradientAndHessian + TFloat::k_cSIMDPack;
   }

   template<typename TObjective,
         bool bCollapsed,
         bool bValidation,
         bool bWeight,
         bool bHessian,
         bool bUseApprox,
         size_t cCompilerScores,
         int cCompilerPack>
   GPU_DEVICE NEVER_INLINE void ChildApplyUpdate(ApplyUpdateBridge* const pData) const {
      using TFloat = typename TObjective::TFloatInternal;
      const TObjective* const pObjective = static_cast<const TObjective*>(this);

      static_assert(k_oneScore == cCompilerScores, "We special case the classifiers so do not need to handle them");
      static_assert(!bValidation || !bHessian, "bHessian can only be true if bValidation is false");
      static_assert(bValidation || !bWeight, "bWeight can only be true if bValidation is true");

      static constexpr bool bFixedSizePack = k_cItemsPerBitPackUndefined != cCompilerPack;

#ifndef GPU_COMPILE
      EBM_ASSERT(nullptr != pData);
      EBM_ASSERT(nullptr != pData->m_aUpdateTensorScores);
      EBM_ASSERT(1 <= pData->m_cSamples);
      EBM_ASSERT(0 == pData->m_cSamples % size_t{TFloat::k_cSIMDPack});
      EBM_ASSERT(0 == pData->m_cSamples % size_t{(bFixedSizePack ? cCompilerPack : 1) * TFloat::k_cSIMDPack});
      EBM_ASSERT(nullptr != pData->m_aSampleScores);
      EBM_ASSERT(1 == pData->m_cScores);
      EBM_ASSERT(nullptr != pData->m_aTargets);
#endif // GPU_COMPILE

      const typename TFloat::T* const aUpdateTensorScores =
            reinterpret_cast<const typename TFloat::T*>(pData->m_aUpdateTensorScores);

      const size_t cSamples = pData->m_cSamples;

      typename TFloat::T* pSampleScore = reinterpret_cast<typename TFloat::T*>(pData->m_aSampleScores);
      const typename TFloat::T* const pSampleScoresEnd = pSampleScore + cSamples;

      int cBitsPerItemMax;
      int cShift;
      int cShiftReset;
      typename TFloat::TInt maskBits;
      const typename TFloat::TInt::T* pInputData;

      TFloat updateScore;

      if(bCollapsed) {
         updateScore = aUpdateTensorScores[0];
      } else {
         const int cItemsPerBitPack = GET_ITEMS_PER_BIT_PACK(cCompilerPack, pData->m_cPack);
#ifndef GPU_COMPILE
         EBM_ASSERT(1 <= cItemsPerBitPack);
         EBM_ASSERT(cItemsPerBitPack <= COUNT_BITS(typename TFloat::TInt::T));
#endif // GPU_COMPILE

         cBitsPerItemMax = GetCountBits<typename TFloat::TInt::T>(cItemsPerBitPack);
#ifndef GPU_COMPILE
         EBM_ASSERT(1 <= cBitsPerItemMax);
         EBM_ASSERT(cBitsPerItemMax <= COUNT_BITS(typename TFloat::TInt::T));
#endif // GPU_COMPILE

         maskBits = MakeLowMask<typename TFloat::TInt::T>(cBitsPerItemMax);

         pInputData = reinterpret_cast<const typename TFloat::TInt::T*>(pData->m_aPacked);
#ifndef GPU_COMPILE
         EBM_ASSERT(nullptr != pInputData);
#endif // GPU_COMPILE

         cShiftReset = (cItemsPerBitPack - 1) * cBitsPerItemMax;
         if(bFixedSizePack) {
            updateScore = TFloat::Load(aUpdateTensorScores, TFloat::TInt::Load(pInputData) & maskBits);
            pInputData += TFloat::TInt::k_cSIMDPack;
         } else {
            cShift = static_cast<int>((cSamples >> TFloat::k_cSIMDShift) % static_cast<size_t>(cItemsPerBitPack)) *
                  cBitsPerItemMax;
            updateScore = TFloat::Load(aUpdateTensorScores, (TFloat::TInt::Load(pInputData) >> cShift) & maskBits);
            cShift -= cBitsPerItemMax;
            if(cShift < 0) {
               cShift = cShiftReset;
               pInputData += TFloat::TInt::k_cSIMDPack;
            }
         }
      }

      const typename TFloat::T* pTargetData = reinterpret_cast<const typename TFloat::T*>(pData->m_aTargets);

      typename TFloat::T* pGradientAndHessian;
      const typename TFloat::T* pWeight;
      TFloat metricSum;
      if(bValidation) {
         if(bWeight) {
            pWeight = reinterpret_cast<const typename TFloat::T*>(pData->m_aWeights);
#ifndef GPU_COMPILE
            EBM_ASSERT(nullptr != pWeight);
#endif // GPU_COMPILE
         }
         metricSum = 0.0;
      } else {
         pGradientAndHessian = reinterpret_cast<typename TFloat::T*>(pData->m_aGradientsAndHessians);
#ifndef GPU_COMPILE
         EBM_ASSERT(nullptr != pGradientAndHessian);
#endif // GPU_COMPILE
      }
      do {
         // TODO: the speed of this loop can probably be improved by:
         //   1) fetch the score from memory (predictable load is fast)
         //   2) issue the gather operation FOR THE NEXT loop(unpredictable load is slow)
         //   3) move the fetched gather operation from the previous loop into a new register
         //   4) do the computation using the fetched score and updateScore from the previous loop iteration
         // This will allow the CPU to do the gathering operation in the background while it works on computation.
         // Probably we want to put the code below inside the loop into an inline function that we can call
         // either at the start during init or the end once the rest is done.. not sure which.

         typename TFloat::TInt iTensorBinCombined;
         if(!bCollapsed) {
            iTensorBinCombined = TFloat::TInt::Load(pInputData);
            pInputData += TFloat::TInt::k_cSIMDPack;
         }
         if(bFixedSizePack) {
            // If we have a fixed sized cCompilerPack then the compiler should be able to unroll
            // the loop below. The compiler can only do that though if it can guarantee that all
            // iterations of the loop have the name number of loops.  Setting cShift here allows this
            cShift = cShiftReset;
         }
         while(true) {
            TFloat sampleScore = TFloat::Load(pSampleScore);

            const TFloat target = TFloat::Load(pTargetData);
            pTargetData += TFloat::k_cSIMDPack;

            TFloat weight;
            if(bValidation) {
               if(bWeight) {
                  weight = TFloat::Load(pWeight);
                  pWeight += TFloat::k_cSIMDPack;
               }
            }

            typename TFloat::TInt iTensorBin;
            if(!bCollapsed) {
               iTensorBin = (iTensorBinCombined >> cShift) & maskBits;
            }
            sampleScore += updateScore;
            if(!bCollapsed) {
               updateScore = TFloat::Load(aUpdateTensorScores, iTensorBin);
            }

            sampleScore.Store(pSampleScore);
            pSampleScore += TFloat::k_cSIMDPack;

            if(bValidation) {
               TFloat metric = pObjective->CalcMetric(sampleScore, target);
               if(bWeight) {
                  metricSum = FusedMultiplyAdd(metric, weight, metricSum);
               } else {
                  metricSum += metric;
               }
            } else {
               pGradientAndHessian =
                     HandleGradHess<TObjective, TFloat, bHessian>(pGradientAndHessian, sampleScore, target);
            }

            if(bCollapsed) {
               if(pSampleScoresEnd == pSampleScore) {
                  break;
               }
            } else {
               cShift -= cBitsPerItemMax;
               if(cShift < 0) {
                  break;
               }
            }
         }
         if(bCollapsed) {
            break;
         }
         if(!bFixedSizePack) {
            cShift = cShiftReset;
         }
      } while(pSampleScoresEnd != pSampleScore);

      if(bValidation) {
         pData->m_metricOut += static_cast<double>(Sum(metricSum));
      }
   }

   template<typename TObjective>
   INLINE_RELEASE_TEMPLATED ErrorEbm ParentApplyUpdate(ApplyUpdateBridge* const pData) const {
      static_assert(
            IsEdgeObjective<TObjective>(), "TObjective must inherit from one of the children of the Objective class");
      if(k_cItemsPerBitPackUndefined == pData->m_cPack) {
         return OptionsApplyUpdate<TObjective, true>(pData);
      } else {
         return OptionsApplyUpdate<TObjective, false>(pData);
      }
   }

   template<typename TObjective,
         typename std::enable_if<AccelerationFlags_NONE == TObjective::TFloatInternal::k_zone &&
                     IdentifyTask(TObjective::k_linkFunction) == Task_Regression,
               int>::type = 0>
   INLINE_RELEASE_TEMPLATED BoolEbm TypeCheckTargets(const size_t c, const void* const aTargets) const noexcept {
      // regression
      EBM_ASSERT(1 <= c);
      const TObjective* const pObjective = static_cast<const TObjective*>(this);
      const FloatShared* pTarget = static_cast<const FloatShared*>(aTargets);
      const FloatShared* const pTargetEnd = &pTarget[c];
      do {
         if(pObjective->CheckRegressionTarget(static_cast<double>(*pTarget))) {
            return EBM_TRUE;
         }
         ++pTarget;
      } while(pTargetEnd != pTarget);
      return EBM_FALSE;
   }
   template<typename TObjective,
         typename std::enable_if<AccelerationFlags_NONE == TObjective::TFloatInternal::k_zone &&
                     IdentifyTask(TObjective::k_linkFunction) == Task_GeneralClassification,
               int>::type = 0>
   INLINE_RELEASE_TEMPLATED BoolEbm TypeCheckTargets(const size_t c, const void* const aTargets) const noexcept {
      // classification
      UNUSED(c);
      UNUSED(aTargets);
      return EBM_FALSE;
   }
   template<typename TObjective,
         typename std::enable_if<AccelerationFlags_NONE == TObjective::TFloatInternal::k_zone &&
                     IdentifyTask(TObjective::k_linkFunction) == Task_Ranking,
               int>::type = 0>
   INLINE_RELEASE_TEMPLATED BoolEbm TypeCheckTargets(const size_t c, const void* const aTargets) const noexcept {
      // classification
      UNUSED(c);
      UNUSED(aTargets);
      return EBM_FALSE;
   }
   template<typename TObjective,
         typename std::enable_if<AccelerationFlags_NONE == TObjective::TFloatInternal::k_zone, int>::type = 0>
   inline BoolEbm ParentCheckTargets(const size_t c, const void* const aTargets) const noexcept {
      static_assert(
            IsEdgeObjective<TObjective>(), "TObjective must inherit from one of the children of the Objective class");
      return TypeCheckTargets<TObjective>(c, aTargets);
   }

   template<typename TObjective,
         typename std::enable_if<AccelerationFlags_NONE == TObjective::TFloatInternal::k_zone, int>::type = 0>
   INLINE_RELEASE_TEMPLATED static void SetCpu(ObjectiveWrapper* const pObjectiveWrapper) noexcept {
      FunctionPointersCpp* const pFunctionPointers =
            static_cast<FunctionPointersCpp*>(pObjectiveWrapper->m_pFunctionPointersCpp);
      pFunctionPointers->m_pFinishMetricCpp = &TObjective::StaticFinishMetric;
      pFunctionPointers->m_pCheckTargetsCpp = &TObjective::StaticCheckTargets;
   }
   template<typename TObjective,
         typename std::enable_if<AccelerationFlags_NONE != TObjective::TFloatInternal::k_zone, int>::type = 0>
   INLINE_RELEASE_TEMPLATED static void SetCpu(ObjectiveWrapper* const pObjectiveWrapper) noexcept {
      FunctionPointersCpp* const pFunctionPointers =
            static_cast<FunctionPointersCpp*>(pObjectiveWrapper->m_pFunctionPointersCpp);
      pFunctionPointers->m_pFinishMetricCpp = nullptr;
      pFunctionPointers->m_pCheckTargetsCpp = nullptr;
   }

   template<typename TObjective>
   INLINE_RELEASE_TEMPLATED void FillObjectiveWrapper(const AccelerationFlags zones, void* const pWrapperOut) noexcept {
      EBM_ASSERT(nullptr != pWrapperOut);
      ObjectiveWrapper* const pObjectiveWrapperOut = static_cast<ObjectiveWrapper*>(pWrapperOut);
      FunctionPointersCpp* const pFunctionPointers =
            static_cast<FunctionPointersCpp*>(pObjectiveWrapperOut->m_pFunctionPointersCpp);
      EBM_ASSERT(nullptr != pFunctionPointers);

      pFunctionPointers->m_pApplyUpdateCpp = &TObjective::StaticApplyUpdate;

      const auto bMaximizeMetric = TObjective::k_bMaximizeMetric;
      constexpr bool bMaximizeMetricGood = std::is_same<decltype(bMaximizeMetric), const BoolEbm>::value;
      static_assert(bMaximizeMetricGood, "TObjective::k_bMaximizeMetric should be a BoolEbm");
      pObjectiveWrapperOut->m_bMaximizeMetric = bMaximizeMetric;

      pObjectiveWrapperOut->m_objective = TObjective::k_objective;
      pObjectiveWrapperOut->m_linkFunction = TObjective::k_linkFunction;

      const auto linkParam = (static_cast<TObjective*>(this))->LinkParam();
      constexpr bool bLinkParamGood = std::is_same<decltype(linkParam), const double>::value;
      static_assert(bLinkParamGood, "this->LinkParam() should return a double");
      pObjectiveWrapperOut->m_linkParam = linkParam;

      const auto learningRateAdjustmentDifferentialPrivacy =
            (static_cast<TObjective*>(this))->LearningRateAdjustmentDifferentialPrivacy();
      constexpr bool bLearningRateAdjustmentDifferentialPrivacyGood =
            std::is_same<decltype(learningRateAdjustmentDifferentialPrivacy), const double>::value;
      static_assert(bLearningRateAdjustmentDifferentialPrivacyGood,
            "this->LearningRateAdjustmentDifferentialPrivacy() should return a double");
      pObjectiveWrapperOut->m_learningRateAdjustmentDifferentialPrivacy = learningRateAdjustmentDifferentialPrivacy;

      const auto learningRateAdjustmentGradientBoosting =
            (static_cast<TObjective*>(this))->LearningRateAdjustmentGradientBoosting();
      constexpr bool bLearningRateAdjustmentGradientBoostingGood =
            std::is_same<decltype(learningRateAdjustmentGradientBoosting), const double>::value;
      static_assert(bLearningRateAdjustmentGradientBoostingGood,
            "this->LearningRateAdjustmentGradientBoosting() should return a double");
      pObjectiveWrapperOut->m_learningRateAdjustmentGradientBoosting = learningRateAdjustmentGradientBoosting;

      const auto learningRateAdjustmentHessianBoosting =
            (static_cast<TObjective*>(this))->LearningRateAdjustmentHessianBoosting();
      constexpr bool bLearningRateAdjustmentHessianBoostingGood =
            std::is_same<decltype(learningRateAdjustmentHessianBoosting), const double>::value;
      static_assert(bLearningRateAdjustmentHessianBoostingGood,
            "this->LearningRateAdjustmentHessianBoosting() should return a double");
      pObjectiveWrapperOut->m_learningRateAdjustmentHessianBoosting = learningRateAdjustmentHessianBoosting;

      const auto gainAdjustmentGradientBoosting = (static_cast<TObjective*>(this))->GainAdjustmentGradientBoosting();
      constexpr bool bGainAdjustmentGradientBoostingGood =
            std::is_same<decltype(gainAdjustmentGradientBoosting), const double>::value;
      static_assert(
            bGainAdjustmentGradientBoostingGood, "this->GainAdjustmentGradientBoosting() should return a double");
      pObjectiveWrapperOut->m_gainAdjustmentGradientBoosting = gainAdjustmentGradientBoosting;

      const auto gainAdjustmentHessianBoosting = (static_cast<TObjective*>(this))->GainAdjustmentHessianBoosting();
      constexpr bool bGainAdjustmentHessianBoostingGood =
            std::is_same<decltype(gainAdjustmentHessianBoosting), const double>::value;
      static_assert(bGainAdjustmentHessianBoostingGood, "this->GainAdjustmentHessianBoosting() should return a double");
      pObjectiveWrapperOut->m_gainAdjustmentHessianBoosting = gainAdjustmentHessianBoosting;

      const auto gradientConstant = (static_cast<TObjective*>(this))->GradientConstant();
      constexpr bool bGradientConstantGood = std::is_same<decltype(gradientConstant), const double>::value;
      static_assert(bGradientConstantGood, "this->GradientConstant() should return a double");
      pObjectiveWrapperOut->m_gradientConstant = gradientConstant;

      const auto hessianConstant = (static_cast<TObjective*>(this))->HessianConstant();
      constexpr bool bHessianConstantGood = std::is_same<decltype(hessianConstant), const double>::value;
      static_assert(bHessianConstantGood, "this->HessianConstant() should return a double");
      pObjectiveWrapperOut->m_hessianConstant = hessianConstant;

      pObjectiveWrapperOut->m_bObjectiveHasHessian = TObjective::k_bHessian ? EBM_TRUE : EBM_FALSE;

      pObjectiveWrapperOut->m_pObjective = this;

      pObjectiveWrapperOut->m_zones = zones;

      SetCpu<TObjective>(pObjectiveWrapperOut);
   }

   Objective() = default;
   ~Objective() = default;

 public:
   template<typename TFloat>
   static ErrorEbm CreateObjective(const Config* const pConfig,
         const char* const sObjective,
         const char* const sObjectiveEnd,
         ObjectiveWrapper* const pObjectiveWrapperOut) noexcept {
      EBM_ASSERT(nullptr != pConfig);
      EBM_ASSERT(1 <= pConfig->cOutputs);
      EBM_ASSERT(EBM_FALSE == pConfig->isDifferentialPrivacy || EBM_TRUE == pConfig->isDifferentialPrivacy);
      EBM_ASSERT(nullptr != sObjective);
      EBM_ASSERT(nullptr != sObjectiveEnd);
      EBM_ASSERT(sObjective < sObjectiveEnd); // empty string not allowed
      EBM_ASSERT('\0' != *sObjective);
      EBM_ASSERT(!(0x20 == *sObjective || (0x9 <= *sObjective && *sObjective <= 0xd)));
      EBM_ASSERT('\0' == *sObjectiveEnd);
      EBM_ASSERT(nullptr != pObjectiveWrapperOut);
      EBM_ASSERT(nullptr == pObjectiveWrapperOut->m_pObjective);
      EBM_ASSERT(nullptr != pObjectiveWrapperOut->m_pFunctionPointersCpp);

      LOG_0(Trace_Info, "Entered Objective::CreateObjective");

      ErrorEbm error;

      try {
         const std::vector<std::shared_ptr<const Registration>> registrations = RegisterObjectives<TFloat>();
         const bool bFailed =
               Registration::CreateRegistrable(pConfig, sObjective, sObjectiveEnd, pObjectiveWrapperOut, registrations);
         if(!bFailed) {
            EBM_ASSERT(nullptr != pObjectiveWrapperOut->m_pObjective);

            LOG_0(Trace_Info, "Exited Objective::CreateObjective");
            return Error_None;
         }
         EBM_ASSERT(nullptr == pObjectiveWrapperOut->m_pObjective);
         LOG_0(Trace_Info, "Exited Objective::CreateObjective unknown objective");
         error = Error_ObjectiveUnknown;
      } catch(const ParamValMalformedException&) {
         EBM_ASSERT(nullptr == pObjectiveWrapperOut->m_pObjective);
         LOG_0(Trace_Warning, "WARNING Objective::CreateObjective ParamValMalformedException");
         error = Error_ObjectiveParamValMalformed;
      } catch(const ParamUnknownException&) {
         EBM_ASSERT(nullptr == pObjectiveWrapperOut->m_pObjective);
         LOG_0(Trace_Warning, "WARNING Objective::CreateObjective ParamUnknownException");
         error = Error_ObjectiveParamUnknown;
      } catch(const RegistrationConstructorException&) {
         EBM_ASSERT(nullptr == pObjectiveWrapperOut->m_pObjective);
         LOG_0(Trace_Warning, "WARNING Objective::CreateObjective RegistrationConstructorException");
         error = Error_ObjectiveConstructorException;
      } catch(const ParamValOutOfRangeException&) {
         EBM_ASSERT(nullptr == pObjectiveWrapperOut->m_pObjective);
         LOG_0(Trace_Warning, "WARNING Objective::CreateObjective ParamValOutOfRangeException");
         error = Error_ObjectiveParamValOutOfRange;
      } catch(const ParamMismatchWithConfigException&) {
         EBM_ASSERT(nullptr == pObjectiveWrapperOut->m_pObjective);
         LOG_0(Trace_Warning, "WARNING Objective::CreateObjective ParamMismatchWithConfigException");
         error = Error_ObjectiveParamMismatchWithConfig;
      } catch(const IllegalRegistrationNameException&) {
         EBM_ASSERT(nullptr == pObjectiveWrapperOut->m_pObjective);
         LOG_0(Trace_Warning, "WARNING Objective::CreateObjective IllegalRegistrationNameException");
         error = Error_ObjectiveIllegalRegistrationName;
      } catch(const IllegalParamNameException&) {
         EBM_ASSERT(nullptr == pObjectiveWrapperOut->m_pObjective);
         LOG_0(Trace_Warning, "WARNING Objective::CreateObjective IllegalParamNameException");
         error = Error_ObjectiveIllegalParamName;
      } catch(const DuplicateParamNameException&) {
         EBM_ASSERT(nullptr == pObjectiveWrapperOut->m_pObjective);
         LOG_0(Trace_Warning, "WARNING Objective::CreateObjective DuplicateParamNameException");
         error = Error_ObjectiveDuplicateParamName;
      } catch(const NonPrivateRegistrationException&) {
         EBM_ASSERT(nullptr == pObjectiveWrapperOut->m_pObjective);
         LOG_0(Trace_Warning, "WARNING Objective::CreateObjective NonPrivateRegistrationException");
         error = Error_ObjectiveNonPrivate;
      } catch(const NonPrivateParamException&) {
         EBM_ASSERT(nullptr == pObjectiveWrapperOut->m_pObjective);
         LOG_0(Trace_Warning, "WARNING Objective::CreateObjective NonPrivateParamException");
         error = Error_ObjectiveParamNonPrivate;
      } catch(const std::bad_alloc&) {
         LOG_0(Trace_Warning, "WARNING Objective::CreateObjective Out of Memory");
         error = Error_OutOfMemory;
      } catch(...) {
         LOG_0(Trace_Warning, "WARNING Objective::CreateObjective internal error, unknown exception");
         error = Error_UnexpectedInternal;
      }

      return error;
   }
};
static_assert(std::is_standard_layout<Objective>::value && std::is_trivially_copyable<Objective>::value,
      "This allows offsetof, memcpy, memset, inter-language, GPU and cross-machine use where needed");

// TODO: include ranking
//
// We use the following terminology:
// Target      : the thing we're trying to predict.  For classification this is the label.  For regression this
//               is what we're predicting directly.  Target and Output seem to be used interchangeably in other
//               packages.  We choose Target here.
// Score       : the values we use to generate predictions.  For classification these are logits.  For regression these
//               are the predictions themselves.  For multiclass there are N scores per target when there are N classes.
//               For multiclass you could eliminate one score to get N-1 scores, but we don't use that trick in this
//               package yet.
// Prediction  : the prediction of the model.  We output scores in our model and generate predictions from them.
//               For multiclass the scores are the logits, and the predictions would be the outputs of softmax.
//               We have N scores per target for an N class multiclass problem.
// Binary      : binary classification.  Target is 0 or 1
// Multiclass  : multiclass classification.  Target is 0, 1, 2, ...
// Regression  : regression
// Multioutput : a model that can predict multiple different things.  A single model could predict binary,
//               multiclass, regression, etc. different targets.
// Multitask   : A slightly more restricted form of multioutput where training jointly optimizes the targets.
//               The different targets can still be of different types (binary, multiclass, regression, etc), but
//               importantly they share a single objective.  In C++ we deal only with multitask since otherwise
//               it would make more sense to train the targets separately.  In higher level languages the models can
//               either be Multitask or Multioutput depending on how they were generated.
// Multilabel  : A more restricted version of multitask where the tasks are all binary classification.  We use
//               the term MultitaskBinary* here since it fits better into our ontology.
//
// The most general objective that we could handle in C++ would be to take a custom objective that jointly
// optimizes a multitask problem that contains regression, binary, and multiclass tasks.  This would be:
// "CustomMultitaskObjective"

struct SingletaskObjective : public Objective {
 protected:
   SingletaskObjective() = default;
   ~SingletaskObjective() = default;
};

struct BinaryObjective : public SingletaskObjective {
 protected:
   BinaryObjective() = default;
   ~BinaryObjective() = default;

 public:
   static constexpr bool IsMultiScore = false;
};

struct MulticlassObjective : public SingletaskObjective {
 protected:
   MulticlassObjective() = default;
   ~MulticlassObjective() = default;

 public:
   static constexpr bool IsMultiScore = true;
};

struct RegressionObjective : public SingletaskObjective {
 protected:
   RegressionObjective() = default;
   ~RegressionObjective() = default;

 public:
   static constexpr bool IsMultiScore = false;
};

struct MultitaskObjective : public Objective {
 protected:
   MultitaskObjective() = default;
   ~MultitaskObjective() = default;

 public:
   static constexpr bool IsMultiScore = true;
};

struct BinaryMultitaskObjective : public MultitaskObjective {
 protected:
   BinaryMultitaskObjective() = default;
   ~BinaryMultitaskObjective() = default;
};

struct MulticlassMultitaskObjective : public MultitaskObjective {
 protected:
   MulticlassMultitaskObjective() = default;
   ~MulticlassMultitaskObjective() = default;
};

struct RegressionMultitaskObjective : public MultitaskObjective {
 protected:
   RegressionMultitaskObjective() = default;
   ~RegressionMultitaskObjective() = default;
};

#define OBJECTIVE_CONSTANTS_BOILERPLATE(__EBM_TYPE,                                                                    \
      __MAXIMIZE_METRIC,                                                                                               \
      __OBJECTIVE,                                                                                                     \
      __LINK_FUNCTION,                                                                                                 \
      bHessian,                                                                                                        \
      bHasApprox,                                                                                                      \
      cItemsPerBitPackMax,                                                                                             \
      cItemsPerBitPackMin)                                                                                             \
 public:                                                                                                               \
   using TFloatInternal = TFloat;                                                                                      \
   static constexpr bool k_bHessian = (bHessian);                                                                      \
   static constexpr bool k_bHasApprox = (bHasApprox);                                                                  \
   static constexpr BoolEbm k_bMaximizeMetric = (__MAXIMIZE_METRIC);                                                   \
   static constexpr ObjectiveEbm k_objective = (__OBJECTIVE);                                                          \
   static constexpr LinkEbm k_linkFunction = (__LINK_FUNCTION);                                                        \
   static constexpr int k_cItemsPerBitPackMax = (cItemsPerBitPackMax);                                                 \
   static constexpr int k_cItemsPerBitPackMin = (cItemsPerBitPackMin);                                                 \
   static ErrorEbm StaticApplyUpdate(const Objective* const pThis, ApplyUpdateBridge* const pData) {                   \
      return (static_cast<const __EBM_TYPE<TFloat>*>(pThis))->ParentApplyUpdate<const __EBM_TYPE<TFloat>>(pData);      \
   }                                                                                                                   \
   template<typename T = void, typename std::enable_if<AccelerationFlags_NONE == TFloat::k_zone, T>::type* = nullptr>  \
   static double StaticFinishMetric(const Objective* const pThis, const double metricSum) {                            \
      return (static_cast<const __EBM_TYPE<TFloat>*>(pThis))->FinishMetric(metricSum);                                 \
   }                                                                                                                   \
   template<typename T = void, typename std::enable_if<AccelerationFlags_NONE == TFloat::k_zone, T>::type* = nullptr>  \
   static BoolEbm StaticCheckTargets(const Objective* const pThis, const size_t c, const void* const aTargets) {       \
      return (static_cast<const __EBM_TYPE<TFloat>*>(pThis))                                                           \
            ->ParentCheckTargets<const __EBM_TYPE<TFloat>>(c, aTargets);                                               \
   }                                                                                                                   \
   void FillWrapper(const AccelerationFlags zones, void* const pWrapperOut) noexcept {                                 \
      static_assert(std::is_same<__EBM_TYPE<TFloat>, typename std::remove_pointer<decltype(this)>::type>::value,       \
            "*Objective types mismatch");                                                                              \
      FillObjectiveWrapper<typename std::remove_pointer<decltype(this)>::type>(zones, pWrapperOut);                    \
   }

#define OBJECTIVE_TEMPLATE_BOILERPLATE                                                                                 \
 public:                                                                                                               \
   template<bool bCollapsed,                                                                                           \
         bool bValidation,                                                                                             \
         bool bWeight,                                                                                                 \
         bool bHessian,                                                                                                \
         bool bUseApprox,                                                                                              \
         size_t cCompilerScores,                                                                                       \
         int cCompilerPack>                                                                                            \
   GPU_DEVICE void InjectedApplyUpdate(ApplyUpdateBridge* const pData) const {                                         \
      Objective::ChildApplyUpdate<typename std::remove_pointer<decltype(this)>::type,                                  \
            bCollapsed,                                                                                                \
            bValidation,                                                                                               \
            bWeight,                                                                                                   \
            bHessian,                                                                                                  \
            bUseApprox,                                                                                                \
            cCompilerScores,                                                                                           \
            cCompilerPack>(pData);                                                                                     \
   }

#define OBJECTIVE_BOILERPLATE(__EBM_TYPE, __MAXIMIZE_METRIC, __OBJECTIVE, __LINK_FUNCTION, bHessian)                   \
   OBJECTIVE_CONSTANTS_BOILERPLATE(__EBM_TYPE,                                                                         \
         __MAXIMIZE_METRIC,                                                                                            \
         __OBJECTIVE,                                                                                                  \
         __LINK_FUNCTION,                                                                                              \
         bHessian,                                                                                                     \
         false,                                                                                                        \
         k_cItemsPerBitPackUndefined,                                                                                  \
         k_cItemsPerBitPackUndefined)                                                                                  \
   OBJECTIVE_TEMPLATE_BOILERPLATE

} // namespace DEFINED_ZONE_NAME

#endif // OBJECTIVE_HPP
