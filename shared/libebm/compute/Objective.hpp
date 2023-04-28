// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// !!! NOTE: To add a new objective in C++, follow the steps listed at the top of the "objective_registrations.hpp" file !!!

#ifndef OBJECTIVE_HPP
#define OBJECTIVE_HPP

#include <stddef.h> // size_t, ptrdiff_t
#include <memory> // shared_ptr, unique_ptr

#include "libebm.h" // ErrorEbm
#include "logging.h" // EBM_ASSERT
#include "common_c.h" // INLINE_ALWAYS
#include "zones.h"

#include "zoned_bridge_cpp_functions.hpp" // FunctionPointersCpp
#include "compute.hpp" // GPU_GLOBAL

struct ApplyUpdateBridge;

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

class Registration;
typedef const std::vector<std::shared_ptr<const Registration>> (* REGISTER_OBJECTIVES_FUNCTION)();

struct SingletaskObjective;
struct BinaryObjective;
struct MulticlassObjective;
struct RegressionObjective;

struct MultitaskObjective;
struct BinaryMultitaskObjective;
struct MulticlassMultitaskObjective;
struct RegressionMultitaskObjective;


template<typename TFloat>
class GradientHessian {
   TFloat m_gradient;
   TFloat m_hessian;

public:
   template<typename T1, typename T2>
   GPU_DEVICE inline GradientHessian(const T1 gradient, const T2 hessian) : 
      m_gradient(TFloat(gradient)), 
      m_hessian(TFloat(hessian)) {
   }

   GPU_DEVICE inline TFloat GetGradient() const noexcept { return m_gradient; }
   GPU_DEVICE inline TFloat GetHessian() const noexcept { return m_hessian; }
};
template<typename TFloat>
GPU_DEVICE inline GradientHessian<TFloat> MakeGradientHessian(const TFloat gradient, const TFloat hessian) {
   return GradientHessian<TFloat>(gradient, hessian);
}
template<typename TFloat>
GPU_DEVICE inline GradientHessian<TFloat> MakeGradientHessian(const TFloat gradient, const double hessian) {
   return GradientHessian<TFloat>(gradient, hessian);
}
template<typename TFloat>
GPU_DEVICE inline GradientHessian<TFloat> MakeGradientHessian(const double gradient, const TFloat hessian) {
   return GradientHessian<TFloat>(gradient, hessian);
}
template<typename TFloat>
GPU_DEVICE inline GradientHessian<TFloat> MakeGradientHessian(const double gradient, const double hessian) {
   return GradientHessian<TFloat>(gradient, hessian);
}


template<typename TObjective, size_t cCompilerScores, ptrdiff_t cCompilerPack, bool bHessian, bool bKeepGradHess, bool bCalcMetric, bool bWeight>
GPU_GLOBAL static void RemoteApplyUpdate(const Objective * const pObjective, ApplyUpdateBridge * const pData) {
   const TObjective * const pObjectiveSpecific = static_cast<const TObjective *>(pObjective);
   pObjectiveSpecific->template InjectedApplyUpdate<cCompilerScores, cCompilerPack, bHessian, bKeepGradHess, bCalcMetric, bWeight>(pData);
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

   template<class TObjective, typename TFloat>
   struct HasHessianInternal {
      // use SFINAE to determine if TObjective has the function CalcGradientHessian with the correct signature

      template<typename T>
      static auto check(T * p) -> decltype(p->CalcGradientHessian(TFloat { 0 }, TFloat { 0 }), std::true_type());

      static std::false_type check(...);

      using internal_type = decltype(check(static_cast<typename std::remove_reference<TObjective>::type *>(nullptr)));
      static constexpr bool value = internal_type::value;
   };
   template<typename TObjective, typename TFloat>
   constexpr static bool HasHessian() {
      // use SFINAE to determine if TObjective has the function CalcGradientHessian with the correct signature
      return HasHessianInternal<TObjective, TFloat>::value;
   }

   template<typename TObjective>
   constexpr static bool IsEdgeObjective() {
      return
         std::is_base_of<BinaryObjective, TObjective>::value ||
         std::is_base_of<MulticlassObjective, TObjective>::value ||
         std::is_base_of<RegressionObjective, TObjective>::value ||
         std::is_base_of<BinaryMultitaskObjective, TObjective>::value ||
         std::is_base_of<MulticlassMultitaskObjective, TObjective>::value ||
         std::is_base_of<RegressionMultitaskObjective, TObjective>::value;
   }


   // if we have multiple scores AND multiple bitpacks, then we have two nested loops in our final function
   // and the compiler will only unroll the inner loop.  That inner loop will be for the scores, so there
   // is not much benefit in generating hard coded loop counts for the bitpacks, so short circut the
   // bit packing to use the dynamic value if we don't have the single bin case.  This also solves
   // part of our template blowup issue of having N * M starting point templates where N is the number
   // of scores and M is the number of bit packs.  If we use 8 * 16 that's already 128 copies of the
   // templated function at this point and more later.  Reducing this to just 16 is very very helpful.
   template<typename TObjective, typename TFloat, typename std::enable_if<!TObjective::IsMultiScore, void>::type * = nullptr>
   INLINE_RELEASE_TEMPLATED ErrorEbm TypeApplyUpdate(ApplyUpdateBridge * const pData) const {
      if(k_cItemsPerBitPackNone == pData->m_cPack) {
         return HessianApplyUpdate<TObjective, TFloat, k_oneScore, k_cItemsPerBitPackNone>(pData);
      } else {
         return BitPack<TObjective, TFloat, k_oneScore, k_cItemsPerBitPackMax>::Func(this, pData);
      }
   }
   template<typename TObjective, typename TFloat, typename std::enable_if<TObjective::IsMultiScore && std::is_base_of<MulticlassMultitaskObjective, TObjective>::value, void>::type * = nullptr>
   INLINE_RELEASE_TEMPLATED ErrorEbm TypeApplyUpdate(ApplyUpdateBridge * const pData) const {
      if(k_cItemsPerBitPackNone == pData->m_cPack) {
         // don't blow up our complexity if we have only 1 bin.. just use dynamic for the count of scores
         return HessianApplyUpdate<TObjective, TFloat, k_dynamicScores, k_cItemsPerBitPackNone>(pData);
      } else {
         // if our inner loop is dynamic scores, then the compiler won't do a full unwind of the bit pack
         // loop, so just short circuit it to using dynamic
         return HessianApplyUpdate<TObjective, TFloat, k_dynamicScores, k_cItemsPerBitPackDynamic>(pData);
      }
   }
   template<typename TObjective, typename TFloat, typename std::enable_if<TObjective::IsMultiScore && !std::is_base_of<MulticlassMultitaskObjective, TObjective>::value, void>::type * = nullptr>
   INLINE_RELEASE_TEMPLATED ErrorEbm TypeApplyUpdate(ApplyUpdateBridge * const pData) const {
      if(k_cItemsPerBitPackNone == pData->m_cPack) {
         // don't blow up our complexity if we have only 1 bin.. just use dynamic for the count of scores
         return HessianApplyUpdate<TObjective, TFloat, k_dynamicScores, k_cItemsPerBitPackNone>(pData);
      } else {
         return CountScores<TObjective, TFloat, (k_cCompilerScoresMax < k_cCompilerScoresStart ? k_dynamicScores : k_cCompilerScoresStart)>::Func(this, pData);
      }
   }


   template<typename TObjective, typename TFloat, size_t cCompilerScores>
   struct CountScores final {
      INLINE_ALWAYS static ErrorEbm Func(const Objective * const pObjective, ApplyUpdateBridge * const pData) {
         if(cCompilerScores == pData->m_cScores) {
            return pObjective->HessianApplyUpdate<TObjective, TFloat, cCompilerScores, k_cItemsPerBitPackDynamic>(pData);
         } else {
            return CountScores<TObjective, TFloat, k_cCompilerScoresMax == cCompilerScores ? k_dynamicScores : cCompilerScores + 1>::Func(pObjective, pData);
         }
      }
   };
   template<typename TObjective, typename TFloat>
   struct CountScores<TObjective, TFloat, k_dynamicScores> final {
      INLINE_ALWAYS static ErrorEbm Func(const Objective * const pObjective, ApplyUpdateBridge * const pData) {
         return pObjective->HessianApplyUpdate<TObjective, TFloat, k_dynamicScores, k_cItemsPerBitPackDynamic>(pData);
      }
   };


   // in our current format cCompilerScores will always be 1, but just in case we change our code to allow
   // for special casing multiclass with compile time unrolling of the compiler pack, leave cCompilerScores here
   template<typename TObjective, typename TFloat, size_t cCompilerScores, ptrdiff_t cCompilerPack>
   struct BitPack final {
      INLINE_ALWAYS static ErrorEbm Func(const Objective * const pObjective, ApplyUpdateBridge * const pData) {
         if(cCompilerPack == pData->m_cPack) {
            return pObjective->HessianApplyUpdate<TObjective, TFloat, cCompilerScores, cCompilerPack>(pData);
         } else {
            return BitPack<TObjective, TFloat, cCompilerScores, GetNextBitPack(cCompilerPack)>::Func(pObjective, pData);
         }
      }
   };
   template<typename TObjective, typename TFloat, size_t cCompilerScores>
   struct BitPack<TObjective, TFloat, cCompilerScores, k_cItemsPerBitPackLast> final {
      INLINE_ALWAYS static ErrorEbm Func(const Objective * const pObjective, ApplyUpdateBridge * const pData) {
         return pObjective->HessianApplyUpdate<TObjective, TFloat, cCompilerScores, k_cItemsPerBitPackLast>(pData);
      }
   };


   template<typename TObjective, typename TFloat, size_t cCompilerScores, ptrdiff_t cCompilerPack, typename std::enable_if<HasHessian<TObjective, TFloat>(), void>::type * = nullptr>
   INLINE_RELEASE_TEMPLATED ErrorEbm HessianApplyUpdate(ApplyUpdateBridge * const pData) const {
      if(pData->m_bHessianNeeded) {
         return OptionsApplyUpdate<TObjective, TFloat, cCompilerScores, cCompilerPack, true>(pData);
      } else {
         return OptionsApplyUpdate<TObjective, TFloat, cCompilerScores, cCompilerPack, false>(pData);
      }
   }
   template<typename TObjective, typename TFloat, size_t cCompilerScores, ptrdiff_t cCompilerPack, typename std::enable_if<!HasHessian<TObjective, TFloat>(), void>::type * = nullptr>
   INLINE_RELEASE_TEMPLATED ErrorEbm HessianApplyUpdate(ApplyUpdateBridge * const pData) const {
      return OptionsApplyUpdate<TObjective, TFloat, cCompilerScores, cCompilerPack, false>(pData);
   }


   template<typename TObjective, typename TFloat, size_t cCompilerScores, ptrdiff_t cCompilerPack, bool bHessian, typename std::enable_if<!TObjective::k_bRmse, void>::type * = nullptr>
   INLINE_RELEASE_TEMPLATED ErrorEbm OptionsApplyUpdate(ApplyUpdateBridge * const pData) const {
      if(nullptr != pData->m_aGradientsAndHessians) {
         static constexpr bool bKeepGradHess = true;

         // if we are updating the gradients then we are doing training and do not need to calculate the metric
         EBM_ASSERT(!pData->m_bCalcMetric);
         static constexpr bool bCalcMetric = false;

         if(nullptr != pData->m_aWeights) {
            static constexpr bool bWeight = true;

            // this branch will only be taking during interaction initialization

            return OperatorApplyUpdate<TObjective, TFloat, cCompilerScores, cCompilerPack, bHessian, bKeepGradHess, bCalcMetric, bWeight>(pData);
         } else {
            static constexpr bool bWeight = false;
            return OperatorApplyUpdate<TObjective, TFloat, cCompilerScores, cCompilerPack, bHessian, bKeepGradHess, bCalcMetric, bWeight>(pData);
         }
      } else {
         static constexpr bool bKeepGradHess = false;

         if(pData->m_bCalcMetric) {
            static constexpr bool bCalcMetric = true;

            if(nullptr != pData->m_aWeights) {
               static constexpr bool bWeight = true;
               return OperatorApplyUpdate<TObjective, TFloat, cCompilerScores, cCompilerPack, bHessian, bKeepGradHess, bCalcMetric, bWeight>(pData);
            } else {
               static constexpr bool bWeight = false;
               return OperatorApplyUpdate<TObjective, TFloat, cCompilerScores, cCompilerPack, bHessian, bKeepGradHess, bCalcMetric, bWeight>(pData);
            }
         } else {
            static constexpr bool bCalcMetric = false;

            // currently this branch is not taken, but if would be if we wanted to allow in the future
            // non-metric calculating validation for boosting.  For instance if we wanted to substitute an alternate
            // metric or if for performance reasons we only want to calculate the metric every N rounds of boosting

            EBM_ASSERT(nullptr == pData->m_aWeights);
            static constexpr bool bWeight = false; // if we are not calculating the metric or updating gradients then we never need the weights

            return OperatorApplyUpdate<TObjective, TFloat, cCompilerScores, cCompilerPack, bHessian, bKeepGradHess, bCalcMetric, bWeight>(pData);
         }
      }
   }
   template<typename TObjective, typename TFloat, size_t cCompilerScores, ptrdiff_t cCompilerPack, bool bHessian, typename std::enable_if<TObjective::k_bRmse, void>::type * = nullptr>
   INLINE_RELEASE_TEMPLATED ErrorEbm OptionsApplyUpdate(ApplyUpdateBridge * const pData) const {
      EBM_ASSERT(nullptr != pData->m_aGradientsAndHessians); // we always keep gradients for regression
      static constexpr bool bKeepGradHess = true;

      if(pData->m_bCalcMetric) {
         static constexpr bool bCalcMetric = true;

         if(nullptr != pData->m_aWeights) {
            static constexpr bool bWeight = true;
            return OperatorApplyUpdate<TObjective, TFloat, cCompilerScores, cCompilerPack, bHessian, bKeepGradHess, bCalcMetric, bWeight>(pData);
         } else {
            static constexpr bool bWeight = false;
            return OperatorApplyUpdate<TObjective, TFloat, cCompilerScores, cCompilerPack, bHessian, bKeepGradHess, bCalcMetric, bWeight>(pData);
         }
      } else {
         static constexpr bool bCalcMetric = false;

         EBM_ASSERT(nullptr == pData->m_aWeights);
         static constexpr bool bWeight = false; // if we are not calculating the metric then we never need the weights

         return OperatorApplyUpdate<TObjective, TFloat, cCompilerScores, cCompilerPack, bHessian, bKeepGradHess, bCalcMetric, bWeight>(pData);
      }
   }


   template<typename TObjective, typename TFloat, size_t cCompilerScores, ptrdiff_t cCompilerPack, bool bHessian, bool bKeepGradHess, bool bCalcMetric, bool bWeight>
   INLINE_RELEASE_TEMPLATED ErrorEbm OperatorApplyUpdate(ApplyUpdateBridge * const pData) const {
      return TFloat::template OperatorApplyUpdate<TObjective, cCompilerScores, cCompilerPack, bHessian, bKeepGradHess, bCalcMetric, bWeight>(this, pData);
   }

protected:

   template<typename TObjective, typename TFloat, bool bHessian, bool bWeight, typename std::enable_if<bHessian, void>::type * = nullptr>
   INLINE_ALWAYS typename TFloat::T * HandleGradHess(
      typename TFloat::T * const pGradientAndHessian,
      const TFloat sampleScore,
      const TFloat target,
      const TFloat weight
   ) const noexcept {
      const TObjective * const pObjective = static_cast<const TObjective *>(this);
      const GradientHessian<TFloat> gradientHessian = pObjective->CalcGradientHessian(sampleScore, target);
      TFloat gradient = gradientHessian.GetGradient();
      TFloat hessian = gradientHessian.GetHessian();
      if(bWeight) {
         // This is only used during the initialization of interaction detection. For boosting
         // we currently multiply by the weight during bin summation instead since we use the weight
         // there to include the inner bagging counts of occurences.
         // Whether this multiplication happens or not is controlled by the caller by passing in the
         // weight array or not.
         gradient *= weight;
         hessian *= weight;
      }
      gradient.SaveAligned(pGradientAndHessian);
      hessian.SaveAligned(pGradientAndHessian + TFloat::cPack);
      return pGradientAndHessian + (TFloat::cPack + TFloat::cPack);
   }
   template<typename TObjective, typename TFloat, bool bHessian, bool bWeight, typename std::enable_if<!bHessian, void>::type * = nullptr>
   INLINE_ALWAYS typename TFloat::T * HandleGradHess(
      typename TFloat::T * const pGradientAndHessian, 
      const TFloat sampleScore, 
      const TFloat target,
      const TFloat weight
   ) const noexcept {
      const TObjective * const pObjective = static_cast<const TObjective *>(this);
      TFloat gradient = pObjective->CalcGradient(sampleScore, target);
      if(bWeight) {
         // This is only used during the initialization of interaction detection. For boosting
         // we currently multiply by the weight during bin summation instead since we use the weight
         // there to include the inner bagging counts of occurences.
         // Whether this multiplication happens or not is controlled by the caller by passing in the
         // weight array or not.
         gradient *= weight;
      }
      gradient.SaveAligned(pGradientAndHessian);
      return pGradientAndHessian + TFloat::cPack;
   }

   template<typename TObjective, typename TFloat, size_t cCompilerScores, ptrdiff_t cCompilerPack, bool bHessian, bool bKeepGradHess, bool bCalcMetric, bool bWeight>
   GPU_DEVICE void ChildApplyUpdate(ApplyUpdateBridge * const pData) const {
      const TObjective * const pObjective = static_cast<const TObjective *>(this);

      static_assert(k_oneScore == cCompilerScores, "We special case the classifiers so do not need to handle them");
      static constexpr bool bCompilerZeroDimensional = k_cItemsPerBitPackNone == cCompilerPack;
      static constexpr bool bGetTarget = bCalcMetric || bKeepGradHess;

      const typename TFloat::T * const aUpdateTensorScores = reinterpret_cast<const typename TFloat::T *>(pData->m_aUpdateTensorScores);

      const size_t cSamples = pData->m_cSamples;

      typename TFloat::T * pSampleScore = reinterpret_cast<typename TFloat::T *>(pData->m_aSampleScores);
      const typename TFloat::T * const pSampleScoresEnd = pSampleScore + cSamples;

      size_t cBitsPerItemMax;
      ptrdiff_t cShift;
      ptrdiff_t cShiftReset;
      size_t maskBits;
      const StorageDataType * pInputData;

      alignas(16) typename TFloat::T updateScores[TFloat::cPack];
      TFloat updateScore;

      if(bCompilerZeroDimensional) {
         const typename TFloat::T singleScore = aUpdateTensorScores[0];
         for(int i = 0; i < TFloat::cPack; ++i) {
            updateScores[i] = singleScore;
         }
         updateScore.LoadAligned(updateScores);
      } else {
         const ptrdiff_t cPack = GET_ITEMS_PER_BIT_PACK(cCompilerPack, pData->m_cPack);

         const size_t cItemsPerBitPack = static_cast<size_t>(cPack);

         cBitsPerItemMax = GetCountBits<StorageDataType>(cItemsPerBitPack);

         cShift = static_cast<ptrdiff_t>((cSamples - 1) % cItemsPerBitPack * cBitsPerItemMax);
         cShiftReset = static_cast<ptrdiff_t>((cItemsPerBitPack - 1) * cBitsPerItemMax);

         maskBits = static_cast<size_t>(MakeLowMask<StorageDataType>(cBitsPerItemMax));

         pInputData = pData->m_aPacked;
      }

      const typename TFloat::T * pTargetData;
      if(bGetTarget) {
         pTargetData = reinterpret_cast<const typename TFloat::T *>(pData->m_aTargets);
      }

      typename TFloat::T * pGradientAndHessian;
      if(bKeepGradHess) {
         pGradientAndHessian = reinterpret_cast<typename TFloat::T *>(pData->m_aGradientsAndHessians);
      }

      const typename TFloat::T * pWeight;
      if(bWeight) {
         pWeight = reinterpret_cast<const typename TFloat::T *>(pData->m_aWeights);
      }

      TFloat metricSum;
      if(bCalcMetric) {
         metricSum = 0.0;
      }
      do {
         alignas(16) StorageDataType iTensorBinCombined[TFloat::cPack];
         if(!bCompilerZeroDimensional) {
            // we store the already multiplied dimensional value in *pInputData
            for(int i = 0; i < TFloat::cPack; ++i) {
               iTensorBinCombined[i] = pInputData[i];
            }
            pInputData += TFloat::cPack;
         }
         while(true) {
            if(!bCompilerZeroDimensional) {
               // in later versions of SIMD there are scatter/gather intrinsics that do this in one operation
               for(int i = 0; i < TFloat::cPack; ++i) {
                  const size_t iTensorBin = static_cast<size_t>(iTensorBinCombined[i] >> cShift) & maskBits;
                  updateScores[i] = aUpdateTensorScores[iTensorBin];
               }
               updateScore.LoadAligned(updateScores);
            }

            TFloat target;
            if(bGetTarget) {
               target.LoadAligned(pTargetData);
               pTargetData += TFloat::cPack;
            }

            TFloat sampleScore;
            sampleScore.LoadAligned(pSampleScore);
            sampleScore += updateScore;
            sampleScore.SaveAligned(pSampleScore);
            pSampleScore += TFloat::cPack;

            TFloat weight;
            if(bWeight) {
               weight.LoadAligned(pWeight);
               pWeight += TFloat::cPack;
            }

            if(bKeepGradHess) {
               pGradientAndHessian = HandleGradHess<TObjective, TFloat, bHessian, bWeight>(pGradientAndHessian, sampleScore, target, weight);
            }

            if(bCalcMetric) {
               TFloat metric = pObjective->CalcMetric(sampleScore, target);
               if(bWeight) {
                  metric *= weight;
               }
               metricSum += metric;
            }

            if(bCompilerZeroDimensional) {
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
         if(bCompilerZeroDimensional) {
            break;
         }
         cShift = cShiftReset;
      } while(pSampleScoresEnd != pSampleScore);

      if(bCalcMetric) {
         pData->m_metricOut = static_cast<double>(Sum(metricSum));
      }
   }


   template<typename TObjective, typename TFloat>
   INLINE_RELEASE_TEMPLATED ErrorEbm ParentApplyUpdate(ApplyUpdateBridge * const pData) const {
      static_assert(IsEdgeObjective<TObjective>(), "TObjective must inherit from one of the children of the Objective class");
      return TypeApplyUpdate<TObjective, TFloat>(pData);
   }

   template<typename TObjective, typename TFloat, typename std::enable_if<TFloat::bCpu, void>::type * = nullptr>
   INLINE_RELEASE_TEMPLATED static void SetFinishMetric(FunctionPointersCpp * const pFunctionPointers) noexcept {
      pFunctionPointers->m_pFinishMetricCpp = &TObjective::StaticFinishMetric;
   }
   template<typename TObjective, typename TFloat, typename std::enable_if<!TFloat::bCpu, void>::type * = nullptr>
   INLINE_RELEASE_TEMPLATED static void SetFinishMetric(FunctionPointersCpp * const pFunctionPointers) noexcept {
      pFunctionPointers->m_pFinishMetricCpp = nullptr;
   }

   template<typename TObjective, typename TFloat>
   INLINE_RELEASE_TEMPLATED void FillObjectiveWrapper(void * const pWrapperOut) noexcept {
      EBM_ASSERT(nullptr != pWrapperOut);
      ObjectiveWrapper * const pObjectiveWrapperOut = static_cast<ObjectiveWrapper *>(pWrapperOut);
      FunctionPointersCpp * const pFunctionPointers =
         static_cast<FunctionPointersCpp *>(pObjectiveWrapperOut->m_pFunctionPointersCpp);
      EBM_ASSERT(nullptr != pFunctionPointers);

      pFunctionPointers->m_pApplyUpdateCpp = &TObjective::StaticApplyUpdate;
      
      const auto bMaximizeMetric = TObjective::k_bMaximizeMetric;
      static_assert(std::is_same<decltype(bMaximizeMetric), const BoolEbm>::value, "TObjective::k_bMaximizeMetric should be a BoolEbm");
      pObjectiveWrapperOut->m_bMaximizeMetric = bMaximizeMetric;

      pObjectiveWrapperOut->m_linkFunction = TObjective::k_linkFunction;

      const auto linkParam = (static_cast<TObjective *>(this))->LinkParam();
      static_assert(std::is_same<decltype(linkParam), const double>::value, "this->LinkParam() should return a double");
      pObjectiveWrapperOut->m_linkParam = linkParam;

      const auto learningRateAdjustmentDifferentialPrivacy = (static_cast<TObjective *>(this))->LearningRateAdjustmentDifferentialPrivacy();
      static_assert(std::is_same<decltype(learningRateAdjustmentDifferentialPrivacy), const double>::value, 
         "this->LearningRateAdjustmentDifferentialPrivacy() should return a double");
      pObjectiveWrapperOut->m_learningRateAdjustmentDifferentialPrivacy = learningRateAdjustmentDifferentialPrivacy;

      const auto learningRateAdjustmentGradientBoosting = (static_cast<TObjective *>(this))->LearningRateAdjustmentGradientBoosting();
      static_assert(std::is_same<decltype(learningRateAdjustmentGradientBoosting), const double>::value, 
         "this->LearningRateAdjustmentGradientBoosting() should return a double");
      pObjectiveWrapperOut->m_learningRateAdjustmentGradientBoosting = learningRateAdjustmentGradientBoosting;

      const auto learningRateAdjustmentHessianBoosting = (static_cast<TObjective *>(this))->LearningRateAdjustmentHessianBoosting();
      static_assert(std::is_same<decltype(learningRateAdjustmentHessianBoosting), const double>::value, 
         "this->LearningRateAdjustmentHessianBoosting() should return a double");
      pObjectiveWrapperOut->m_learningRateAdjustmentHessianBoosting = learningRateAdjustmentHessianBoosting;

      const auto gainAdjustmentGradientBoosting = (static_cast<TObjective *>(this))->GainAdjustmentGradientBoosting();
      static_assert(std::is_same<decltype(gainAdjustmentGradientBoosting), const double>::value, 
         "this->GainAdjustmentGradientBoosting() should return a double");
      pObjectiveWrapperOut->m_gainAdjustmentGradientBoosting = gainAdjustmentGradientBoosting;

      const auto gainAdjustmentHessianBoosting = (static_cast<TObjective *>(this))->GainAdjustmentHessianBoosting();
      static_assert(std::is_same<decltype(gainAdjustmentHessianBoosting), const double>::value, 
         "this->GainAdjustmentHessianBoosting() should return a double");
      pObjectiveWrapperOut->m_gainAdjustmentHessianBoosting = gainAdjustmentHessianBoosting;

      const auto gradientConstant = (static_cast<TObjective *>(this))->GradientConstant();
      static_assert(std::is_same<decltype(gradientConstant), const double>::value, "this->GradientConstant() should return a double");
      pObjectiveWrapperOut->m_gradientConstant = gradientConstant;
      
      const auto hessianConstant = (static_cast<TObjective *>(this))->HessianConstant();
      static_assert(std::is_same<decltype(hessianConstant), const double>::value, "this->HessianConstant() should return a double");
      pObjectiveWrapperOut->m_hessianConstant = hessianConstant;

      pObjectiveWrapperOut->m_bObjectiveHasHessian = HasHessian<TObjective, TFloat>() ? EBM_TRUE : EBM_FALSE;
      pObjectiveWrapperOut->m_bRmse = TObjective::k_bRmse ? EBM_TRUE : EBM_FALSE;

      pObjectiveWrapperOut->m_pObjective = this;

      SetFinishMetric<TObjective, TFloat>(pFunctionPointers);
   }

   Objective() = default;
   ~Objective() = default;

public:

   static ErrorEbm CreateObjective(
      const REGISTER_OBJECTIVES_FUNCTION registerObjectivesFunction,
      const Config * const pConfig,
      const char * const sObjective,
      const char * const sObjectiveEnd,
      ObjectiveWrapper * const pObjectiveWrapperOut
   ) noexcept;
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


#define OBJECTIVE_CONSTANTS_BOILERPLATE(__EBM_TYPE, __MAXIMIZE_METRIC, __LINK_FUNCTION) \
   public: \
      static constexpr bool k_bRmse = false; \
      static constexpr BoolEbm k_bMaximizeMetric = (__MAXIMIZE_METRIC); \
      static constexpr LinkEbm k_linkFunction = (__LINK_FUNCTION); \
      static ErrorEbm StaticApplyUpdate(const Objective * const pThis, ApplyUpdateBridge * const pData) { \
         return (static_cast<const __EBM_TYPE<TFloat> *>(pThis))->ParentApplyUpdate<const __EBM_TYPE<TFloat>, TFloat>(pData); \
      } \
      template<typename T = void, typename std::enable_if<TFloat::bCpu, T>::type * = nullptr> \
      static double StaticFinishMetric(const Objective * const pThis, const double metricSum) { \
         return (static_cast<const __EBM_TYPE<TFloat> *>(pThis))->FinishMetric(metricSum); \
      } \
      void FillWrapper(void * const pWrapperOut) noexcept { \
         static_assert( \
            std::is_same<__EBM_TYPE<TFloat>, typename std::remove_pointer<decltype(this)>::type>::value, \
            "*Objective types mismatch"); \
         FillObjectiveWrapper<typename std::remove_pointer<decltype(this)>::type, TFloat>(pWrapperOut); \
      }

#define OBJECTIVE_TEMPLATE_BOILERPLATE \
   public: \
      template<size_t cCompilerScores, ptrdiff_t cCompilerPack, bool bHessian, bool bKeepGradHess, bool bCalcMetric, bool bWeight> \
      GPU_DEVICE void InjectedApplyUpdate(ApplyUpdateBridge * const pData) const { \
         Objective::ChildApplyUpdate<typename std::remove_pointer<decltype(this)>::type, TFloat, \
            cCompilerScores, cCompilerPack, bHessian, bKeepGradHess, bCalcMetric, bWeight>(pData); \
      }

#define OBJECTIVE_BOILERPLATE(__EBM_TYPE, __MAXIMIZE_METRIC, __LINK_FUNCTION) \
   OBJECTIVE_CONSTANTS_BOILERPLATE(__EBM_TYPE, __MAXIMIZE_METRIC, __LINK_FUNCTION) \
   OBJECTIVE_TEMPLATE_BOILERPLATE

} // DEFINED_ZONE_NAME

#endif // OBJECTIVE_HPP
