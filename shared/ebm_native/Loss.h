// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef LOSS_H
#define LOSS_H

#include <stddef.h> // size_t, ptrdiff_t

#include "EbmInternal.h" // INLINE_ALWAYS
#include "Logging.h" // EBM_ASSERT & LOG
#include "FeatureGroup.h"
#include "ThreadStateBoosting.h"

class Loss {

   template<typename TLoss, ptrdiff_t compilerCountClasses>
   struct CountClasses final {
      INLINE_ALWAYS static ErrorEbmType ApplyTraining(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) {
         if(compilerCountClasses == pThreadStateBoosting->GetBooster()->GetRuntimeLearningTypeOrCountTargetClasses()) {
            return BitPack<TLoss, compilerCountClasses, k_cItemsPerBitPackMax2>::ApplyTraining(pLoss, pThreadStateBoosting, pFeatureGroup);
         } else {
            return CountClasses<TLoss, compilerCountClasses + 1>::ApplyTraining(pLoss, pThreadStateBoosting, pFeatureGroup);
         }
      }
      INLINE_ALWAYS static ErrorEbmType ApplyValidation(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) {
         if(compilerCountClasses == pThreadStateBoosting->GetBooster()->GetRuntimeLearningTypeOrCountTargetClasses()) {
            return BitPack<TLoss, compilerCountClasses, k_cItemsPerBitPackMax2>::ApplyValidation(pLoss, pThreadStateBoosting, pFeatureGroup, pMetricOut);
         } else {
            return CountClasses<TLoss, compilerCountClasses + 1>::ApplyValidation(pLoss, pThreadStateBoosting, pFeatureGroup, pMetricOut);
         }
      }
   };
   template<typename TLoss>
   struct CountClasses<TLoss, k_cCompilerOptimizedTargetClassesMax + 1> final {
      INLINE_ALWAYS static ErrorEbmType ApplyTraining(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) {
         return BitPack<TLoss, k_dynamicClassification, k_cItemsPerBitPackMax2>::ApplyTraining(pLoss, pThreadStateBoosting, pFeatureGroup);
      }
      INLINE_ALWAYS static ErrorEbmType ApplyValidation(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) {
         return BitPack<TLoss, k_dynamicClassification, k_cItemsPerBitPackMax2>::ApplyValidation(pLoss, pThreadStateBoosting, pFeatureGroup, pMetricOut);
      }
   };


   template<typename TLoss, ptrdiff_t compilerCountClasses, ptrdiff_t compilerBitPack>
   struct BitPack final {
      INLINE_ALWAYS static ErrorEbmType ApplyTraining(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) {
         if(compilerBitPack == pFeatureGroup->GetBitPack()) {
            return pLoss->BitPackApplyTraining<TLoss, compilerCountClasses, compilerBitPack>(pThreadStateBoosting, pFeatureGroup);
         } else {
            return BitPack<TLoss, compilerCountClasses, GetNextBitPack(compilerBitPack)>::ApplyTraining(pLoss, pThreadStateBoosting, pFeatureGroup);
         }
      }
      INLINE_ALWAYS static ErrorEbmType ApplyValidation(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) {
         if(compilerBitPack == pFeatureGroup->GetBitPack()) {
            return pLoss->BitPackApplyValidation<TLoss, compilerCountClasses, compilerBitPack>(pThreadStateBoosting, pFeatureGroup, pMetricOut);
         } else {
            return BitPack<TLoss, compilerCountClasses, GetNextBitPack(compilerBitPack)>::ApplyValidation(pLoss, pThreadStateBoosting, pFeatureGroup, pMetricOut);
         }
      }
   };
   template<typename TLoss, ptrdiff_t compilerCountClasses>
   struct BitPack<TLoss, compilerCountClasses, k_cItemsPerBitPackNone> final {
      INLINE_ALWAYS static ErrorEbmType ApplyTraining(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) {
         return pLoss->BitPackApplyTraining<TLoss, compilerCountClasses, k_cItemsPerBitPackNone>(pThreadStateBoosting, pFeatureGroup);
      }
      INLINE_ALWAYS static ErrorEbmType ApplyValidation(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) {
         return pLoss->BitPackApplyValidation<TLoss, compilerCountClasses, k_cItemsPerBitPackNone>(pThreadStateBoosting, pFeatureGroup, pMetricOut);
      }
   };


   template<typename TLoss, ptrdiff_t compilerCountClasses, ptrdiff_t compilerBitPack>
   struct RemoveMulti final {
      INLINE_ALWAYS static ErrorEbmType ApplyTraining(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) {
         const TLoss * const pTLoss = static_cast<const TLoss *>(pLoss);
         return pTLoss->template ApplyTrainingMultiTemplated<compilerCountClasses, compilerBitPack>(pThreadStateBoosting, pFeatureGroup);
      }
      INLINE_ALWAYS static ErrorEbmType ApplyValidation(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) {
         const TLoss * const pTLoss = static_cast<const TLoss *>(pLoss);
         return pTLoss->template ApplyValidationMultiTemplated<compilerCountClasses, compilerBitPack>(pThreadStateBoosting, pFeatureGroup, pMetricOut);
      }
   };
   template<typename TLoss, ptrdiff_t compilerBitPack>
   struct RemoveMulti<TLoss, k_regression, compilerBitPack> final {
      INLINE_ALWAYS static ErrorEbmType ApplyTraining(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) {
         const TLoss * const pTLoss = static_cast<const TLoss *>(pLoss);
         return pTLoss->template ApplyTrainingTemplated<compilerBitPack>(pThreadStateBoosting, pFeatureGroup);
      }
      INLINE_ALWAYS static ErrorEbmType ApplyValidation(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) {
         const TLoss * const pTLoss = static_cast<const TLoss *>(pLoss);
         return pTLoss->template ApplyValidationTemplated<compilerBitPack>(pThreadStateBoosting, pFeatureGroup, pMetricOut);
      }
   };


   template<typename TLoss, ptrdiff_t compilerCountClasses, ptrdiff_t compilerBitPack>
   INLINE_RELEASE_TEMPLATED ErrorEbmType BitPackApplyTraining(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) const {
      return RemoveMulti<TLoss, compilerCountClasses, compilerBitPack>::ApplyTraining(this, pThreadStateBoosting, pFeatureGroup);
   }
   template<typename TLoss, ptrdiff_t compilerCountClasses, ptrdiff_t compilerBitPack>
   INLINE_RELEASE_TEMPLATED ErrorEbmType BitPackApplyValidation(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) const {
      return RemoveMulti<TLoss, compilerCountClasses, compilerBitPack>::ApplyValidation(this, pThreadStateBoosting, pFeatureGroup, pMetricOut);
   }


   template<typename TLoss, ptrdiff_t compilerBitPack>
   struct Shared final {
      INLINE_ALWAYS static ErrorEbmType ApplyTraining(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) {
         UNUSED(pLoss);
         UNUSED(pThreadStateBoosting);
         UNUSED(pFeatureGroup);
         return Error_None;
      }
      INLINE_ALWAYS static ErrorEbmType ApplyValidation(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) {
         UNUSED(pLoss);
         UNUSED(pThreadStateBoosting);
         UNUSED(pFeatureGroup);
         UNUSED(pMetricOut);
         return Error_None;
      }
   };
   template<typename TLoss>
   struct Shared<TLoss, k_cItemsPerBitPackNone> final {
      INLINE_ALWAYS static ErrorEbmType ApplyTraining(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) {
         UNUSED(pLoss);
         UNUSED(pThreadStateBoosting);
         UNUSED(pFeatureGroup);
         return Error_None;
      }
      INLINE_ALWAYS static ErrorEbmType ApplyValidation(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) {
         UNUSED(pLoss);
         UNUSED(pThreadStateBoosting);
         UNUSED(pFeatureGroup);
         UNUSED(pMetricOut);
         return Error_None;
      }
   };


   template<typename TLoss, ptrdiff_t compilerCountClasses, ptrdiff_t compilerBitPack>
   struct SharedMulti final {
      INLINE_ALWAYS static ErrorEbmType ApplyTraining(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) {
         UNUSED(pLoss);
         UNUSED(pThreadStateBoosting);
         UNUSED(pFeatureGroup);
         return Error_None;
      }
      INLINE_ALWAYS static ErrorEbmType ApplyValidation(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) {
         UNUSED(pLoss);
         UNUSED(pThreadStateBoosting);
         UNUSED(pFeatureGroup);
         UNUSED(pMetricOut);
         return Error_None;
      }
   };
   template<typename TLoss, ptrdiff_t compilerCountClasses>
   struct SharedMulti<TLoss, compilerCountClasses, k_cItemsPerBitPackNone> final {
      INLINE_ALWAYS static ErrorEbmType ApplyTraining(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) {
         UNUSED(pLoss);
         UNUSED(pThreadStateBoosting);
         UNUSED(pFeatureGroup);
         return Error_None;
      }
      INLINE_ALWAYS static ErrorEbmType ApplyValidation(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) {
         UNUSED(pLoss);
         UNUSED(pThreadStateBoosting);
         UNUSED(pFeatureGroup);
         UNUSED(pMetricOut);
         return Error_None;
      }
   };

protected:

   Loss() = default;

   template<typename TLoss, ptrdiff_t compilerBitPack>
   INLINE_RELEASE_TEMPLATED ErrorEbmType SharedApplyTraining(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) const {
      static_assert(std::is_base_of<Loss, TLoss>::value, "TLoss must inherit from Loss");
      return Shared<TLoss, compilerBitPack>::ApplyTraining(this, pThreadStateBoosting, pFeatureGroup);
   }
   template<typename TLoss, ptrdiff_t compilerBitPack>
   INLINE_RELEASE_TEMPLATED ErrorEbmType SharedApplyValidation(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) const {
      static_assert(std::is_base_of<Loss, TLoss>::value, "TLoss must inherit from Loss");
      return Shared<TLoss, compilerBitPack>::ApplyValidation(this, pThreadStateBoosting, pFeatureGroup, pMetricOut);
   }


   template<typename TLoss, ptrdiff_t compilerCountClasses, ptrdiff_t compilerBitPack>
   INLINE_RELEASE_TEMPLATED ErrorEbmType SharedApplyTrainingMulti(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) const {
      static_assert(std::is_base_of<Loss, TLoss>::value, "TLoss must inherit from Loss");
      return SharedMulti<TLoss, compilerCountClasses, compilerBitPack>::ApplyTraining(this, pThreadStateBoosting, pFeatureGroup);
   }

   template<typename TLoss, ptrdiff_t compilerCountClasses, ptrdiff_t compilerBitPack>
   INLINE_RELEASE_TEMPLATED ErrorEbmType SharedApplyValidationMulti(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) const {
      static_assert(std::is_base_of<Loss, TLoss>::value, "TLoss must inherit from Loss");
      return SharedMulti<TLoss, compilerCountClasses, compilerBitPack>::ApplyValidation(this, pThreadStateBoosting, pFeatureGroup, pMetricOut);
   }


   template<typename TLoss>
   INLINE_RELEASE_TEMPLATED ErrorEbmType LossApplyTraining(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) const {
      static_assert(std::is_base_of<Loss, TLoss>::value, "TLoss must inherit from Loss");
      return BitPack<TLoss, k_regression, k_cItemsPerBitPackMax2>::ApplyTraining(this, pThreadStateBoosting, pFeatureGroup);
   }
   template<typename TLoss>
   INLINE_RELEASE_TEMPLATED ErrorEbmType LossApplyValidation(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) const {
      static_assert(std::is_base_of<Loss, TLoss>::value, "TLoss must inherit from Loss");
      return BitPack<TLoss, k_regression, k_cItemsPerBitPackMax2>::ApplyValidation(this, pThreadStateBoosting, pFeatureGroup, pMetricOut);
   }


   template<typename TLoss>
   INLINE_RELEASE_TEMPLATED ErrorEbmType LossApplyTrainingMulti(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) const {
      static_assert(std::is_base_of<Loss, TLoss>::value, "TLoss must inherit from Loss");
      return CountClasses<TLoss, 3>::ApplyTraining(this, pThreadStateBoosting, pFeatureGroup);
   }
   template<typename TLoss>
   INLINE_RELEASE_TEMPLATED ErrorEbmType LossApplyValidationMulti(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) const {
      static_assert(std::is_base_of<Loss, TLoss>::value, "TLoss must inherit from Loss");
      return CountClasses<TLoss, 3>::ApplyValidation(this, pThreadStateBoosting, pFeatureGroup, pMetricOut);
   }

public:

   // TODO: we need to actually use this now..
   virtual FloatEbmType GetUpdateMultiple() const;

   virtual ErrorEbmType ApplyTraining(
      ThreadStateBoosting * const pThreadStateBoosting,
      const FeatureGroup * const pFeatureGroup
   ) const = 0;

   virtual ErrorEbmType ApplyValidation(
      ThreadStateBoosting * const pThreadStateBoosting,
      const FeatureGroup * const pFeatureGroup,
      FloatEbmType * const pMetricOut
   ) const = 0;

   static ErrorEbmType CreateLoss(
      const char * const sLoss,
      const Config * const pConfig,
      const Loss ** const ppLoss
   ) noexcept;

   virtual ~Loss() = default;
};

class SkipLossException final : public std::exception {
   // we don't derrive from EbmException since this exception isn't meant to percolate up past the C interface
public:
   SkipLossException() = default;

   const char * what() const noexcept override {
      return "Skip this loss function";
   }
};

class LossParameterValueOutOfRangeException final : public EbmException {

public:
   LossParameterValueOutOfRangeException() : EbmException(Error_LossParameterValueOutOfRange) {
   }

   const char * what() const noexcept override {
      return "Loss parameter value out of range";
   }
};

class LossParameterMismatchWithConfigException final : public EbmException {

public:
   LossParameterMismatchWithConfigException() : EbmException(Error_LossParameterMismatchWithConfig) {
   }

   const char * what() const noexcept override {
      return "Loss parameter mismatches the Config";
   }
};

class FloatLossParam final {
   const char * const m_sParamName;
   const FloatEbmType m_defaultValue;

public:

   typedef FloatEbmType LossParamType;

   INLINE_ALWAYS const char * GetParamName() const noexcept {
      return m_sParamName;
   }

   INLINE_ALWAYS FloatEbmType GetDefaultValue() const noexcept {
      return m_defaultValue;
   }

   INLINE_ALWAYS FloatLossParam(const char * const sParamName, const FloatEbmType defaultValue) :
      m_sParamName(sParamName),
      m_defaultValue(defaultValue) {
   }
};

class BoolLossParam final {
   const char * const m_sParamName;
   const bool m_defaultValue;

public:

   typedef bool LossParamType;

   INLINE_ALWAYS const char * GetParamName() const noexcept {
      return m_sParamName;
   }

   INLINE_ALWAYS bool GetDefaultValue() const noexcept {
      return m_defaultValue;
   }

   INLINE_ALWAYS BoolLossParam(const char * const sParamName, const bool defaultValue) :
      m_sParamName(sParamName),
      m_defaultValue(defaultValue) {
   }
};

class LossRegistrationBase {
protected:

   LossRegistrationBase() = default;

public:
   virtual std::unique_ptr<const Loss> AttemptCreateLoss(const Config & config, const char * const sLoss) const = 0;
   virtual ~LossRegistrationBase() = default;
};

template<typename TLoss, typename... Args>
class LossRegistrationPack final : public LossRegistrationBase {
   const char * m_sLossName;
   std::function<std::unique_ptr<const Loss>(const Config & config, const char * const sLoss)> m_callBack;

   static INLINE_ALWAYS const char * ConvertStringToLossType(const char * const s, FloatEbmType * const pResultOut) noexcept {
      return ConvertStringToFloat(s, pResultOut);
   }

   static INLINE_ALWAYS const char * ConvertStringToLossType(const char * const s, bool * const pResultOut) noexcept {
      // TODO : implement
      UNUSED(s);
      UNUSED(pResultOut);
      return nullptr;
   }

   template<typename TLossParam>
   static typename TLossParam::LossParamType UnpackLossParam(const TLossParam & param, const char * sLoss, std::vector<const char *> & usedLocations) {
      typename TLossParam::LossParamType paramValue = param.GetDefaultValue();
      while(true) {
         const char * sNext;

         // check and handle a possible parameter
         static const char k_sDeltaTag[] = "delta";
         sNext = IsStringEqualsCaseInsensitive(sLoss, k_sDeltaTag);
         if(nullptr != sNext) {
            if('=' == *sNext) {
               usedLocations.push_back(sLoss);

               // before this point we could have been seeing a longer version of our proposed tag
               // eg: the given tag was "something_else=" but our tag was "something="
               sLoss = sNext + 1;
               sLoss = ConvertStringToLossType(sLoss, &paramValue);
               if(nullptr == sLoss) {
                  throw EbmException(Error_LossParameterValueMalformed);
               }
               if(0 == *sLoss) {
                  break;
               }
               if(',' != *sLoss) {
                  throw EbmException(Error_LossParameterValueMalformed);
               }
               ++sLoss;
               continue;
            }
         }
         sLoss = strchr(sLoss, ',');
         if(nullptr == sLoss) {
            break;
         }
         ++sLoss;
      }
      return paramValue;
   }

public:
   LossRegistrationPack(const char * sLossName, Args...args) : m_sLossName(sLossName) {
      m_callBack = [args...](const Config & config, const char * const sLoss) {
         std::vector<const char *> usedLocations;

         std::unique_ptr<const Loss> pLoss = std::unique_ptr<const Loss>(new TLoss(config, UnpackLossParam(args, sLoss, usedLocations)...));

         std::sort(usedLocations.begin(), usedLocations.end());

         // TODO: check here if we used all our comma separated values
         UNUSED(sLoss); // remove this once we check for unused tags

         return pLoss;
      };
   }

   std::unique_ptr<const Loss> AttemptCreateLoss(const Config & config, const char * sLoss) const override {
      EBM_ASSERT(nullptr != sLoss);

      sLoss = IsStringEqualsCaseInsensitive(sLoss, m_sLossName);
      if(nullptr == sLoss) {
         // we are not the specified objective
         return nullptr;
      }
      if(0 != *sLoss) {
         if(':' != *sLoss) {
            // we are not the specified objective, but the objective could still be something with a longer string
            // eg: the given tag was "something_else:" but our tag was "something:", so we matched on "something" only
            return std::unique_ptr<const Loss>();
         }
         sLoss = SkipWhitespace(sLoss + 1);
      }
      return m_callBack(config, sLoss);
   }
};

template<typename TLoss, typename... Args>
std::shared_ptr<const LossRegistrationBase> LossRegistration(const char * sLossName, Args...args) noexcept {
   return std::make_shared<const LossRegistrationPack<TLoss, Args...>>(sLossName, args...);
}


#define LOSS_DEFAULT_MECHANICS_PUT_AT_END_OF_CLASS \
   public: \
      template<ptrdiff_t compilerBitPack> \
      ErrorEbmType ApplyTrainingTemplated(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) const { \
         return Loss::SharedApplyTraining<std::remove_pointer<decltype(this)>::type, compilerBitPack>(pThreadStateBoosting, pFeatureGroup); \
      } \
      template<ptrdiff_t compilerBitPack> \
      ErrorEbmType ApplyValidationTemplated(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) const { \
         return Loss::SharedApplyValidation<std::remove_pointer<decltype(this)>::type, compilerBitPack>(pThreadStateBoosting, pFeatureGroup, pMetricOut); \
      } \
      ErrorEbmType ApplyTraining(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) const override { \
         return Loss::LossApplyTraining<std::remove_pointer<decltype(this)>::type>(pThreadStateBoosting, pFeatureGroup); \
      } \
      ErrorEbmType ApplyValidation(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) const override { \
         return Loss::LossApplyValidation<std::remove_pointer<decltype(this)>::type>(pThreadStateBoosting, pFeatureGroup, pMetricOut); \
      }

#define LOSS_MULTI_DEFAULT_MECHANICS_PUT_AT_END_OF_CLASS \
   public: \
      template<ptrdiff_t compilerCountClasses, ptrdiff_t compilerBitPack> \
      ErrorEbmType ApplyTrainingMultiTemplated(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) const { \
         return Loss::SharedApplyTrainingMulti<std::remove_pointer<decltype(this)>::type, compilerCountClasses, compilerBitPack>(pThreadStateBoosting, pFeatureGroup); \
      } \
      template<ptrdiff_t compilerCountClasses, ptrdiff_t compilerBitPack> \
      ErrorEbmType ApplyValidationMultiTemplated(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) const { \
         return Loss::SharedApplyValidationMulti<std::remove_pointer<decltype(this)>::type, compilerCountClasses, compilerBitPack>(pThreadStateBoosting, pFeatureGroup, pMetricOut); \
      } \
      ErrorEbmType ApplyTraining(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) const override { \
         return Loss::LossApplyTrainingMulti<std::remove_pointer<decltype(this)>::type>(pThreadStateBoosting, pFeatureGroup); \
      } \
      ErrorEbmType ApplyValidation(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) const override { \
         return Loss::LossApplyValidationMulti<std::remove_pointer<decltype(this)>::type>(pThreadStateBoosting, pFeatureGroup, pMetricOut); \
      }




//class LossBinaryLogLoss final : public Loss {
//   // TODO: put this in it's own file
//public:
//
//   INLINE_ALWAYS LossBinaryLogLoss() {
//   }
//
//   template <typename T>
//   static INLINE_ALWAYS T CalculatePrediction(const T score) {
//      return score * 10000;
//   }
//
//   template <typename T>
//   static INLINE_ALWAYS T GetGradientFromBoolTarget(const bool target, const T prediction) {
//      return -100000;
//   }
//
//   template <typename T>
//   static INLINE_ALWAYS T GetHessianFromBoolTargetAndGradient(const bool target, const T gradient) {
//      // normally we'd get the hessian from the prediction, but for binary logistic regression it's a bit better
//      // to use the gradient, and the mathematics is a special case where we can do this.
//      return -100000;
//   }
//
//   INLINE_ALWAYS FloatEbmType GetUpdateMultiple() const noexcept override {
//      return FloatEbmType { 1 };
//   }
//
//};
//
//class LossRegressionMSE final : public Loss {
//   // TODO: put this in it's own file
//public:
//
//   INLINE_ALWAYS LossRegressionMSE() {
//   }
//
//   // for MSE regression, we get target - score at initialization and only store the gradients, so we never
//   // make a prediction, so we don't need CalculatePrediction(...)
//
//   template <typename T>
//   static INLINE_ALWAYS T GetGradientFromGradientPrev(const T target, const T gradientPrev) {
//      // for MSE regression, we get target - score at initialization and only store the gradients, so we
//      // never need the targets.  We just work from the previous gradients.
//
//      return -100000;
//   }
//
//   INLINE_ALWAYS FloatEbmType GetUpdateMultiple() const noexcept override {
//      return FloatEbmType { 1 };
//   }
//
//};

#endif // LOSS_H
