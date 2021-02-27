// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// !!! NOTE: To add a new loss/objective function in C++, follow the steps listed at the top of the "Loss.cpp" file !!!

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

   Loss() = default;

public:

   // TODO: we should wrap all our calls to the potentially re-written client code with try/catch blocks since we don't know if it throws exceptions

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
};

class LossParameterValueOutOfRangeException final : public EbmException {
public:
   INLINE_ALWAYS LossParameterValueOutOfRangeException() : EbmException(Error_LossParameterValueOutOfRange) {
   }
};

class LossParameterMismatchWithConfigException final : public EbmException {
public:
   INLINE_ALWAYS LossParameterMismatchWithConfigException() : EbmException(Error_LossParameterMismatchWithConfig) {
   }
};

class LossParam {
   const char * const m_sParamName;

protected:

   INLINE_ALWAYS LossParam(const char * const sParamName) : m_sParamName(sParamName) {
   }

public:

   INLINE_ALWAYS const char * GetParamName() const noexcept {
      return m_sParamName;
   }
};


class FloatLossParam final : public LossParam {
   const FloatEbmType m_defaultValue;

public:

   typedef FloatEbmType LossParamType;

   INLINE_ALWAYS FloatEbmType GetDefaultValue() const noexcept {
      return m_defaultValue;
   }

   INLINE_ALWAYS FloatLossParam(const char * const sParamName, const FloatEbmType defaultValue) :
      LossParam(sParamName),
      m_defaultValue(defaultValue) {
   }
};

class BoolLossParam final : public LossParam {
   const bool m_defaultValue;

public:

   typedef bool LossParamType;

   INLINE_ALWAYS bool GetDefaultValue() const noexcept {
      return m_defaultValue;
   }

   INLINE_ALWAYS BoolLossParam(const char * const sParamName, const bool defaultValue) :
      LossParam(sParamName),
      m_defaultValue(defaultValue) {
   }
};

class RegisterLossBase {
   const char * const m_sLossName;

   static INLINE_ALWAYS const char * ConvertStringToLossType(const char * const s, FloatEbmType * const pResultOut) noexcept {
      return ConvertStringToFloat(s, pResultOut);
   }

   static INLINE_ALWAYS const char * ConvertStringToLossType(const char * const s, bool * const pResultOut) noexcept {
      // TODO : implement
      UNUSED(s);
      UNUSED(pResultOut);
      return nullptr;
   }

protected:

   template<typename TLossParam>
   static typename TLossParam::LossParamType UnpackLossParam(
      const TLossParam & param,
      const char * sLoss,
      std::vector<const char *> & usedLocations) 
   {
      typename TLossParam::LossParamType paramValue = param.GetDefaultValue();
      while(true) {
         // check and handle a possible parameter
         const char * sNext = IsStringEqualsCaseInsensitive(sLoss, param.GetParamName());
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

   static INLINE_ALWAYS void FinalCheckParameters(const char * sLoss, std::vector<const char *> & usedLocations) {
      std::sort(usedLocations.begin(), usedLocations.end());

      for(const char * sParam : usedLocations) {
         if(sParam != sLoss) {
            throw EbmException(Error_LossParameterUnknown);
         }
         sLoss = strchr(sLoss, ',');
         if(nullptr == sLoss) {
            return;
         }
         ++sLoss;
         if(0 == *SkipWhitespace(sLoss)) {
            return;
         }
      }
      if(0 != *SkipWhitespace(sLoss)) {
         throw EbmException(Error_LossParameterUnknown);
      }
   }

   INLINE_ALWAYS const char * CheckLossName(const char * sLoss) const {
      EBM_ASSERT(nullptr != sLoss);

      sLoss = IsStringEqualsCaseInsensitive(sLoss, m_sLossName);
      if(nullptr == sLoss) {
         // we are not the specified loss function
         return nullptr;
      }
      if(0 != *sLoss) {
         if(':' != *sLoss) {
            // we are not the specified objective, but the objective could still be something with a longer string
            // eg: the given tag was "something_else:" but our tag was "something:", so we matched on "something" only
            return nullptr;
         }
         sLoss = SkipWhitespace(sLoss + 1);
      }
      return sLoss;
   }

   INLINE_ALWAYS RegisterLossBase(const char * sLossName) : m_sLossName(sLossName) {
   }

public:

   virtual std::unique_ptr<const Loss> AttemptCreateLoss(const Config & config, const char * const sLoss) const = 0;
   virtual ~RegisterLossBase() = default;
};

template<typename TLoss, typename... Args>
class RegisterLossPack final : public RegisterLossBase {

   std::function<std::unique_ptr<const Loss>(const Config & config, const char * const sLoss)> m_callBack;

   template<typename... ArgsConverted>
   static std::unique_ptr<const Loss> CheckAndCallNew(
      const Config & config,
      const char * const sLoss,
      std::vector<const char *> & usedLocations,
      ArgsConverted...args
   ) {
      FinalCheckParameters(sLoss, usedLocations); // this throws if it finds anything wrong
      return std::unique_ptr<const Loss>(new TLoss(config, args...));
   }

public:

   RegisterLossPack(const char * sLossName, Args...args) : RegisterLossBase(sLossName) {
      m_callBack = [args...](
         const Config & config,
         const char * const sLoss
      ) {
         std::vector<const char *> usedLocations;
         return std::unique_ptr<const Loss>(CheckAndCallNew(
            config,
            sLoss,
            usedLocations,
            UnpackLossParam(args, sLoss, usedLocations)...)
         );
      };
   }

   std::unique_ptr<const Loss> AttemptCreateLoss(const Config & config, const char * sLoss) const override {
      sLoss = CheckLossName(sLoss);
      if(nullptr == sLoss) {
         // we are not the specified loss function
         return std::unique_ptr<const Loss>();
      }
      return m_callBack(config, sLoss);
   }
};

template<typename TLoss, typename... Args>
std::shared_ptr<const RegisterLossBase> RegisterLoss(const char * sLossName, Args...args) noexcept {
   // ideally we'd be returning unique_ptr here, but we pass this to an initialization list which doesn't work in C++11
   return std::make_shared<const RegisterLossPack<TLoss, Args...>>(sLossName, args...);
}


// TODO: add parameters to these macros to pass in what options we want in our template generation
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

#endif // LOSS_H
