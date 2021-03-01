// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// !!! NOTE: To add a new loss/objective function in C++, follow the steps listed at the top of the "Loss.cpp" file !!!

#ifndef REGISTER_LOSS_H
#define REGISTER_LOSS_H

#include <stddef.h> // size_t, ptrdiff_t
#include <vector>
#include <functional>
#include <memory>
#include <algorithm>

#include "EbmInternal.h" // INLINE_ALWAYS
#include "Logging.h" // EBM_ASSERT & LOG
#include "FeatureGroup.h"
#include "ThreadStateBoosting.h"
#include "Loss.h"

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

   static void FinalCheckParameters(const char * sLoss, std::vector<const char *> & usedLocations);
   const char * CheckLossName(const char * sLoss) const;

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

#endif // REGISTER_LOSS_H