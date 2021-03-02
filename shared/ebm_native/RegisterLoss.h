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

   // these ConvertStringToLossType functions are here to de-template the various LossParam types

   static INLINE_ALWAYS const char * ConvertStringToLossType(
      const char * const s, 
      FloatEbmType * const pResultOut
   ) noexcept {
      return ConvertStringToFloat(s, pResultOut);
   }

   static INLINE_ALWAYS const char * ConvertStringToLossType(
      const char * const s, 
      bool * const pResultOut
   ) noexcept {
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

   // this lambda function holds our templated parameter pack until we need it
   std::function<std::unique_ptr<const Loss>(const Config & config, const char * const sLoss)> m_callBack;

   template<typename... ArgsConverted>
   static std::unique_ptr<const Loss> CheckAndCallNew(
      const Config & config,
      const char * const sLoss,
      std::vector<const char *> & usedLocations,
      ArgsConverted...args
   ) {
      // The loss string has been processed so we now have either the default param values or we have the parameter
      // values specified in the loss string.  Now we need to verify that there weren't any unused parameters,
      // which would have been an error.  FinalCheckParameters does this and throws an exception if it finds any errors
      FinalCheckParameters(sLoss, usedLocations);
      return std::unique_ptr<const Loss>(new TLoss(config, args...));
   }

public:

   RegisterLossPack(const char * sLossName, Args...args) : RegisterLossBase(sLossName) {

      // hide our parameter pack in a lambda so that we don't have to think about it yet.  Seems easier than using a tuple
      m_callBack = [args...](
         const Config & config,
         const char * const sLoss
      ) {
         std::vector<const char *> usedLocations;
         // UnpackLossParam processes each LossParam type independently, but we keep a list of all the points
         // in the string that were processed with usedLocations.  C++ gives us no guarantees about which order
         // the UnpackLossParam functions are called, but we are guaranteed that they are all called before 
         // CheckAndCallNew is called, so inside there we verify whether all the parameters were used
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
         return nullptr;
      }

      // m_callBack contains the parameter pack that our constructor was created with, so we're regaining access here
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