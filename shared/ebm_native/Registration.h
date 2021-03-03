// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// !!! NOTE: To add a new loss/objective function in C++, follow the steps listed at the top of the "Loss.cpp" file !!!

#ifndef REGISTRATION_H
#define REGISTRATION_H

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

class SkipRegistrationException final : public std::exception {
   // we don't derrive from EbmException since this exception isn't meant to percolate up past the C interface
public:
   SkipRegistrationException() = default;
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

class ParamBase {
   const char * const m_sParamName;

protected:

   INLINE_ALWAYS ParamBase(const char * const sParamName) : m_sParamName(sParamName) {
   }

public:

   INLINE_ALWAYS const char * GetParamName() const noexcept {
      return m_sParamName;
   }
};

class FloatParam final : public ParamBase {
   const FloatEbmType m_defaultValue;

public:

   typedef FloatEbmType ParamType;

   INLINE_ALWAYS FloatEbmType GetDefaultValue() const noexcept {
      return m_defaultValue;
   }

   INLINE_ALWAYS FloatParam(const char * const sParamName, const FloatEbmType defaultValue) :
      ParamBase(sParamName),
      m_defaultValue(defaultValue) {
   }
};

class BoolParam final : public ParamBase {
   const bool m_defaultValue;

public:

   typedef bool ParamType;

   INLINE_ALWAYS bool GetDefaultValue() const noexcept {
      return m_defaultValue;
   }

   INLINE_ALWAYS BoolParam(const char * const sParamName, const bool defaultValue) :
      ParamBase(sParamName),
      m_defaultValue(defaultValue) {
   }
};

class Registration {
   constexpr static const char k_paramSeparator = ';';
   constexpr static char k_valueSeparator = '=';
   constexpr static char k_typeTerminator = ':';

   const char * const m_sRegistrationName;

   // these ConvertStringToRegistrationType functions are here to de-template the various Param types

   static INLINE_ALWAYS const char * ConvertStringToRegistrationType(
      const char * const s, 
      FloatEbmType * const pResultOut
   ) noexcept {
      return ConvertStringToFloat(s, pResultOut);
   }

   static INLINE_ALWAYS const char * ConvertStringToRegistrationType(
      const char * const s, 
      bool * const pResultOut
   ) noexcept {
      // TODO : implement
      UNUSED(s);
      UNUSED(pResultOut);
      return nullptr;
   }

protected:

   template<typename TParam>
   static typename TParam::ParamType UnpackParam(
      const TParam & param,
      const char * sRegistration,
      std::vector<const char *> & usedLocations) 
   {
      typename TParam::ParamType paramValue = param.GetDefaultValue();
      while(true) {
         // check and handle a possible parameter
         const char * sNext = IsStringEqualsCaseInsensitive(sRegistration, param.GetParamName());
         if(nullptr != sNext) {
            if(k_valueSeparator == *sNext) {
               usedLocations.push_back(sRegistration);

               // before this point we could have been seeing a longer version of our proposed tag
               // eg: the given tag was "something_else=" but our tag was "something="
               sRegistration = sNext + 1;
               sRegistration = ConvertStringToRegistrationType(sRegistration, &paramValue);
               if(nullptr == sRegistration) {
                  throw EbmException(Error_LossParameterValueMalformed);
               }
               if(0 == *sRegistration) {
                  break;
               }
               if(k_paramSeparator != *sRegistration) {
                  throw EbmException(Error_LossParameterValueMalformed);
               }
               ++sRegistration;
               continue;
            }
         }
         sRegistration = strchr(sRegistration, k_paramSeparator);
         if(nullptr == sRegistration) {
            break;
         }
         ++sRegistration;
      }
      return paramValue;
   }

   static void FinalCheckParameters(const char * sRegistration, std::vector<const char *> & usedLocations);
   const char * CheckRegistrationName(const char * sRegistration) const;

   INLINE_ALWAYS Registration(const char * sRegistrationName) : m_sRegistrationName(sRegistrationName) {
   }

public:

   virtual std::unique_ptr<const Registrable> AttemptCreate(const Config & config, const char * const sRegistration) const = 0;
   virtual ~Registration() = default;
};

template<typename TRegistrable, typename... Args>
class RegistrationPack final : public Registration {

   // this lambda function holds our templated parameter pack until we need it
   std::function<std::unique_ptr<const Registrable>(const Config & config, const char * const sRegistration)> m_callBack;

   template<typename... ArgsConverted>
   static std::unique_ptr<const Registrable> CheckAndCallNew(
      const Config & config,
      const char * const sRegistration,
      std::vector<const char *> & usedLocations,
      ArgsConverted...args
   ) {
      // The registration string has been processed so we now have either the default param values or we have the parameter
      // values specified in the registration string.  Now we need to verify that there weren't any unused parameters,
      // which would have been an error.  FinalCheckParameters does this and throws an exception if it finds any errors
      FinalCheckParameters(sRegistration, usedLocations);
      try {
         // unique_ptr constructor is noexcept, so it should be safe to call it inside the try/catch block
         return std::unique_ptr<const Registrable>(new TRegistrable(config, args...));
      } catch(const SkipRegistrationException &) {
         throw;
      } catch(const EbmException &) {
         throw;
      } catch(const std::bad_alloc &) {
         throw;
      } catch(...) {
         // our client Registration functions should only ever throw SkipRegistrationException, a derivative of EbmException, 
         // or std::bad_alloc, but check anyways
         throw EbmException(Error_LossConstructorException);
      }
   }

public:

   RegistrationPack(const char * sRegistrationName, Args...args) : Registration(sRegistrationName) {

      // hide our parameter pack in a lambda so that we don't have to think about it yet.  Seems easier than using a tuple
      m_callBack = [args...](
         const Config & config,
         const char * const sRegistration
      ) {
         std::vector<const char *> usedLocations;
         // UnpackParam processes each Param type independently, but we keep a list of all the points
         // in the string that were processed with usedLocations.  C++ gives us no guarantees about which order
         // the UnpackParam functions are called, but we are guaranteed that they are all called before 
         // CheckAndCallNew is called, so inside there we verify whether all the parameters were used
         return std::unique_ptr<const Registrable>(CheckAndCallNew(
            config,
            sRegistration,
            usedLocations,
            UnpackParam(args, sRegistration, usedLocations)...)
         );
      };
   }

   std::unique_ptr<const Registrable> AttemptCreate(const Config & config, const char * sRegistration) const override {
      sRegistration = CheckRegistrationName(sRegistration);
      if(nullptr == sRegistration) {
         // we are not the specified registration function
         return nullptr;
      }

      // m_callBack contains the parameter pack that our constructor was created with, so we're regaining access here
      return m_callBack(config, sRegistration);
   }
};

template<typename TRegistrable, typename... Args>
std::shared_ptr<const Registration> Register(const char * sRegistrationName, Args...args) {
   // ideally we'd be returning unique_ptr here, but we pass this to an initialization list which doesn't work in C++11
   return std::make_shared<const RegistrationPack<TRegistrable, Args...>>(sRegistrationName, args...);
}

#endif // REGISTRATION_H
