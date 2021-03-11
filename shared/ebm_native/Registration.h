// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef REGISTRATION_H
#define REGISTRATION_H

#include <stddef.h> // size_t, ptrdiff_t
#include <vector>
#include <functional>
#include <memory>

#include "EbmInternal.h" // INLINE_ALWAYS
#include "Logging.h" // EBM_ASSERT & LOG
#include "Config.h"
#include "Registrable.h"

class SkipRegistrationException final : public std::exception {
   // we don't derrive from EbmException since this exception isn't meant to percolate up past the C interface
public:
   SkipRegistrationException() = default;
};

class ParameterValueOutOfRangeException final : public std::exception {
public:
   ParameterValueOutOfRangeException() = default;
};

class ParameterMismatchWithConfigException final : public std::exception {
public:
   ParameterMismatchWithConfigException() = default;
};

class ParameterValueMalformedException final : public std::exception {
   // this should not be thrown from the Registrable constructor
public:
   ParameterValueMalformedException() = default;
};

class ParameterUnknownException final : public std::exception {
   // this should not be thrown from the Registrable constructor
public:
   ParameterUnknownException() = default;
};

class RegistrationUnknownException final : public std::exception {
   // this should not be thrown from the Registrable constructor
public:
   RegistrationUnknownException() = default;
};

class RegistrationConstructorException final : public std::exception {
   // this should not be thrown from the Registrable constructor
public:
   RegistrationConstructorException() = default;
};

class IllegalParamNameException final : public std::exception {
   // this should not be thrown from the Registrable constructor
public:
   IllegalParamNameException() = default;
};

class IllegalRegistrationNameException final : public std::exception {
   // this should not be thrown from the Registrable constructor
public:
   IllegalRegistrationNameException() = default;
};

class DuplicateParamNameException final : public std::exception {
   // this should not be thrown from the Registrable constructor
public:
   DuplicateParamNameException() = default;
};

class ParamBase {
   const char * const m_sParamName;

protected:

   ParamBase(const char * const sParamName);

public:

   INLINE_ALWAYS const char * GetParamName() const noexcept {
      return m_sParamName;
   }
};

class FloatParam final : public ParamBase {
   const double m_defaultValue;

public:

   typedef double ParamType;

   INLINE_ALWAYS double GetDefaultValue() const noexcept {
      return m_defaultValue;
   }

   INLINE_ALWAYS FloatParam(const char * const sParamName, const double defaultValue) :
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
   const char * const m_sRegistrationName;

   // these ConvertStringToRegistrationType functions are here to de-template the various Param types

   static INLINE_ALWAYS const char * ConvertStringToRegistrationType(
      const char * const s, 
      double * const pResultOut
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

   static void CheckParamNames(const char * const sParamName, std::vector<const char *> usedParamNames);

   template<typename TParam>
   static typename TParam::ParamType UnpackParam(
      const TParam & param,
      const char * sRegistration,
      const char * const sRegistrationEnd,
      size_t & cUsedParamsInOut
   ) {
      static_assert(std::is_base_of<ParamBase, TParam>::value, "TParam not derrived from ParamBase");

      typename TParam::ParamType paramValue = param.GetDefaultValue();
      while(true) {
         // check and handle a possible parameter
         const char * sNext = IsStringEqualsCaseInsensitive(sRegistration, param.GetParamName());
         if(nullptr != sNext) {
            if(k_valueSeparator == *sNext) {
               ++cUsedParamsInOut;

               // before this point we could have been seeing a longer version of our proposed tag
               // eg: the given tag was "something_else=" but our tag was "something="
               sRegistration = sNext + 1;
               sRegistration = ConvertStringToRegistrationType(sRegistration, &paramValue);
               if(nullptr == sRegistration) {
                  throw ParameterValueMalformedException();
               }
               if(sRegistrationEnd == sRegistration) {
                  break;
               }
               EBM_ASSERT(k_registrationSeparator != *sRegistration);
               EBM_ASSERT(k_valueSeparator != *sRegistration);
               EBM_ASSERT(k_typeTerminator != *sRegistration);

               if(k_paramSeparator != *sRegistration) {
                  throw ParameterValueMalformedException();
               }
               ++sRegistration;
               continue;
            }
         }
         sRegistration = strchr(sRegistration, k_paramSeparator);
         if(nullptr == sRegistration || sRegistrationEnd <= sRegistration) {
            break;
         }
         ++sRegistration;
      }
      return paramValue;
   }

   static void FinalCheckParameters(
      const char * sRegistration, 
      const char * const sRegistrationEnd,
      const size_t cUsedParams
   );
   const char * CheckRegistrationName(const char * sRegistration, const char * const sRegistrationEnd) const;

   virtual std::unique_ptr<const Registrable> AttemptCreate(
      const Config & config,
      const char * const sRegistration,
      const char * const sRegistrationEnd
   ) const = 0;

   Registration(const char * const sRegistrationName);

public:

   constexpr static const char k_registrationSeparator = ',';
   constexpr static const char k_paramSeparator = ';';
   constexpr static char k_valueSeparator = '=';
   constexpr static char k_typeTerminator = ':';

   static std::vector<std::unique_ptr<const Registrable>> CreateRegistrables(
      const Config & config,
      const char * sRegistration,
      const std::vector<std::shared_ptr<const Registration>> & registrations
   );

   virtual ~Registration() = default;
};

template<template <typename> class TRegistrable, typename TFloat, typename... Args>
class RegistrationPack final : public Registration {

   // this lambda function holds our templated parameter pack until we need it
   std::function<std::unique_ptr<const Registrable>(
      const Config & config, 
      const char * const sRegistration,
      const char * const sRegistrationEnd
   )> m_callBack;

   INLINE_ALWAYS static void UnpackRecursive(std::vector<const char *> & paramNames) {
      UNUSED(paramNames);
      return;
   }

   template<typename TParam, typename... ArgsConverted>
   INLINE_ALWAYS static void UnpackRecursive(std::vector<const char *> & paramNames, const TParam param, const ArgsConverted...args) {
      static_assert(std::is_base_of<ParamBase, TParam>::value, "RegistrationPack::UnpackRecursive TParam must derive from ParamBase");
      CheckParamNames(param.GetParamName(), paramNames);
      UnpackRecursive(paramNames, args...);
   }

   template<typename... ArgsConverted>
   static std::unique_ptr<const Registrable> CheckAndCallNew(
      const Config & config,
      const char * const sRegistration,
      const char * const sRegistrationEnd,
      const size_t & cUsedParams,
      const ArgsConverted...args
   ) {
      // The registration string has been processed so we now have either the default param values or we have the parameter
      // values specified in the registration string.  Now we need to verify that there weren't any unused parameters,
      // which would have been an error.  FinalCheckParameters does this and throws an exception if it finds any errors
      FinalCheckParameters(sRegistration, sRegistrationEnd, cUsedParams);
      try {
         // unique_ptr constructor is noexcept, so it should be safe to call it inside the try/catch block
         return std::unique_ptr<const Registrable>(new TRegistrable<TFloat>(config, args...));
      } catch(const SkipRegistrationException &) {
         return nullptr;
      } catch(const ParameterValueOutOfRangeException &) {
         throw;
      } catch(const ParameterMismatchWithConfigException &) {
         throw;
      } catch(const EbmException &) {
         // generally we'd prefer that the Registration constructors avoid this exception, but pass it along if thrown
         throw;
      } catch(const std::bad_alloc &) {
         throw;
      } catch(...) {
         // our client Registration functions should only ever throw SkipRegistrationException, a derivative of EbmException, 
         // or std::bad_alloc, but check anyways
         throw RegistrationConstructorException();
      }
   }

   std::unique_ptr<const Registrable> AttemptCreate(
      const Config & config, 
      const char * sRegistration,
      const char * const sRegistrationEnd
   ) const override {
      sRegistration = CheckRegistrationName(sRegistration, sRegistrationEnd);
      if(nullptr == sRegistration) {
         // we are not the specified registration function
         return nullptr;
      }

      // m_callBack contains the parameter pack that our constructor was created with, so we're regaining access here
      return m_callBack(config, sRegistration, sRegistrationEnd);
   }

public:

   RegistrationPack(const char * sRegistrationName, const Args...args) : Registration(sRegistrationName) {

      std::vector<const char *> usedParamNames;
      UnpackRecursive(usedParamNames, args...);

      // hide our parameter pack in a lambda so that we don't have to think about it yet.  Seems easier than using a tuple
      m_callBack = [args...](
         const Config & config,
         const char * const sRegistration,
         const char * const sRegistrationEnd
      ) {
         // The usage of cUsedParams is a bit unusual.  It starts off at zero, but gets incremented in the calls to
         // UnpackParam.  When CheckAndCallNew is called, the value in cUsedParams is the total of all parameters
         // that were "claimed" by calls to UnpackParam.  Inside CheckAndCallNew we check that the total number
         // of valid parameters equals the number of parameters that were processed.  This is just to get arround
         // the issue that template parameter packs are hard to deal with in C++11 at least.
         size_t cUsedParams = 0;

         // UnpackParam processes each Param type independently, but we keep a count of all the valid parameters 
         // that were processed.  C++ gives us no guarantees about which order
         // the UnpackParam functions are called, but we are guaranteed that they are all called before 
         // CheckAndCallNew is called, so inside there we verify whether all the parameters were used
         return CheckAndCallNew(
            config,
            sRegistration,
            sRegistrationEnd,
            cUsedParams,
            UnpackParam(args, sRegistration, sRegistrationEnd, INOUT cUsedParams)...);
      };
   }
};

template<template <typename> class TRegistrable, typename TFloat, typename... Args>
std::shared_ptr<const Registration> Register(const char * const sRegistrationName, const Args...args) {
   // ideally we'd be returning unique_ptr here, but we pass this to an initialization list which doesn't work in C++11
   return std::make_shared<const RegistrationPack<TRegistrable, TFloat, Args...>>(sRegistrationName, args...);
}

#endif // REGISTRATION_H
