// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef REGISTRATION_HPP
#define REGISTRATION_HPP

#include <stddef.h> // size_t, ptrdiff_t
#include <vector>
#include <functional> // std::function
#include <memory>

#include "ebm_native.h"
#include "logging.h" // EBM_ASSERT
#include "common_c.h" // INLINE_ALWAYS
#include "bridge_c.h" // Config
#include "zones.h"

#include "registration_exceptions.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

class ParamBase {
   const char * const m_sParamName;

   void * operator new(std::size_t) = delete; // no virtual destructor so disallow pointer delete
   void operator delete (void *) = delete; // no virtual destructor so disallow pointer delete

protected:

   ParamBase(const char * const sParamName);

public:

   INLINE_ALWAYS const char * GetParamName() const noexcept {
      return m_sParamName;
   }
};

class FloatParam final : public ParamBase {
   const double m_defaultVal;

   void * operator new(std::size_t) = delete; // no virtual destructor so disallow pointer delete
   void operator delete (void *) = delete; // no virtual destructor so disallow pointer delete

public:

   typedef double ParamType;

   INLINE_ALWAYS double GetDefaultVal() const noexcept {
      return m_defaultVal;
   }

   INLINE_ALWAYS FloatParam(const char * const sParamName, const double defaultVal) :
      ParamBase(sParamName),
      m_defaultVal(defaultVal) {
   }
};

class BoolParam final : public ParamBase {
   const bool m_defaultVal;

   void * operator new(std::size_t) = delete; // no virtual destructor so disallow pointer delete
   void operator delete (void *) = delete; // no virtual destructor so disallow pointer delete

public:

   typedef bool ParamType;

   INLINE_ALWAYS bool GetDefaultVal() const noexcept {
      return m_defaultVal;
   }

   INLINE_ALWAYS BoolParam(const char * const sParamName, const bool defaultVal) :
      ParamBase(sParamName),
      m_defaultVal(defaultVal) {
   }
};

class Registration {
   const char * const m_sRegistrationName;

   // these ConvertStringToRegistrationType functions are here to de-template the various Param types

   INLINE_ALWAYS static const char * ConvertStringToRegistrationType(
      const char * const s, 
      double * const pResultOut
   ) noexcept {
      return ConvertStringToFloat(s, pResultOut);
   }

   INLINE_ALWAYS static const char * ConvertStringToRegistrationType(
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

      EBM_ASSERT(nullptr != sRegistration);
      EBM_ASSERT(nullptr != sRegistrationEnd);
      EBM_ASSERT(sRegistration <= sRegistrationEnd); // sRegistration contains the part after the tag now
      EBM_ASSERT(!(0x20 == *sRegistration || (0x9 <= *sRegistration && *sRegistration <= 0xd)));
      EBM_ASSERT(!(0x20 == *(sRegistrationEnd - 1) || (0x9 <= *(sRegistrationEnd - 1) && *(sRegistrationEnd - 1) <= 0xd)));
      EBM_ASSERT('\0' == *sRegistrationEnd || k_registrationSeparator == *sRegistrationEnd || 0x20 == *sRegistrationEnd || (0x9 <= *sRegistrationEnd && *sRegistrationEnd <= 0xd));

      typename TParam::ParamType paramVal = param.GetDefaultVal();
      while(true) {
         // check and handle a possible parameter
         const char * sNext = IsStringEqualsCaseInsensitive(sRegistration, param.GetParamName());
         if(nullptr != sNext) {
            if(k_valueSeparator == *sNext) {
               ++cUsedParamsInOut;

               // before this point we could have been seeing a longer version of our proposed tag
               // eg: the given tag was "something_else=" but our tag was "something="
               sRegistration = sNext + 1;
               sRegistration = ConvertStringToRegistrationType(sRegistration, &paramVal);
               if(nullptr == sRegistration) {
                  throw ParamValMalformedException();
               }
               if(sRegistrationEnd <= sRegistration) {
                  // if there are trailing spaces we can blow past the sRegistrationEnd which has spaces removed
                  break;
               }
               if(k_paramSeparator != *sRegistration) {
                  throw ParamValMalformedException();
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
      return paramVal;
   }

   static void FinalCheckParams(
      const char * sRegistration, 
      const char * const sRegistrationEnd,
      const size_t cUsedParams
   );
   const char * CheckRegistrationName(const char * sRegistration, const char * const sRegistrationEnd) const;

   virtual bool AttemptCreate(
      const Config * const pConfig,
      const char * sRegistration,
      const char * const sRegistrationEnd,
      void * const pWrapperOut
   ) const = 0;

   Registration(const char * const sRegistrationName);

public:

   static constexpr const char k_paramSeparator = ';';
   static constexpr char k_valueSeparator = '=';
   static constexpr char k_typeTerminator = ':';

   static bool CreateRegistrable(
      const Config * const pConfig,
      const char * sRegistration,
      const char * sRegistrationEnd,
      void * const pWrapperOut,
      const std::vector<std::shared_ptr<const Registration>> & registrations
   );

   virtual ~Registration() = default;
};

template<template <typename> class TRegistrable, typename TFloat, typename... Args>
class RegistrationPack final : public Registration {

   // this lambda function holds our templated parameter pack until we need it
   std::function<bool(
      const Config * const pConfig,
      const char * const sRegistration,
      const char * const sRegistrationEnd,
      void * const pWrapperOut
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
   static bool CheckAndCallNew(
      const Config * const pConfig,
      const char * const sRegistration,
      const char * const sRegistrationEnd,
      void * const pWrapperOut,
      const size_t & cUsedParams,
      const ArgsConverted...args
   ) {
      // The registration string has been processed so we now have either the default param values or we have the parameter
      // values specified in the registration string.  Now we need to verify that there weren't any unused parameters,
      // which would have been an error.  FinalCheckParams does this and throws an exception if it finds any errors
      FinalCheckParams(sRegistration, sRegistrationEnd, cUsedParams);

      // use malloc so that we can use the C free function on the main zone side.
      // it is legal for the destructor to not be called on a placement new object when the destructor is trivial
      // or the caller does not rely on any side effects of the destructor
      // https://stackoverflow.com/questions/41385355/is-it-ok-not-to-call-the-destructor-on-placement-new-allocated-objects
      void * const pRegistrableMemory = malloc(sizeof(TRegistrable<TFloat>));
      if(nullptr != pRegistrableMemory) {
         try {
            static_assert(std::is_standard_layout<TRegistrable<TFloat>>::value,
               "This allows offsetof, memcpy, memset, inter-language, GPU and cross-machine use where needed");
#if !(defined(__GNUC__) && __GNUC__ < 5)
            static_assert(std::is_trivially_copyable<TRegistrable<TFloat>>::value,
               "This allows offsetof, memcpy, memset, inter-language, GPU and cross-machine use where needed");
#endif // !(defined(__GNUC__) && __GNUC__ < 5)

            // use the in-place constructor to constrct our specialized Loss/Metric function in our pre-reserved memory
            // this works because the *Loss/Metric classes need to be standard layout and trivially copyable anyways
            TRegistrable<TFloat> * const pRegistrable = new (pRegistrableMemory) TRegistrable<TFloat>(*pConfig, args...);
            EBM_ASSERT(nullptr != pRegistrable); // since allocation already happened
            EBM_ASSERT(pRegistrableMemory == pRegistrable);
            // this cannot fail or throw exceptions.  It takes ownership of our pRegistrable pointer
            pRegistrable->FillWrapper(pWrapperOut);
            return false;
         } catch(const SkipRegistrationException &) {
            free(pRegistrableMemory);
            return true;
         } catch(const ParamValOutOfRangeException &) {
            free(pRegistrableMemory);
            throw;
         } catch(const ParamMismatchWithConfigException &) {
            free(pRegistrableMemory);
            throw;
         } catch(const std::bad_alloc &) {
            // it's possible in theory that the constructor allocates some temporary memory, so pass this through
            free(pRegistrableMemory);
            throw;
         } catch(...) {
            // our client Registration functions should only ever throw a limited range of exceptions listed above, 
            // but check anyways
            free(pRegistrableMemory);
            throw RegistrationConstructorException();
         }
      }
      throw std::bad_alloc();
   }

   bool AttemptCreate(
      const Config * const pConfig,
      const char * sRegistration,
      const char * const sRegistrationEnd,
      void * const pWrapperOut
   ) const override {
      sRegistration = CheckRegistrationName(sRegistration, sRegistrationEnd);
      if(nullptr == sRegistration) {
         // we are not the specified registration function
         return true;
      }

      // m_callBack contains the parameter pack that our constructor was created with, so we're regaining access here
      return m_callBack(pConfig, sRegistration, sRegistrationEnd, pWrapperOut);
   }

public:

   RegistrationPack(const char * sRegistrationName, const Args...args) : Registration(sRegistrationName) {

      std::vector<const char *> usedParamNames;
      UnpackRecursive(usedParamNames, args...);

      // hide our parameter pack in a lambda so that we don't have to think about it yet.  Seems easier than using a tuple
      m_callBack = [args...](
         const Config * const pConfig,
         const char * const sRegistration,
         const char * const sRegistrationEnd,
         void * const pWrapperOut
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
            pConfig,
            sRegistration,
            sRegistrationEnd,
            pWrapperOut,
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

} // DEFINED_ZONE_NAME

#endif // REGISTRATION_HPP
