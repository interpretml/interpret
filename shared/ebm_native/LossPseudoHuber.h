// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// Steps for adding a new objective in C++:
//   1) Copy one of the existing Loss*.h include files (like this one) into a new renamed Loss*.h file
//   2) Modify the class below to handle your new Loss function
//   3) Add [#include "Loss*.h"] to the list of other include files near the top of the Loss.cpp file
//   4) Add [Loss*::AttemptCreateLoss] to the list of objectives in k_registeredLosss in Loss.cpp
//   5) Recompile the C++ with either build.sh or build.bat depending on your operating system
//   6) Enjoy your new Loss function, and send us a PR on Github if you think others would benefit  :-)

// IMPORTANT: This file should only be included ONCE in the project, and that place should be in the Loss.cpp file

#include <stddef.h>

#include "EbmInternal.h"
#include "Logging.h"
#include "FeatureGroup.h"
#include "ThreadStateBoosting.h"

#include "Loss.h"

class LossPseudoHuber final : public Loss {

   FloatEbmType m_deltaInverted;

   INLINE_ALWAYS LossPseudoHuber(const FloatEbmType deltaInverted) {
      m_deltaInverted = deltaInverted;
   }

public:

   template <typename T>
   INLINE_ALWAYS T CalculatePrediction(const T score) const {
      return score;
   }

   template <typename T>
   INLINE_ALWAYS T CalculateGradient(const T target, const T prediction) const {
      const T residualNegative = prediction - target;
      const T residualNegativeFraction = residualNegative * static_cast<T>(m_deltaInverted);
      const T calc = T { 1 } + residualNegativeFraction * residualNegativeFraction;
      const T sqrtCalc = std::sqrt(calc);
      return residualNegative / sqrtCalc;
   }

   template <typename T>
   INLINE_ALWAYS void CalculateHessian(const T target, const T prediction) const {
      const T residualNegative = prediction - target;
      const T residualNegativeFraction = residualNegative * static_cast<T>(m_deltaInverted);
      const T calc = T { 1 } + residualNegativeFraction * residualNegativeFraction;
      const T sqrtCalc = std::sqrt(calc);
      return T { 1 } / (calc * sqrtCalc);
   }

   static ErrorEbmType AttemptCreateLoss(
      const char * sLoss, 
      size_t countTargetClasses, 
      const Loss ** const ppLoss
   ) {
      EBM_ASSERT(nullptr != sLoss);
      EBM_ASSERT(nullptr != ppLoss);
      EBM_ASSERT(nullptr == *ppLoss);

      static const char k_sLossTag[] = "pseudo_huber";
      sLoss = IsStringEqualsCaseInsensitive(sLoss, k_sLossTag);
      if(nullptr == sLoss) {
         // we are not the specified objective
         return Error_None;
      }
      FloatEbmType delta = 1;
      if(0 != *sLoss) {
         if(':' != *sLoss) {
            // we are not the specified objective, but the objective could still be something with a longer string
            // eg: the given tag was "something_else:" but our tag was "something:", so we matched on "something" only
            return Error_None;
         }
         // at this point we now know that we're the specified objective
         while(true) {
            const char * sNext;

            sLoss = SkipWhitespace(sLoss + 1);
            if(0 == *sLoss) {
               // we ended on a ':' at the start, or on a ','.  But just like in some programming languages,
               // we accept the last separator without anything afterwards as a valid formulation
               // eg: "some_objective:" OR "some_objective: some_parameter=1,"
               break;
            }

            // check and handle a possible parameter
            static const char k_sDeltaTag[] = "delta";
            sNext = IsStringEqualsCaseInsensitive(sLoss, k_sDeltaTag);
            if(nullptr != sNext) {
               if('=' == *sNext) {
                  // before this point we could have been seeing a longer version of our proposed tag
                  // eg: the given tag was "something_else=" but our tag was "something="
                  sLoss = sNext + 1;
                  sLoss = ConvertStringToFloat(sLoss, &delta);
                  if(nullptr == sLoss) {
                     return Error_LossParameterValueMalformed;
                  }
                  if(0 == *sLoss) {
                     break;
                  }
                  if(',' != *sLoss) {
                     return Error_LossParameterValueMalformed;
                  }
                  continue;
               }
            }

            // if we see a type that we don't understand, then return an error
            return Error_LossParameterUnknown;
         }
         if(std::isnan(delta) || std::isinf(delta)) {
            // our string readers can read NaN and INF values, so check this
            return Error_LossParameterValueOutOfRange;
         }
         if(FloatEbmType { 0 } == delta) {
            return Error_LossParameterValueOutOfRange;
         }
      }
      const FloatEbmType deltaInverted = FloatEbmType { 1 } / delta;
      EBM_ASSERT(!std::isnan(deltaInverted)); // we checked for 0 and NaN above
      if(std::isinf(deltaInverted)) {
         return Error_LossParameterValueOutOfRange;
      }

      if(1 != countTargetClasses) {
         return Error_LossCountTargetClassesInvalid;
      }

      *ppLoss = new LossPseudoHuber(deltaInverted);
      return Error_None;
   }

   LOSS_DEFAULT_MECHANICS_PUT_AT_END_OF_CLASS

};
