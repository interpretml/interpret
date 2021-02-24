// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// Steps for adding a new objective in C++:
//   1) Copy one of the existing Objective*.h include files (like this one) into a new renamed Objective*.h file
//   2) Modify the class below to handle your new Objective function
//   3) Add [#include "Objective*.h"] to the list of other include files near the top of the Objective.cpp file
//   4) Add [Objective*::AttemptCreateObjective] to the list of objectives in k_registeredObjectives in Objective.cpp
//   5) Recompile the C++ with either build.sh or build.bat depending on your operating system
//   6) Enjoy your new Objective function, and send us a PR on Github if you think others would benefit  :-)

// IMPORTANT: This file should only be included ONCE in the project, and that place should be in the Objective.cpp file

#include <stddef.h>

#include "EbmInternal.h"
#include "Logging.h"
#include "FeatureGroup.h"
#include "ThreadStateBoosting.h"
#include "Objective.h"

class ObjectivePseudoHuber final : public Objective {

   FloatEbmType m_deltaInverted;

   INLINE_ALWAYS ObjectivePseudoHuber(const FloatEbmType deltaInverted) {
      m_deltaInverted = deltaInverted;
   }

public:

   template <typename T>
   INLINE_ALWAYS T CalculatePrediction(const T score) const {
      return score;
   }

   template <typename T>
   INLINE_ALWAYS void CalculateGradient(const T target, const T prediction, T & gradientOut) const {
      const T residualNegative = prediction - target;
      const T residualNegativeFraction = residualNegative * static_cast<T>(m_deltaInverted);
      const T calc = T { 1 } + residualNegativeFraction * residualNegativeFraction;
      const T sqrtCalc = std::sqrt(calc);
      gradientOut = residualNegative / sqrtCalc;
   }

   template <typename T>
   INLINE_ALWAYS void CalculateGradientAndHessian(const T target, const T prediction, T & gradientOut, T & hessianOut) const {
      const T residualNegative = prediction - target;
      const T residualNegativeFraction = residualNegative * static_cast<T>(m_deltaInverted);
      const T calc = T { 1 } + residualNegativeFraction * residualNegativeFraction;
      const T sqrtCalc = std::sqrt(calc);
      gradientOut = residualNegative / sqrtCalc;
      hessianOut = T { 1 } / (calc * sqrtCalc);
   }

   static ErrorEbmType AttemptCreateObjective(
      const char * sObjective, 
      size_t countTargetClasses, 
      const Objective ** const ppObjective
   ) {
      EBM_ASSERT(nullptr != sObjective);
      EBM_ASSERT(nullptr != ppObjective);
      EBM_ASSERT(nullptr == *ppObjective);

      sObjective = IsStringEqualsCaseInsensitive(sObjective, "pseudo_huber");
      if(nullptr == sObjective) {
         // we are not the specified objective
         return Error_None;
      }
      FloatEbmType delta = 1;
      if(0 != *sObjective) {
         if(':' != *sObjective) {
            // we are not the specified objective, but the objective could still be something with a longer string
            // eg: the given tag was "something_else:" but our tag was "something:", so we matched on "something" only
            return Error_None;
         }
         // at this point we now know that we're the specified objective
         while(true) {
            const char * sNext;

            sObjective = SkipWhitespace(sObjective + 1);
            if(0 == *sObjective) {
               // we ended on a ':' at the start, or on a ','.  But just like in some programming languages,
               // we accept the last separator without anything afterwards as a valid formulation
               // eg: "some_objective:" OR "some_objective: some_parameter=1,"
               break;
            }

            // check and handle a possible parameter
            sNext = IsStringEqualsCaseInsensitive(sObjective, "delta");
            if(nullptr != sNext) {
               if('=' == *sNext) {
                  // before this point we could have been seeing a longer version of our proposed tag
                  // eg: the given tag was "something_else=" but our tag was "something="
                  sObjective = sNext + 1;
                  sObjective = ConvertStringToFloat(sObjective, &delta);
                  if(nullptr == sObjective) {
                     return Error_ObjectiveParameterValueMalformed;
                  }
                  if(0 == *sObjective) {
                     break;
                  }
                  if(',' != *sObjective) {
                     return Error_ObjectiveParameterValueMalformed;
                  }
                  continue;
               }
            }

            // if we see a type that we don't understand, then return an error
            return Error_ObjectiveParameterUnknown;
         }
         if(std::isnan(delta) || std::isinf(delta)) {
            // our string readers can read NaN and INF values, so check this
            return Error_ObjectiveParameterValueOutOfRange;
         }
         if(FloatEbmType { 0 } == delta) {
            return Error_ObjectiveParameterValueOutOfRange;
         }
      }
      const FloatEbmType deltaInverted = FloatEbmType { 1 } / delta;
      EBM_ASSERT(!std::isnan(deltaInverted)); // we checked for 0 and NaN above
      if(std::isinf(deltaInverted)) {
         return Error_ObjectiveParameterValueOutOfRange;
      }

      if(1 != countTargetClasses) {
         return Error_ObjectiveCountTargetClassesInvalid;
      }

      *ppObjective = new ObjectivePseudoHuber(deltaInverted);
      return Error_None;
   }

   template<
      ptrdiff_t compilerCountItemsPerBitPackedDataUnit
   >
   ErrorEbmType ApplyModelUpdateTrainingTemplated(
      ThreadStateBoosting * const pThreadStateBoosting,
      const FeatureGroup * const pFeatureGroup
   ) const {
      return Objective::ApplyModelUpdateTrainingShared<
         std::remove_pointer<decltype(this)>::type,
         compilerCountItemsPerBitPackedDataUnit
      >(
         pThreadStateBoosting,
         pFeatureGroup
      );
   }

   template<
      ptrdiff_t compilerCountItemsPerBitPackedDataUnit
   >
   ErrorEbmType ApplyModelUpdateValidationTemplated(
      ThreadStateBoosting * const pThreadStateBoosting,
      const FeatureGroup * const pFeatureGroup,
      FloatEbmType * const pMetricOut
   ) const {
      return Objective::ApplyModelUpdateValidationShared<
         std::remove_pointer<decltype(this)>::type,
         compilerCountItemsPerBitPackedDataUnit
      >(
         pThreadStateBoosting,
         pFeatureGroup,
         pMetricOut
      );
   }

   ErrorEbmType ApplyModelUpdateTraining(
      ThreadStateBoosting * const pThreadStateBoosting,
      const FeatureGroup * const pFeatureGroup
   ) const override {
      return Objective::ApplyModelUpdateTrainingExpand<std::remove_pointer<decltype(this)>::type>(
         pThreadStateBoosting, 
         pFeatureGroup
      );
   }

   ErrorEbmType ApplyModelUpdateValidation(
      ThreadStateBoosting * const pThreadStateBoosting,
      const FeatureGroup * const pFeatureGroup,
      FloatEbmType * const pMetricOut
   ) const override {
      return Objective::ApplyModelUpdateValidationExpand<std::remove_pointer<decltype(this)>::type>(
         pThreadStateBoosting, 
         pFeatureGroup,
         pMetricOut
      );
   }
};
