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
#include "ObjectiveMulticlass.h"

class ObjectiveMulticlassSoftmax final : public ObjectiveMulticlass {

   size_t m_countTargetClasses;

   INLINE_ALWAYS ObjectiveMulticlassSoftmax(const size_t countTargetClasses) {
      m_countTargetClasses = countTargetClasses;
   }

public:

   static ErrorEbmType AttemptCreateObjective(
      const char * sObjective,
      size_t countTargetClasses,
      const Objective ** const ppObjective
   ) {
      EBM_ASSERT(nullptr != sObjective);
      EBM_ASSERT(nullptr != ppObjective);
      EBM_ASSERT(nullptr == *ppObjective);

      static const char k_sObjectiveTag[] = "multiclass_softmax";
      sObjective = IsStringEqualsCaseInsensitive(sObjective, k_sObjectiveTag);
      if(nullptr == sObjective) {
         // we are not the specified objective
         return Error_None;
      }
      if(0 != *sObjective) {
         // we are not the specified objective, but the objective could still be something with a longer string
         // eg: the given tag was "something_else:" but our tag was "something:", so we matched on "something" only
         return Error_None;
      }

      *ppObjective = new ObjectiveMulticlassSoftmax(countTargetClasses);
      return Error_None;
   }

   template<
      ptrdiff_t compilerCountItemsPerBitPackedDataUnit,
      ptrdiff_t compilerCountTargetClasses
   >
   ErrorEbmType ApplyModelUpdateTrainingTemplated(
      ThreadStateBoosting * const pThreadStateBoosting,
      const FeatureGroup * const pFeatureGroup
   ) const {

      UNUSED(pThreadStateBoosting);
      UNUSED(pFeatureGroup);

      return Error_None;
   }

   template<
      ptrdiff_t compilerCountItemsPerBitPackedDataUnit,
      ptrdiff_t compilerCountTargetClasses
   >
   ErrorEbmType ApplyModelUpdateValidationTemplated(
      ThreadStateBoosting * const pThreadStateBoosting,
      const FeatureGroup * const pFeatureGroup,
      FloatEbmType * const pMetricOut
   ) const {

      UNUSED(pThreadStateBoosting);
      UNUSED(pFeatureGroup);
      UNUSED(pMetricOut);

      return Error_None;
   }

   template<
      ptrdiff_t compilerCountItemsPerBitPackedDataUnit
   >
   ErrorEbmType ApplyModelUpdateTrainingTemplated(
      ThreadStateBoosting * const pThreadStateBoosting,
      const FeatureGroup * const pFeatureGroup
   ) const {
      return ObjectiveMulticlass::ApplyModelUpdateTrainingExpand<
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
      return ObjectiveMulticlass::ApplyModelUpdateValidationExpand<
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
