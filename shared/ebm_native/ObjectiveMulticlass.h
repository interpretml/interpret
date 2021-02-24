// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef OBJECTIVE_MULTICLASS_H
#define OBJECTIVE_MULTICLASS_H

#include <stddef.h> // size_t, ptrdiff_t

#include "EbmInternal.h" // INLINE_ALWAYS
#include "Logging.h" // EBM_ASSERT & LOG
#include "FeatureGroup.h"
#include "ThreadStateBoosting.h"

#include "Objective.h"

class ObjectiveMulticlass : public Objective {
   template<
      typename TObjective,
      ptrdiff_t compilerCountItemsPerBitPackedDataUnit,
      ptrdiff_t compilerCountTargetClassesPossible
   >
   class ApplyModelUpdateTrainingTargetClasses final {
   public:

      ApplyModelUpdateTrainingTargetClasses() = delete; // this is a static class.  Do not construct

      INLINE_ALWAYS static ErrorEbmType Func(
         const ObjectiveMulticlass * const pObjectiveMulticlass,
         ThreadStateBoosting * const pThreadStateBoosting,
         const FeatureGroup * const pFeatureGroup
      ) {
         static_assert(compilerCountTargetClassesPossible <= k_cCompilerOptimizedTargetClassesMax, "We can't have this many items in a data pack.");

         Booster * const pBooster = pThreadStateBoosting->GetBooster();
         const ptrdiff_t runtimeCountTargetClasses = pBooster->GetRuntimeLearningTypeOrCountTargetClasses();
         if(compilerCountTargetClassesPossible == runtimeCountTargetClasses) {
            return pObjectiveMulticlass->ApplyModelUpdateTrainingExpand<
               TObjective,
               compilerCountItemsPerBitPackedDataUnit,
               compilerCountTargetClassesPossible
            >(
               pThreadStateBoosting,
               pFeatureGroup
            );
         } else {
            return ApplyModelUpdateTrainingTargetClasses<
               TObjective,
               compilerCountItemsPerBitPackedDataUnit,
               compilerCountTargetClassesPossible + 1
            >::Func(
               pObjectiveMulticlass,
               pThreadStateBoosting,
               pFeatureGroup
            );
         }
      }
   };

   template<
      typename TObjective,
      ptrdiff_t compilerCountItemsPerBitPackedDataUnit
   >
      class ApplyModelUpdateTrainingTargetClasses<
      TObjective,
      compilerCountItemsPerBitPackedDataUnit,
      k_cCompilerOptimizedTargetClassesMax + 1
   > final {
   public:

      ApplyModelUpdateTrainingTargetClasses() = delete; // this is a static class.  Do not construct

      INLINE_ALWAYS static ErrorEbmType Func(
         const ObjectiveMulticlass * const pObjectiveMulticlass,
         ThreadStateBoosting * const pThreadStateBoosting,
         const FeatureGroup * const pFeatureGroup
      ) {
         return pObjectiveMulticlass->ApplyModelUpdateTrainingExpand<
            TObjective,
            compilerCountItemsPerBitPackedDataUnit,
            k_dynamicClassification
         >(
            pThreadStateBoosting,
            pFeatureGroup
         );
      }
   };

   template<
      typename TObjective,
      ptrdiff_t compilerCountItemsPerBitPackedDataUnit,
      ptrdiff_t compilerCountTargetClassesPossible
   >
   class ApplyModelUpdateValidationTargetClasses final {
   public:

      ApplyModelUpdateValidationTargetClasses() = delete; // this is a static class.  Do not construct

      INLINE_ALWAYS static ErrorEbmType Func(
         const ObjectiveMulticlass * const pObjectiveMulticlass,
         ThreadStateBoosting * const pThreadStateBoosting,
         const FeatureGroup * const pFeatureGroup,
         FloatEbmType * const pMetricOut
      ) {
         static_assert(compilerCountTargetClassesPossible <= k_cCompilerOptimizedTargetClassesMax, "We can't have this many items in a data pack.");

         Booster * const pBooster = pThreadStateBoosting->GetBooster();
         const ptrdiff_t runtimeCountTargetClasses = pBooster->GetRuntimeLearningTypeOrCountTargetClasses();
         if(compilerCountTargetClassesPossible == runtimeCountTargetClasses) {
            return pObjectiveMulticlass->ApplyModelUpdateValidationExpand<
               TObjective,
               compilerCountItemsPerBitPackedDataUnit,
               compilerCountTargetClassesPossible
            >(
               pThreadStateBoosting,
               pFeatureGroup,
               pMetricOut
            );
         } else {
            return ApplyModelUpdateValidationTargetClasses<
               TObjective,
               compilerCountItemsPerBitPackedDataUnit,
               compilerCountTargetClassesPossible + 1
            >::Func(
               pObjectiveMulticlass,
               pThreadStateBoosting,
               pFeatureGroup,
               pMetricOut
            );
         }
      }
   };

   template<
      typename TObjective,
      ptrdiff_t compilerCountItemsPerBitPackedDataUnit
   >
   class ApplyModelUpdateValidationTargetClasses<
      TObjective,
      compilerCountItemsPerBitPackedDataUnit,
      k_cCompilerOptimizedTargetClassesMax + 1
   > final {
   public:

      ApplyModelUpdateValidationTargetClasses() = delete; // this is a static class.  Do not construct

      INLINE_ALWAYS static ErrorEbmType Func(
         const ObjectiveMulticlass * const pObjectiveMulticlass,
         ThreadStateBoosting * const pThreadStateBoosting,
         const FeatureGroup * const pFeatureGroup,
         FloatEbmType * const pMetricOut
      ) {
         return pObjectiveMulticlass->ApplyModelUpdateValidationExpand<
            TObjective,
            compilerCountItemsPerBitPackedDataUnit,
            k_dynamicClassification
         >(
            pThreadStateBoosting,
            pFeatureGroup,
            pMetricOut
         );
      }
   };

   template<
      typename TObjective,
      ptrdiff_t compilerCountItemsPerBitPackedDataUnit,
      ptrdiff_t compilerCountTargetClasses
   >
   ErrorEbmType ApplyModelUpdateTrainingExpand(
      ThreadStateBoosting * const pThreadStateBoosting,
      const FeatureGroup * const pFeatureGroup
   ) const {
      static_assert(std::is_base_of<ObjectiveMulticlass, TObjective>::value, "TObjective must inherit from Objective");
      const TObjective * const pTObjective = static_cast<const TObjective *>(this);
      return pTObjective->template ApplyModelUpdateTrainingTemplated<
         compilerCountItemsPerBitPackedDataUnit,
         compilerCountTargetClasses
      >(
         pThreadStateBoosting,
         pFeatureGroup
      );
   }

   template<
      typename TObjective,
      ptrdiff_t compilerCountItemsPerBitPackedDataUnit,
      ptrdiff_t compilerCountTargetClasses
   >
   ErrorEbmType ApplyModelUpdateValidationExpand(
      ThreadStateBoosting * const pThreadStateBoosting,
      const FeatureGroup * const pFeatureGroup,
      FloatEbmType * const pMetricOut
   ) const {
      static_assert(std::is_base_of<ObjectiveMulticlass, TObjective>::value, "TObjective must inherit from Objective");
      const TObjective * const pTObjective = static_cast<const TObjective *>(this);
      return pTObjective->template ApplyModelUpdateValidationTemplated<
         compilerCountItemsPerBitPackedDataUnit,
         compilerCountTargetClasses
      >(
         pThreadStateBoosting,
         pFeatureGroup,
         pMetricOut
      );
   }

protected:

   ObjectiveMulticlass() = default;

   template<
      typename TObjective,
      ptrdiff_t compilerCountItemsPerBitPackedDataUnit
   >
   INLINE_RELEASE_TEMPLATED ErrorEbmType ApplyModelUpdateTrainingExpand(
      ThreadStateBoosting * const pThreadStateBoosting,
      const FeatureGroup * const pFeatureGroup
   ) const {
      static_assert(std::is_base_of<ObjectiveMulticlass, TObjective>::value, "TObjective must inherit from Objective");

      constexpr bool bOnlyOneBin = k_cItemsPerBitPackedDataUnitNone == compilerCountItemsPerBitPackedDataUnit;
      if(bOnlyOneBin) {
         // we only have 1 bin so there's no need to blow out our downstream super-efficient code.  Just use dynamic
         return ApplyModelUpdateTrainingExpand<
            TObjective, 
            compilerCountItemsPerBitPackedDataUnit, 
            k_dynamicClassification
         >(
            pThreadStateBoosting,
            pFeatureGroup
         );
      } else {
         return ApplyModelUpdateTrainingTargetClasses<
            TObjective,
            compilerCountItemsPerBitPackedDataUnit,
            3
         >::Func(
            this,
            pThreadStateBoosting,
            pFeatureGroup
         );
      }
   }

   template<
      typename TObjective,
      ptrdiff_t compilerCountItemsPerBitPackedDataUnit
   >
   INLINE_RELEASE_TEMPLATED ErrorEbmType ApplyModelUpdateValidationExpand(
      ThreadStateBoosting * const pThreadStateBoosting,
      const FeatureGroup * const pFeatureGroup,
      FloatEbmType * const pMetricOut
   ) const {
      static_assert(std::is_base_of<ObjectiveMulticlass, TObjective>::value, "TObjective must inherit from Objective");

      constexpr bool bOnlyOneBin = k_cItemsPerBitPackedDataUnitNone == compilerCountItemsPerBitPackedDataUnit;
      if(bOnlyOneBin) {
         // we only have 1 bin so there's no need to blow out our downstream super-efficient code.  Just use dynamic
         return ApplyModelUpdateValidationExpand<
            TObjective,
            compilerCountItemsPerBitPackedDataUnit,
            k_dynamicClassification
         >(
            pThreadStateBoosting,
            pFeatureGroup,
            pMetricOut
         );
      } else {
         return ApplyModelUpdateValidationTargetClasses<
            TObjective,
            compilerCountItemsPerBitPackedDataUnit,
            3
         >::Func(
            this,
            pThreadStateBoosting,
            pFeatureGroup,
            pMetricOut
         );
      }
   }
};

#endif // OBJECTIVE_MULTICLASS_H
