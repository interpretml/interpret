// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef OBJECTIVE_H
#define OBJECTIVE_H

#include <stddef.h> // size_t, ptrdiff_t

#include "EbmInternal.h" // INLINE_ALWAYS
#include "Logging.h" // EBM_ASSERT & LOG
#include "FeatureGroup.h"
#include "ThreadStateBoosting.h"

class Objective {

   template<typename TObjective, ptrdiff_t compilerCountItemsPerBitPackedDataUnitPossible>
   class ApplyModelUpdateTrainingBits final {
      ApplyModelUpdateTrainingBits() = delete; // this is a static class.  Do not construct

   public:

      INLINE_ALWAYS static ErrorEbmType Func(
         const Objective * const pObjective,
         ThreadStateBoosting * const pThreadStateBoosting,
         const FeatureGroup * const pFeatureGroup
      ) {
         const ptrdiff_t runtimeCountItemsPerBitPackedDataUnit = pFeatureGroup->GetCountItemsPerBitPackedDataUnit();

         EBM_ASSERT(k_cItemsPerBitPackedDataUnitDynamic2 != runtimeCountItemsPerBitPackedDataUnit);
         EBM_ASSERT(runtimeCountItemsPerBitPackedDataUnit <= k_cItemsPerBitPackedDataUnitMax2);
         EBM_ASSERT(k_cItemsPerBitPackedDataUnitMin2 <= runtimeCountItemsPerBitPackedDataUnit || 
            k_cItemsPerBitPackedDataUnitNone == runtimeCountItemsPerBitPackedDataUnit);
         static_assert(compilerCountItemsPerBitPackedDataUnitPossible <= k_cItemsPerBitPackedDataUnitMax2, 
            "We can't have this many items in a data pack.");
         static_assert(k_cItemsPerBitPackedDataUnitMin2 <= compilerCountItemsPerBitPackedDataUnitPossible ||
            k_cItemsPerBitPackedDataUnitDynamic2 == compilerCountItemsPerBitPackedDataUnitPossible ||
            k_cItemsPerBitPackedDataUnitNone == compilerCountItemsPerBitPackedDataUnitPossible,
            "We can't have this few items in a data pack.");
         if(compilerCountItemsPerBitPackedDataUnitPossible == runtimeCountItemsPerBitPackedDataUnit) {
            return pObjective->ApplyModelUpdateTrainingExpand<TObjective, compilerCountItemsPerBitPackedDataUnitPossible>(
               pThreadStateBoosting,
               pFeatureGroup
            );
         } else {
            return ApplyModelUpdateTrainingBits<
               TObjective,
               GetNextCountItemsBitPacked2(compilerCountItemsPerBitPackedDataUnitPossible)
            >::Func(
               pObjective,
               pThreadStateBoosting,
               pFeatureGroup
            );
         }
      }
   };

   template<typename TObjective>
   class ApplyModelUpdateTrainingBits<TObjective, k_cItemsPerBitPackedDataUnitNone> final {
   public:

      ApplyModelUpdateTrainingBits() = delete; // this is a static class.  Do not construct

      INLINE_ALWAYS static ErrorEbmType Func(
         const Objective * const pObjective,
         ThreadStateBoosting * const pThreadStateBoosting,
         const FeatureGroup * const pFeatureGroup
      ) {
         EBM_ASSERT(k_cItemsPerBitPackedDataUnitNone == pFeatureGroup->GetCountItemsPerBitPackedDataUnit());
         return pObjective->ApplyModelUpdateTrainingExpand<TObjective, k_cItemsPerBitPackedDataUnitNone>(
            pThreadStateBoosting,
            pFeatureGroup
         );
      }
   };

   template<typename TObjective, ptrdiff_t compilerCountItemsPerBitPackedDataUnitPossible>
   class ApplyModelUpdateValidationBits final {
      ApplyModelUpdateValidationBits() = delete; // this is a static class.  Do not construct

   public:

      INLINE_ALWAYS static ErrorEbmType Func(
         const Objective * const pObjective,
         ThreadStateBoosting * const pThreadStateBoosting,
         const FeatureGroup * const pFeatureGroup,
         FloatEbmType * const pMetricOut
      ) {
         const ptrdiff_t runtimeCountItemsPerBitPackedDataUnit = pFeatureGroup->GetCountItemsPerBitPackedDataUnit();

         EBM_ASSERT(k_cItemsPerBitPackedDataUnitDynamic2 != runtimeCountItemsPerBitPackedDataUnit);
         EBM_ASSERT(runtimeCountItemsPerBitPackedDataUnit <= k_cItemsPerBitPackedDataUnitMax2);
         EBM_ASSERT(k_cItemsPerBitPackedDataUnitMin2 <= runtimeCountItemsPerBitPackedDataUnit ||
            k_cItemsPerBitPackedDataUnitNone == runtimeCountItemsPerBitPackedDataUnit);
         static_assert(compilerCountItemsPerBitPackedDataUnitPossible <= k_cItemsPerBitPackedDataUnitMax2, 
            "We can't have this many items in a data pack.");
         static_assert(k_cItemsPerBitPackedDataUnitMin2 <= compilerCountItemsPerBitPackedDataUnitPossible ||
            k_cItemsPerBitPackedDataUnitDynamic2 == compilerCountItemsPerBitPackedDataUnitPossible ||
            k_cItemsPerBitPackedDataUnitNone == compilerCountItemsPerBitPackedDataUnitPossible,
            "We can't have this few items in a data pack.");
         if(compilerCountItemsPerBitPackedDataUnitPossible == runtimeCountItemsPerBitPackedDataUnit) {
            return pObjective->ApplyModelUpdateValidationExpand<TObjective, compilerCountItemsPerBitPackedDataUnitPossible>(
               pThreadStateBoosting,
               pFeatureGroup,
               pMetricOut
            );
         } else {
            return ApplyModelUpdateValidationBits<
               TObjective,
               GetNextCountItemsBitPacked2(compilerCountItemsPerBitPackedDataUnitPossible)
            >::Func(
               pObjective,
               pThreadStateBoosting,
               pFeatureGroup,
               pMetricOut
            );
         }
      }
   };

   template<typename TObjective>
   class ApplyModelUpdateValidationBits<TObjective, k_cItemsPerBitPackedDataUnitNone> final {
   public:

      ApplyModelUpdateValidationBits() = delete; // this is a static class.  Do not construct

      INLINE_ALWAYS static ErrorEbmType Func(
         const Objective * const pObjective,
         ThreadStateBoosting * const pThreadStateBoosting,
         const FeatureGroup * const pFeatureGroup,
         FloatEbmType * const pMetricOut
      ) {
         EBM_ASSERT(k_cItemsPerBitPackedDataUnitNone == pFeatureGroup->GetCountItemsPerBitPackedDataUnit());
         return pObjective->ApplyModelUpdateValidationExpand<TObjective, k_cItemsPerBitPackedDataUnitNone>(
            pThreadStateBoosting,
            pFeatureGroup,
            pMetricOut
         );
      }
   };

   template<typename TObjective, ptrdiff_t compilerCountItemsPerBitPackedDataUnit>
   INLINE_RELEASE_TEMPLATED ErrorEbmType ApplyModelUpdateTrainingExpand(
      ThreadStateBoosting * const pThreadStateBoosting,
      const FeatureGroup * const pFeatureGroup
   ) const {
      static_assert(std::is_base_of<Objective, TObjective>::value, "TObjective must inherit from Objective");
      const TObjective * const pTObjective = static_cast<const TObjective *>(this);
      return pTObjective->template ApplyModelUpdateTrainingTemplated<compilerCountItemsPerBitPackedDataUnit>(
         pThreadStateBoosting, 
         pFeatureGroup
      );
   }

   template<typename TObjective, ptrdiff_t compilerCountItemsPerBitPackedDataUnit>
   INLINE_RELEASE_TEMPLATED ErrorEbmType ApplyModelUpdateValidationExpand(
      ThreadStateBoosting * const pThreadStateBoosting,
      const FeatureGroup * const pFeatureGroup,
      FloatEbmType * const pMetricOut
   ) const {
      static_assert(std::is_base_of<Objective, TObjective>::value, "TObjective must inherit from Objective");
      const TObjective * const pTObjective = static_cast<const TObjective *>(this);
      return pTObjective->template ApplyModelUpdateValidationTemplated<compilerCountItemsPerBitPackedDataUnit>(
         pThreadStateBoosting, 
         pFeatureGroup,
         pMetricOut
      );
   }

protected:

   Objective() = default;

   template<typename TObjective, ptrdiff_t compilerCountItemsPerBitPackedDataUnit>
   ErrorEbmType ApplyModelUpdateTrainingShared(
      ThreadStateBoosting * const pThreadStateBoosting,
      const FeatureGroup * const pFeatureGroup
   ) const {
      static_assert(std::is_base_of<Objective, TObjective>::value, "TObjective must inherit from Objective");
      const TObjective * const pTObjective = static_cast<const TObjective *>(this);
      Booster * const pBooster = pThreadStateBoosting->GetBooster();

      UNUSED(pTObjective);
      UNUSED(pBooster);
      UNUSED(pFeatureGroup);

      constexpr bool bOnlyOneBin = k_cItemsPerBitPackedDataUnitNone == compilerCountItemsPerBitPackedDataUnit;
      if(bOnlyOneBin) {
      } else {
         const ptrdiff_t runtimeCountItemsPerBitPackedDataUnit = pFeatureGroup->GetCountItemsPerBitPackedDataUnit();
         UNUSED(runtimeCountItemsPerBitPackedDataUnit);

         //DataFrameBoosting * const pTrainingSet = pBooster->GetTrainingSet();

         //const size_t cSamples = pTrainingSet->GetCountSamples();
         //EBM_ASSERT(1 <= cSamples);
         //EBM_ASSERT(1 <= pFeatureGroup->GetCountSignificantDimensions());

         //const size_t cItemsPerBitPackedDataUnit = GET_COUNT_ITEMS_PER_BIT_PACKED_DATA_UNIT(
         //   compilerCountItemsPerBitPackedDataUnit,
         //   runtimeCountItemsPerBitPackedDataUnit
         //);
         //EBM_ASSERT(1 <= cItemsPerBitPackedDataUnit);
         //EBM_ASSERT(cItemsPerBitPackedDataUnit <= k_cBitsForStorageType);
         //const size_t cBitsPerItemMax = GetCountBits(cItemsPerBitPackedDataUnit);
         //EBM_ASSERT(1 <= cBitsPerItemMax);
         //EBM_ASSERT(cBitsPerItemMax <= k_cBitsForStorageType);
         //const size_t maskBits = std::numeric_limits<size_t>::max() >> (k_cBitsForStorageType - cBitsPerItemMax);

         //const FloatEbmType * const aModelFeatureGroupUpdateTensor = pThreadStateBoosting->GetAccumulatedModelUpdate()->GetValuePointer();
         //EBM_ASSERT(nullptr != aModelFeatureGroupUpdateTensor);

         //FloatEbmType * pGradientAndHessian = pTrainingSet->GetGradientsAndHessiansPointer();
         //const StorageDataType * pInputData = pTrainingSet->GetInputDataPointer(pFeatureGroup);
         //const FloatEbmType * pTargetData = pTrainingSet->GetTargetDataPointer();
         //FloatEbmType * pPredictorScores = pTrainingSet->GetPredictorScores();

         //// this shouldn't overflow since we're accessing existing memory
         //const FloatEbmType * const pPredictorScoresTrueEnd = pPredictorScores + cSamples;
         //const FloatEbmType * pPredictorScoresExit = pPredictorScoresTrueEnd;
         //const FloatEbmType * pPredictorScoresInnerEnd = pPredictorScoresTrueEnd;
         //if(cSamples <= cItemsPerBitPackedDataUnit) {
         //   goto one_last_loop;
         //}
         //pPredictorScoresExit = pPredictorScoresTrueEnd - ((cSamples - 1) % cItemsPerBitPackedDataUnit + 1);
         //EBM_ASSERT(pPredictorScores < pPredictorScoresExit);
         //EBM_ASSERT(pPredictorScoresExit < pPredictorScoresTrueEnd);

         //do {
         //   pPredictorScoresInnerEnd = pPredictorScores + cItemsPerBitPackedDataUnit;
         //   // jumping back into this loop and changing pPredictorScoresInnerEnd to a dynamic value that isn't compile time determinable causes this 
         //   // function to NOT be optimized for templated cItemsPerBitPackedDataUnit, but that's ok since avoiding one unpredictable branch here is negligible
         //one_last_loop:;
         //   // we store the already multiplied dimensional value in *pInputData
         //   size_t iTensorBinCombined = static_cast<size_t>(*pInputData);
         //   ++pInputData;
         //   do {
         //      FloatEbmType targetData = *pTargetData;
         //      ++pTargetData;

         //      const size_t iTensorBin = maskBits & iTensorBinCombined;

         //      const FloatEbmType smallChangeToPredictorScores = aModelFeatureGroupUpdateTensor[iTensorBin];
         //      // this will apply a small fix to our existing TrainingPredictorScores, either positive or negative, whichever is needed
         //      const FloatEbmType predictorScore = *pPredictorScores + smallChangeToPredictorScores;
         //      *pPredictorScores = predictorScore;
         //      ++pPredictorScores;

         //      const FloatEbmType prediction = pTObjective->CalculatePrediction(predictorScore);

         //      FloatEbmType gradient;
         //      FloatEbmType hessian;
         //      pTObjective->CalculateGradientAndHessian(targetData, prediction, gradient, hessian);

         //      *pGradientAndHessian = gradient;
         //      *(pGradientAndHessian + 1) = hessian;
         //      pGradientAndHessian += 2;

         //      iTensorBinCombined >>= cBitsPerItemMax;
         //   } while(pPredictorScoresInnerEnd != pPredictorScores);
         //} while(pPredictorScoresExit != pPredictorScores);

         //// first time through?
         //if(pPredictorScoresTrueEnd != pPredictorScores) {
         //   pPredictorScoresInnerEnd = pPredictorScoresTrueEnd;
         //   pPredictorScoresExit = pPredictorScoresTrueEnd;
         //   goto one_last_loop;
         //}
      }
      return Error_None;
   }

   template<typename TObjective, ptrdiff_t compilerCountItemsPerBitPackedDataUnit>
   ErrorEbmType ApplyModelUpdateValidationShared(
      ThreadStateBoosting * const pThreadStateBoosting,
      const FeatureGroup * const pFeatureGroup,
      FloatEbmType * const pMetricOut
   ) const {
      static_assert(std::is_base_of<Objective, TObjective>::value, "TObjective must inherit from Objective");
      const TObjective * const pTObjective = static_cast<const TObjective *>(this);
      Booster * const pBooster = pThreadStateBoosting->GetBooster();

      UNUSED(pTObjective);
      UNUSED(pBooster);
      UNUSED(pFeatureGroup);

      constexpr bool bOnlyOneBin = k_cItemsPerBitPackedDataUnitNone == compilerCountItemsPerBitPackedDataUnit;
      if(bOnlyOneBin) {
         *pMetricOut = -9999;
      } else {
         const ptrdiff_t runtimeCountItemsPerBitPackedDataUnit = pFeatureGroup->GetCountItemsPerBitPackedDataUnit();
         UNUSED(runtimeCountItemsPerBitPackedDataUnit);

         FloatEbmType gradient;
         FloatEbmType hessian;
         pTObjective->CalculateGradientAndHessian(FloatEbmType { 10 }, FloatEbmType { 5 }, gradient, hessian);
         *pMetricOut = gradient + hessian;
      }
      return Error_None;
   }

   template<typename TObjective>
   INLINE_RELEASE_TEMPLATED ErrorEbmType ApplyModelUpdateTrainingExpand(
      ThreadStateBoosting * const pThreadStateBoosting,
      const FeatureGroup * const pFeatureGroup
   ) const {
      static_assert(std::is_base_of<Objective, TObjective>::value, "TObjective must inherit from Objective");
      return ApplyModelUpdateTrainingBits<TObjective, k_cItemsPerBitPackedDataUnitMax2>::Func(
         this,
         pThreadStateBoosting,
         pFeatureGroup
      );
   }

   template<typename TObjective>
   INLINE_RELEASE_TEMPLATED ErrorEbmType ApplyModelUpdateValidationExpand(
      ThreadStateBoosting * const pThreadStateBoosting,
      const FeatureGroup * const pFeatureGroup,
      FloatEbmType * const pMetricOut
   ) const {
      static_assert(std::is_base_of<Objective, TObjective>::value, "TObjective must inherit from Objective");
      return ApplyModelUpdateValidationBits<TObjective, k_cItemsPerBitPackedDataUnitMax2>::Func(
         this,
         pThreadStateBoosting,
         pFeatureGroup,
         pMetricOut
      );
   }

public:

   virtual FloatEbmType GetUpdateMultiple() const {
      // TODO: we need to actually use this now..
      return FloatEbmType { 1 };
   }

   virtual ErrorEbmType ApplyModelUpdateTraining(
      ThreadStateBoosting * const pThreadStateBoosting,
      const FeatureGroup * const pFeatureGroup
   ) const = 0;

   virtual ErrorEbmType ApplyModelUpdateValidation(
      ThreadStateBoosting * const pThreadStateBoosting,
      const FeatureGroup * const pFeatureGroup,
      FloatEbmType * const pMetricOut
   ) const = 0;

   static ErrorEbmType CreateObjective(
      const char * const sObjective,
      const size_t cTargetClasses,
      const Objective ** const ppObjective
   ) noexcept;

   virtual ~Objective() = default;
};

typedef ErrorEbmType (*ATTEMPT_CREATE_OBJECTIVE)(
   const char * sObjective,
   size_t countTargetClasses,
   const Objective ** const ppObjective
);







//class ObjectiveMulticlass : public Objective {
//   // TODO: FILL IN!
//};
//
//class ObjectiveMulticlassSoftmax final : public ObjectiveMulticlass {
//   // TODO: put this in it's own file
//
//   FloatEbmType m_multiple;
//
//public:
//
//   INLINE_ALWAYS ObjectiveMulticlassSoftmax(const size_t cClasses) {
//      m_multiple = static_cast<FloatEbmType>(999999 * cClasses);
//   }
//
//   INLINE_ALWAYS FloatEbmType GetUpdateMultiple() const noexcept override {
//      return m_multiple;
//   }
//
//};
//
//class ObjectiveBinaryLogLoss final : public Objective {
//   // TODO: put this in it's own file
//public:
//
//   INLINE_ALWAYS ObjectiveBinaryLogLoss() {
//   }
//
//   template <typename T>
//   static INLINE_ALWAYS T CalculatePrediction(const T score) {
//      return score * 10000;
//   }
//
//   template <typename T>
//   static INLINE_ALWAYS T GetGradientFromBoolTarget(const bool target, const T prediction) {
//      return -100000;
//   }
//
//   template <typename T>
//   static INLINE_ALWAYS T GetHessianFromBoolTargetAndGradient(const bool target, const T gradient) {
//      // normally we'd get the hessian from the prediction, but for binary logistic regression it's a bit better
//      // to use the gradient, and the mathematics is a special case where we can do this.
//      return -100000;
//   }
//
//   INLINE_ALWAYS FloatEbmType GetUpdateMultiple() const noexcept override {
//      return FloatEbmType { 1 };
//   }
//
//};
//
//class ObjectiveRegressionMSE final : public Objective {
//   // TODO: put this in it's own file
//public:
//
//   INLINE_ALWAYS ObjectiveRegressionMSE() {
//   }
//
//   // for MSE regression, we get target - score at initialization and only store the gradients, so we never
//   // make a prediction, so we don't need CalculatePrediction(...)
//
//   template <typename T>
//   static INLINE_ALWAYS T GetGradientFromGradientPrev(const T target, const T gradientPrev) {
//      // for MSE regression, we get target - score at initialization and only store the gradients, so we
//      // never need the targets.  We just work from the previous gradients.
//
//      return -100000;
//   }
//
//   INLINE_ALWAYS FloatEbmType GetUpdateMultiple() const noexcept override {
//      return FloatEbmType { 1 };
//   }
//
//};

#endif // OBJECTIVE_H
