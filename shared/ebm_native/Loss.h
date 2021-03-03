// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// !!! NOTE: To add a new loss/objective function in C++, follow the steps listed at the top of the "Loss.cpp" file !!!

#ifndef LOSS_H
#define LOSS_H

#include <stddef.h> // size_t, ptrdiff_t

#include "EbmInternal.h" // INLINE_ALWAYS
#include "Logging.h" // EBM_ASSERT & LOG
#include "FeatureGroup.h"
#include "ThreadStateBoosting.h"
#include "Config.h"
#include "Registrable.h"

class Loss : public Registrable {

   template<typename TLoss, ptrdiff_t compilerCountClasses>
   struct CountClasses final {
      INLINE_ALWAYS static ErrorEbmType ApplyTraining(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) {
         if(compilerCountClasses == pThreadStateBoosting->GetBooster()->GetRuntimeLearningTypeOrCountTargetClasses()) {
            return BitPack<TLoss, compilerCountClasses, k_cItemsPerBitPackMax2>::ApplyTraining(pLoss, pThreadStateBoosting, pFeatureGroup);
         } else {
            return CountClasses<TLoss, compilerCountClasses + 1>::ApplyTraining(pLoss, pThreadStateBoosting, pFeatureGroup);
         }
      }
      INLINE_ALWAYS static ErrorEbmType ApplyValidation(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) {
         if(compilerCountClasses == pThreadStateBoosting->GetBooster()->GetRuntimeLearningTypeOrCountTargetClasses()) {
            return BitPack<TLoss, compilerCountClasses, k_cItemsPerBitPackMax2>::ApplyValidation(pLoss, pThreadStateBoosting, pFeatureGroup, pMetricOut);
         } else {
            return CountClasses<TLoss, compilerCountClasses + 1>::ApplyValidation(pLoss, pThreadStateBoosting, pFeatureGroup, pMetricOut);
         }
      }
   };
   template<typename TLoss>
   struct CountClasses<TLoss, k_cCompilerOptimizedTargetClassesMax + 1> final {
      INLINE_ALWAYS static ErrorEbmType ApplyTraining(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) {
         return BitPack<TLoss, k_dynamicClassification, k_cItemsPerBitPackMax2>::ApplyTraining(pLoss, pThreadStateBoosting, pFeatureGroup);
      }
      INLINE_ALWAYS static ErrorEbmType ApplyValidation(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) {
         return BitPack<TLoss, k_dynamicClassification, k_cItemsPerBitPackMax2>::ApplyValidation(pLoss, pThreadStateBoosting, pFeatureGroup, pMetricOut);
      }
   };


   template<typename TLoss, ptrdiff_t compilerCountClasses, ptrdiff_t compilerBitPack>
   struct BitPack final {
      INLINE_ALWAYS static ErrorEbmType ApplyTraining(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) {
         if(compilerBitPack == pFeatureGroup->GetBitPack()) {
            return pLoss->BitPackApplyTraining<TLoss, compilerCountClasses, compilerBitPack>(pThreadStateBoosting, pFeatureGroup);
         } else {
            return BitPack<TLoss, compilerCountClasses, GetNextBitPack(compilerBitPack)>::ApplyTraining(pLoss, pThreadStateBoosting, pFeatureGroup);
         }
      }
      INLINE_ALWAYS static ErrorEbmType ApplyValidation(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) {
         if(compilerBitPack == pFeatureGroup->GetBitPack()) {
            return pLoss->BitPackApplyValidation<TLoss, compilerCountClasses, compilerBitPack>(pThreadStateBoosting, pFeatureGroup, pMetricOut);
         } else {
            return BitPack<TLoss, compilerCountClasses, GetNextBitPack(compilerBitPack)>::ApplyValidation(pLoss, pThreadStateBoosting, pFeatureGroup, pMetricOut);
         }
      }
   };
   template<typename TLoss, ptrdiff_t compilerCountClasses>
   struct BitPack<TLoss, compilerCountClasses, k_cItemsPerBitPackNone> final {
      INLINE_ALWAYS static ErrorEbmType ApplyTraining(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) {
         return pLoss->BitPackApplyTraining<TLoss, compilerCountClasses, k_cItemsPerBitPackNone>(pThreadStateBoosting, pFeatureGroup);
      }
      INLINE_ALWAYS static ErrorEbmType ApplyValidation(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) {
         return pLoss->BitPackApplyValidation<TLoss, compilerCountClasses, k_cItemsPerBitPackNone>(pThreadStateBoosting, pFeatureGroup, pMetricOut);
      }
   };


   template<typename TLoss, ptrdiff_t compilerCountClasses, ptrdiff_t compilerBitPack>
   struct RemoveMulti final {
      INLINE_ALWAYS static ErrorEbmType ApplyTraining(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) {
         const TLoss * const pTLoss = static_cast<const TLoss *>(pLoss);
         return pTLoss->template ApplyTrainingMultiTemplated<compilerCountClasses, compilerBitPack>(pThreadStateBoosting, pFeatureGroup);
      }
      INLINE_ALWAYS static ErrorEbmType ApplyValidation(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) {
         const TLoss * const pTLoss = static_cast<const TLoss *>(pLoss);
         return pTLoss->template ApplyValidationMultiTemplated<compilerCountClasses, compilerBitPack>(pThreadStateBoosting, pFeatureGroup, pMetricOut);
      }
   };
   template<typename TLoss, ptrdiff_t compilerBitPack>
   struct RemoveMulti<TLoss, k_regression, compilerBitPack> final {
      INLINE_ALWAYS static ErrorEbmType ApplyTraining(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) {
         const TLoss * const pTLoss = static_cast<const TLoss *>(pLoss);
         return pTLoss->template ApplyTrainingTemplated<compilerBitPack>(pThreadStateBoosting, pFeatureGroup);
      }
      INLINE_ALWAYS static ErrorEbmType ApplyValidation(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) {
         const TLoss * const pTLoss = static_cast<const TLoss *>(pLoss);
         return pTLoss->template ApplyValidationTemplated<compilerBitPack>(pThreadStateBoosting, pFeatureGroup, pMetricOut);
      }
   };


   template<typename TLoss, ptrdiff_t compilerCountClasses, ptrdiff_t compilerBitPack>
   INLINE_RELEASE_TEMPLATED ErrorEbmType BitPackApplyTraining(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) const {
      return RemoveMulti<TLoss, compilerCountClasses, compilerBitPack>::ApplyTraining(this, pThreadStateBoosting, pFeatureGroup);
   }
   template<typename TLoss, ptrdiff_t compilerCountClasses, ptrdiff_t compilerBitPack>
   INLINE_RELEASE_TEMPLATED ErrorEbmType BitPackApplyValidation(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) const {
      return RemoveMulti<TLoss, compilerCountClasses, compilerBitPack>::ApplyValidation(this, pThreadStateBoosting, pFeatureGroup, pMetricOut);
   }


   template<typename TLoss, ptrdiff_t compilerBitPack>
   struct Shared final {
      INLINE_ALWAYS static ErrorEbmType ApplyTraining(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) {
         UNUSED(pLoss);
         UNUSED(pThreadStateBoosting);
         UNUSED(pFeatureGroup);
         return Error_None;
      }
      INLINE_ALWAYS static ErrorEbmType ApplyValidation(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) {
         UNUSED(pLoss);
         UNUSED(pThreadStateBoosting);
         UNUSED(pFeatureGroup);
         UNUSED(pMetricOut);
         return Error_None;
      }
   };
   template<typename TLoss>
   struct Shared<TLoss, k_cItemsPerBitPackNone> final {
      INLINE_ALWAYS static ErrorEbmType ApplyTraining(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) {
         UNUSED(pLoss);
         UNUSED(pThreadStateBoosting);
         UNUSED(pFeatureGroup);
         return Error_None;
      }
      INLINE_ALWAYS static ErrorEbmType ApplyValidation(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) {
         UNUSED(pLoss);
         UNUSED(pThreadStateBoosting);
         UNUSED(pFeatureGroup);
         UNUSED(pMetricOut);
         return Error_None;
      }
   };


   template<typename TLoss, ptrdiff_t compilerCountClasses, ptrdiff_t compilerBitPack>
   struct SharedMulti final {
      INLINE_ALWAYS static ErrorEbmType ApplyTraining(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) {
         UNUSED(pLoss);
         UNUSED(pThreadStateBoosting);
         UNUSED(pFeatureGroup);
         return Error_None;
      }
      INLINE_ALWAYS static ErrorEbmType ApplyValidation(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) {
         UNUSED(pLoss);
         UNUSED(pThreadStateBoosting);
         UNUSED(pFeatureGroup);
         UNUSED(pMetricOut);
         return Error_None;
      }
   };
   template<typename TLoss, ptrdiff_t compilerCountClasses>
   struct SharedMulti<TLoss, compilerCountClasses, k_cItemsPerBitPackNone> final {
      INLINE_ALWAYS static ErrorEbmType ApplyTraining(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) {
         UNUSED(pLoss);
         UNUSED(pThreadStateBoosting);
         UNUSED(pFeatureGroup);
         return Error_None;
      }
      INLINE_ALWAYS static ErrorEbmType ApplyValidation(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) {
         UNUSED(pLoss);
         UNUSED(pThreadStateBoosting);
         UNUSED(pFeatureGroup);
         UNUSED(pMetricOut);
         return Error_None;
      }
   };

protected:

   template<typename TLoss, ptrdiff_t compilerBitPack>
   INLINE_RELEASE_TEMPLATED ErrorEbmType SharedApplyTraining(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) const {
      static_assert(std::is_base_of<Loss, TLoss>::value, "TLoss must inherit from Loss");
      return Shared<TLoss, compilerBitPack>::ApplyTraining(this, pThreadStateBoosting, pFeatureGroup);
   }
   template<typename TLoss, ptrdiff_t compilerBitPack>
   INLINE_RELEASE_TEMPLATED ErrorEbmType SharedApplyValidation(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) const {
      static_assert(std::is_base_of<Loss, TLoss>::value, "TLoss must inherit from Loss");
      return Shared<TLoss, compilerBitPack>::ApplyValidation(this, pThreadStateBoosting, pFeatureGroup, pMetricOut);
   }


   template<typename TLoss, ptrdiff_t compilerCountClasses, ptrdiff_t compilerBitPack>
   INLINE_RELEASE_TEMPLATED ErrorEbmType SharedApplyTrainingMulti(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) const {
      static_assert(std::is_base_of<Loss, TLoss>::value, "TLoss must inherit from Loss");
      return SharedMulti<TLoss, compilerCountClasses, compilerBitPack>::ApplyTraining(this, pThreadStateBoosting, pFeatureGroup);
   }
   template<typename TLoss, ptrdiff_t compilerCountClasses, ptrdiff_t compilerBitPack>
   INLINE_RELEASE_TEMPLATED ErrorEbmType SharedApplyValidationMulti(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) const {
      static_assert(std::is_base_of<Loss, TLoss>::value, "TLoss must inherit from Loss");
      return SharedMulti<TLoss, compilerCountClasses, compilerBitPack>::ApplyValidation(this, pThreadStateBoosting, pFeatureGroup, pMetricOut);
   }


   template<typename TLoss>
   INLINE_RELEASE_TEMPLATED ErrorEbmType LossApplyTraining(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) const {
      static_assert(std::is_base_of<Loss, TLoss>::value, "TLoss must inherit from Loss");
      return BitPack<TLoss, k_regression, k_cItemsPerBitPackMax2>::ApplyTraining(this, pThreadStateBoosting, pFeatureGroup);
   }
   template<typename TLoss>
   INLINE_RELEASE_TEMPLATED ErrorEbmType LossApplyValidation(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) const {
      static_assert(std::is_base_of<Loss, TLoss>::value, "TLoss must inherit from Loss");
      return BitPack<TLoss, k_regression, k_cItemsPerBitPackMax2>::ApplyValidation(this, pThreadStateBoosting, pFeatureGroup, pMetricOut);
   }


   template<typename TLoss>
   INLINE_RELEASE_TEMPLATED ErrorEbmType LossApplyTrainingMulti(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) const {
      static_assert(std::is_base_of<Loss, TLoss>::value, "TLoss must inherit from Loss");
      return CountClasses<TLoss, 3>::ApplyTraining(this, pThreadStateBoosting, pFeatureGroup);
   }
   template<typename TLoss>
   INLINE_RELEASE_TEMPLATED ErrorEbmType LossApplyValidationMulti(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) const {
      static_assert(std::is_base_of<Loss, TLoss>::value, "TLoss must inherit from Loss");
      return CountClasses<TLoss, 3>::ApplyValidation(this, pThreadStateBoosting, pFeatureGroup, pMetricOut);
   }

   Loss() = default;

public:

   // TODO: we should wrap all our calls to the potentially re-written client code with try/catch blocks since we don't know if it throws exceptions

   // TODO: we need to actually use this now..
   virtual FloatEbmType GetUpdateMultiple() const;

   virtual ErrorEbmType ApplyTraining(
      ThreadStateBoosting * const pThreadStateBoosting,
      const FeatureGroup * const pFeatureGroup
   ) const = 0;

   virtual ErrorEbmType ApplyValidation(
      ThreadStateBoosting * const pThreadStateBoosting,
      const FeatureGroup * const pFeatureGroup,
      FloatEbmType * const pMetricOut
   ) const = 0;

   static ErrorEbmType CreateLoss(
      const char * const sLoss,
      const Config * const pConfig,
      const Loss ** const ppLoss
   ) noexcept;
};

// TODO: add parameters to these macros to pass in what options we want in our template generation
#define LOSS_DEFAULT_MECHANICS_PUT_AT_END_OF_CLASS \
   public: \
      template<ptrdiff_t compilerBitPack> \
      ErrorEbmType ApplyTrainingTemplated(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) const { \
         return Loss::SharedApplyTraining<std::remove_pointer<decltype(this)>::type, compilerBitPack>(pThreadStateBoosting, pFeatureGroup); \
      } \
      template<ptrdiff_t compilerBitPack> \
      ErrorEbmType ApplyValidationTemplated(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) const { \
         return Loss::SharedApplyValidation<std::remove_pointer<decltype(this)>::type, compilerBitPack>(pThreadStateBoosting, pFeatureGroup, pMetricOut); \
      } \
      ErrorEbmType ApplyTraining(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) const override { \
         return Loss::LossApplyTraining<std::remove_pointer<decltype(this)>::type>(pThreadStateBoosting, pFeatureGroup); \
      } \
      ErrorEbmType ApplyValidation(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) const override { \
         return Loss::LossApplyValidation<std::remove_pointer<decltype(this)>::type>(pThreadStateBoosting, pFeatureGroup, pMetricOut); \
      }

#define LOSS_MULTI_DEFAULT_MECHANICS_PUT_AT_END_OF_CLASS \
   public: \
      template<ptrdiff_t compilerCountClasses, ptrdiff_t compilerBitPack> \
      ErrorEbmType ApplyTrainingMultiTemplated(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) const { \
         return Loss::SharedApplyTrainingMulti<std::remove_pointer<decltype(this)>::type, compilerCountClasses, compilerBitPack>(pThreadStateBoosting, pFeatureGroup); \
      } \
      template<ptrdiff_t compilerCountClasses, ptrdiff_t compilerBitPack> \
      ErrorEbmType ApplyValidationMultiTemplated(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) const { \
         return Loss::SharedApplyValidationMulti<std::remove_pointer<decltype(this)>::type, compilerCountClasses, compilerBitPack>(pThreadStateBoosting, pFeatureGroup, pMetricOut); \
      } \
      ErrorEbmType ApplyTraining(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) const override { \
         return Loss::LossApplyTrainingMulti<std::remove_pointer<decltype(this)>::type>(pThreadStateBoosting, pFeatureGroup); \
      } \
      ErrorEbmType ApplyValidation(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) const override { \
         return Loss::LossApplyValidationMulti<std::remove_pointer<decltype(this)>::type>(pThreadStateBoosting, pFeatureGroup, pMetricOut); \
      }

#endif // LOSS_H
