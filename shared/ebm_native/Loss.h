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

class LossSingletask;
class LossBinary;
class LossMulticlass;
class LossRegression;

class LossMultitask;
class LossMultitaskBinary;
class LossMultitaskMulticlass;
class LossMultitaskRegression;

class Loss : public Registrable {
   template<typename TLoss, typename std::enable_if<!TLoss::IsMultiScore, TLoss>::type * = nullptr>
   constexpr static ptrdiff_t GetInitialCountScores() {
      return k_oneScore;
   }
   template<typename TLoss, typename std::enable_if<TLoss::IsMultiScore && std::is_base_of<LossMultitaskMulticlass, TLoss>::value, TLoss>::type * = nullptr>
   constexpr static ptrdiff_t GetInitialCountScores() {
      return k_dynamicClassification;
   }
   template<typename TLoss, typename std::enable_if<TLoss::IsMultiScore && !std::is_base_of<LossMultitaskMulticlass, TLoss>::value, TLoss>::type * = nullptr>
   constexpr static ptrdiff_t GetInitialCountScores() {
      // TODO : harden this to be able to handle weird values in k_cCompilerOptimizedTargetClassesMax like 0, 1, 2, 3, etc..
      return 3;
   }
   template<typename TLoss>
   INLINE_RELEASE_TEMPLATED ErrorEbmType CountScoresPreApplyTraining(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) const {
      return CountScores<TLoss, GetInitialCountScores<TLoss>()>::ApplyTraining(this, pThreadStateBoosting, pFeatureGroup);
   }
   template<typename TLoss>
   INLINE_RELEASE_TEMPLATED ErrorEbmType CountScoresPreApplyValidation(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) const {
      return CountScores<TLoss, GetInitialCountScores<TLoss>()>::ApplyValidation(this, pThreadStateBoosting, pFeatureGroup, pMetricOut);
   }
   template<typename TLoss, ptrdiff_t cCompilerScores>
   struct CountScores final {
      INLINE_ALWAYS static ErrorEbmType ApplyTraining(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) {
         if(cCompilerScores == pThreadStateBoosting->GetBooster()->GetRuntimeLearningTypeOrCountTargetClasses()) {
            return pLoss->BitPackPreApplyTraining<TLoss, cCompilerScores>(pThreadStateBoosting, pFeatureGroup);
         } else {
            return CountScores<TLoss, k_cCompilerOptimizedTargetClassesMax == cCompilerScores ? k_dynamicClassification : cCompilerScores + 1>::ApplyTraining(pLoss, pThreadStateBoosting, pFeatureGroup);
         }
      }
      INLINE_ALWAYS static ErrorEbmType ApplyValidation(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) {
         if(cCompilerScores == pThreadStateBoosting->GetBooster()->GetRuntimeLearningTypeOrCountTargetClasses()) {
            return pLoss->BitPackPreApplyValidation<TLoss, cCompilerScores>(pThreadStateBoosting, pFeatureGroup, pMetricOut);
         } else {
            return CountScores<TLoss, k_cCompilerOptimizedTargetClassesMax == cCompilerScores ? k_dynamicClassification : cCompilerScores + 1>::ApplyValidation(pLoss, pThreadStateBoosting, pFeatureGroup, pMetricOut);
         }
      }
   };
   template<typename TLoss>
   struct CountScores<TLoss, k_dynamicClassification> final {
      INLINE_ALWAYS static ErrorEbmType ApplyTraining(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) {
         return pLoss->BitPackPreApplyTraining<TLoss, k_dynamicClassification>(pThreadStateBoosting, pFeatureGroup);
      }
      INLINE_ALWAYS static ErrorEbmType ApplyValidation(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) {
         return pLoss->BitPackPreApplyValidation<TLoss, k_dynamicClassification>(pThreadStateBoosting, pFeatureGroup, pMetricOut);
      }
   };
   template<typename TLoss>
   struct CountScores<TLoss, k_oneScore> final {
      INLINE_ALWAYS static ErrorEbmType ApplyTraining(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) {
         return pLoss->BitPackPreApplyTraining<TLoss, k_oneScore>(pThreadStateBoosting, pFeatureGroup);
      }
      INLINE_ALWAYS static ErrorEbmType ApplyValidation(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) {
         return pLoss->BitPackPreApplyValidation<TLoss, k_oneScore>(pThreadStateBoosting, pFeatureGroup, pMetricOut);
      }
   };



   template<typename TLoss, ptrdiff_t cCompilerScores>
   INLINE_RELEASE_TEMPLATED ErrorEbmType BitPackPreApplyTraining(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) const {
      if(k_cItemsPerBitPackNone == pFeatureGroup->GetBitPack()) {
         return BitPackPostApplyTraining<TLoss, cCompilerScores, k_cItemsPerBitPackNone>(pThreadStateBoosting, pFeatureGroup);
      } else {
         return BitPack<TLoss, cCompilerScores, k_cItemsPerBitPackMax2>::ApplyTraining(this, pThreadStateBoosting, pFeatureGroup);
      }
   }
   template<typename TLoss, ptrdiff_t cCompilerScores>
   INLINE_RELEASE_TEMPLATED ErrorEbmType BitPackPreApplyValidation(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) const {
      if(k_cItemsPerBitPackNone == pFeatureGroup->GetBitPack()) {
         return BitPackPostApplyValidation<TLoss, cCompilerScores, k_cItemsPerBitPackNone>(pThreadStateBoosting, pFeatureGroup, pMetricOut);
      } else {
         return BitPack<TLoss, cCompilerScores, k_cItemsPerBitPackMax2>::ApplyValidation(this, pThreadStateBoosting, pFeatureGroup, pMetricOut);
      }
   }
   template<typename TLoss, ptrdiff_t cCompilerScores, ptrdiff_t cCompilerPack>
   struct BitPack final {
      INLINE_ALWAYS static ErrorEbmType ApplyTraining(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) {
         if(cCompilerPack == pFeatureGroup->GetBitPack()) {
            return pLoss->BitPackPostApplyTraining<TLoss, cCompilerScores, cCompilerPack>(pThreadStateBoosting, pFeatureGroup);
         } else {
            return BitPack<TLoss, cCompilerScores, GetNextBitPack(cCompilerPack)>::ApplyTraining(pLoss, pThreadStateBoosting, pFeatureGroup);
         }
      }
      INLINE_ALWAYS static ErrorEbmType ApplyValidation(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) {
         if(cCompilerPack == pFeatureGroup->GetBitPack()) {
            return pLoss->BitPackPostApplyValidation<TLoss, cCompilerScores, cCompilerPack>(pThreadStateBoosting, pFeatureGroup, pMetricOut);
         } else {
            return BitPack<TLoss, cCompilerScores, GetNextBitPack(cCompilerPack)>::ApplyValidation(pLoss, pThreadStateBoosting, pFeatureGroup, pMetricOut);
         }
      }
   };
   template<typename TLoss, ptrdiff_t cCompilerScores>
   struct BitPack<TLoss, cCompilerScores, k_cItemsPerBitPackLast> final {
      INLINE_ALWAYS static ErrorEbmType ApplyTraining(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) {
         return pLoss->BitPackPostApplyTraining<TLoss, cCompilerScores, k_cItemsPerBitPackLast>(pThreadStateBoosting, pFeatureGroup);
      }
      INLINE_ALWAYS static ErrorEbmType ApplyValidation(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) {
         return pLoss->BitPackPostApplyValidation<TLoss, cCompilerScores, k_cItemsPerBitPackLast>(pThreadStateBoosting, pFeatureGroup, pMetricOut);
      }
   };
   template<typename TLoss, ptrdiff_t cCompilerScores, ptrdiff_t cCompilerPack>
   INLINE_RELEASE_TEMPLATED ErrorEbmType BitPackPostApplyTraining(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) const {
      const TLoss * const pTLoss = static_cast<const TLoss *>(this);
      return pTLoss->template ApplyTrainingTemplated<cCompilerScores, cCompilerPack>(pThreadStateBoosting, pFeatureGroup);
   }
   template<typename TLoss, ptrdiff_t cCompilerScores, ptrdiff_t cCompilerPack>
   INLINE_RELEASE_TEMPLATED ErrorEbmType BitPackPostApplyValidation(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) const {
      const TLoss * const pTLoss = static_cast<const TLoss *>(this);
      return pTLoss->template ApplyValidationTemplated<cCompilerScores, cCompilerPack>(pThreadStateBoosting, pFeatureGroup, pMetricOut);
   }

   template<bool bHessian>
   INLINE_ALWAYS static void ApplyHessian();

   template<>
   INLINE_ALWAYS static void ApplyHessian<true>() {
   }

   template<>
   INLINE_ALWAYS static void ApplyHessian<false>() {
   }

   template<typename TLoss, ptrdiff_t cCompilerScores, ptrdiff_t cCompilerPack, bool bHessian>
   struct Shared final {
      INLINE_ALWAYS static ErrorEbmType ApplyTraining(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) {
         UNUSED(pLoss);
         UNUSED(pThreadStateBoosting);
         UNUSED(pFeatureGroup);

         ApplyHessian<bHessian>(); // TODO: use this

         return Error_None;
      }
      INLINE_ALWAYS static ErrorEbmType ApplyValidation(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) {
         UNUSED(pLoss);
         UNUSED(pThreadStateBoosting);
         UNUSED(pFeatureGroup);
         UNUSED(pMetricOut);

         ApplyHessian<bHessian>(); // TODO: use this

         return Error_None;
      }
   };
   template<typename TLoss, ptrdiff_t cCompilerScores, bool bHessian>
   struct Shared<TLoss, cCompilerScores, k_cItemsPerBitPackNone, bHessian> final {
      INLINE_ALWAYS static ErrorEbmType ApplyTraining(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) {
         UNUSED(pLoss);
         UNUSED(pThreadStateBoosting);
         UNUSED(pFeatureGroup);

         ApplyHessian<bHessian>(); // TODO: use this

         return Error_None;
      }
      INLINE_ALWAYS static ErrorEbmType ApplyValidation(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) {
         UNUSED(pLoss);
         UNUSED(pThreadStateBoosting);
         UNUSED(pFeatureGroup);
         UNUSED(pMetricOut);

         ApplyHessian<bHessian>(); // TODO: use this

         return Error_None;
      }
   };
   template<typename TLoss, ptrdiff_t cCompilerPack, bool bHessian>
   struct Shared <TLoss, k_oneScore, cCompilerPack, bHessian> final {
      INLINE_ALWAYS static ErrorEbmType ApplyTraining(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) {
         UNUSED(pLoss);
         UNUSED(pThreadStateBoosting);
         UNUSED(pFeatureGroup);

         ApplyHessian<bHessian>(); // TODO: use this

         return Error_None;
      }
      INLINE_ALWAYS static ErrorEbmType ApplyValidation(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) {
         UNUSED(pLoss);
         UNUSED(pThreadStateBoosting);
         UNUSED(pFeatureGroup);
         UNUSED(pMetricOut);

         ApplyHessian<bHessian>(); // TODO: use this

         return Error_None;
      }
   };
   template<typename TLoss, bool bHessian>
   struct Shared<TLoss, k_oneScore, k_cItemsPerBitPackNone, bHessian> final {
      INLINE_ALWAYS static ErrorEbmType ApplyTraining(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) {
         UNUSED(pLoss);
         UNUSED(pThreadStateBoosting);
         UNUSED(pFeatureGroup);

         ApplyHessian<bHessian>(); // TODO: use this

         return Error_None;
      }
      INLINE_ALWAYS static ErrorEbmType ApplyValidation(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) {
         UNUSED(pLoss);
         UNUSED(pThreadStateBoosting);
         UNUSED(pFeatureGroup);
         UNUSED(pMetricOut);

         ApplyHessian<bHessian>(); // TODO: use this

         return Error_None;
      }
   };

   template<class TLoss>
   struct HasCalculateHessianFunctionInternal {
      struct TrueStruct { 
      };
      struct FalseStruct {
      };

      template<class TCheck>
      static TrueStruct NotInvokedCheck(TCheck const * pCheck,
         typename std::enable_if<std::is_same<FloatEbmType,
         decltype(pCheck->CalculateHessian<FloatEbmType>(FloatEbmType { 0 }, FloatEbmType { 0 }))>::value>::type * = 
         nullptr);
      static FalseStruct NotInvokedCheck(...);
      static constexpr bool value = std::is_same<TrueStruct, decltype(HasCalculateHessianFunctionInternal::NotInvokedCheck(static_cast<typename std::remove_reference<TLoss>::type *>(nullptr)))>::value;
   };

protected:

   template<typename TLoss>
   constexpr static bool HasCalculateHessianFunction() {
      // use SFINAE to find out if our Loss class has the function CalculateHessian with the correct signature
      return HasCalculateHessianFunctionInternal<TLoss>::value;
   }

   template<typename TLoss>
   constexpr static bool IsEdgeLoss() { 
      return
         std::is_base_of<LossBinary, TLoss>::value ||
         std::is_base_of<LossMulticlass, TLoss>::value ||
         std::is_base_of<LossRegression, TLoss>::value ||
         std::is_base_of<LossMultitaskBinary, TLoss>::value ||
         std::is_base_of<LossMultitaskMulticlass, TLoss>::value ||
         std::is_base_of<LossMultitaskRegression, TLoss>::value;
   }


   template<typename TLoss, ptrdiff_t cCompilerScores, ptrdiff_t cCompilerPack>
   INLINE_RELEASE_TEMPLATED ErrorEbmType SharedApplyTraining(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) const {
      static_assert(IsEdgeLoss<TLoss>(), "TLoss must inherit from one of the children of the Loss class");
      return Shared<TLoss, cCompilerScores, cCompilerPack, HasCalculateHessianFunction<TLoss>()>::ApplyTraining(this, pThreadStateBoosting, pFeatureGroup);
   }
   template<typename TLoss, ptrdiff_t cCompilerScores, ptrdiff_t cCompilerPack>
   INLINE_RELEASE_TEMPLATED ErrorEbmType SharedApplyValidation(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) const {
      static_assert(IsEdgeLoss<TLoss>(), "TLoss must inherit from one of the children of the Loss class");
      return Shared<TLoss, cCompilerScores, cCompilerPack, HasCalculateHessianFunction<TLoss>()>::ApplyValidation(this, pThreadStateBoosting, pFeatureGroup, pMetricOut);
   }


   template<typename TLoss>
   INLINE_RELEASE_TEMPLATED ErrorEbmType LossApplyTraining(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) const {
      static_assert(IsEdgeLoss<TLoss>(), "TLoss must inherit from one of the children of the Loss class");
      return CountScoresPreApplyTraining<TLoss>(pThreadStateBoosting, pFeatureGroup);
   }
   template<typename TLoss>
   INLINE_RELEASE_TEMPLATED ErrorEbmType LossApplyValidation(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) const {
      static_assert(IsEdgeLoss<TLoss>(), "TLoss must inherit from one of the children of the Loss class");
      return CountScoresPreApplyValidation<TLoss>(pThreadStateBoosting, pFeatureGroup, pMetricOut);
   }


   Loss() = default;

public:

   // TODO: we should wrap all our calls to the potentially re-written client code with try/catch blocks since we don't know if it throws exceptions

   // TODO: we need to actually use this now..
   virtual FloatEbmType GetUpdateMultiple() const;
   virtual bool IsSuperSuperSpecialLossWhereTargetNotNeededOnlyMseLossQualifies() const;

   virtual bool HasCalculateHessianFunction() const = 0;
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

class LossSingletask : public Loss {
protected:
   LossSingletask() = default;
};

class LossBinary : public LossSingletask {
protected:
   LossBinary() = default;
public:
   static constexpr bool IsMultiScore = false;
};

class LossMulticlass : public LossSingletask {
protected:
   LossMulticlass() = default;
public:
   static constexpr bool IsMultiScore = true;
};

class LossRegression : public LossSingletask {
protected:
   LossRegression() = default;
public:
   static constexpr bool IsMultiScore = false;
};

class LossMultitask : public Loss {
protected:
   LossMultitask() = default;
public:
   static constexpr bool IsMultiScore = true;
};

class LossMultitaskBinary : public LossMultitask {
protected:
   LossMultitaskBinary() = default;
};

class LossMultitaskMulticlass : public LossMultitask {
protected:
   LossMultitaskMulticlass() = default;
};

class LossMultitaskRegression : public LossMultitask {
protected:
   LossMultitaskRegression() = default;
};


#define LOSS_DEFAULT_MECHANICS_PUT_AT_END_OF_CLASS \
   public: \
      bool HasCalculateHessianFunction() const override { \
         return Loss::HasCalculateHessianFunction<std::remove_pointer<decltype(this)>::type>(); \
      } \
      template<ptrdiff_t cCompilerScores, ptrdiff_t cCompilerPack> \
      ErrorEbmType ApplyTrainingTemplated(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) const { \
         return Loss::SharedApplyTraining<std::remove_pointer<decltype(this)>::type, cCompilerScores, cCompilerPack>(pThreadStateBoosting, pFeatureGroup); \
      } \
      template<ptrdiff_t cCompilerScores, ptrdiff_t cCompilerPack> \
      ErrorEbmType ApplyValidationTemplated(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) const { \
         return Loss::SharedApplyValidation<std::remove_pointer<decltype(this)>::type, cCompilerScores, cCompilerPack>(pThreadStateBoosting, pFeatureGroup, pMetricOut); \
      } \
      ErrorEbmType ApplyTraining(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) const override { \
         return Loss::LossApplyTraining<std::remove_pointer<decltype(this)>::type>(pThreadStateBoosting, pFeatureGroup); \
      } \
      ErrorEbmType ApplyValidation(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) const override { \
         return Loss::LossApplyValidation<std::remove_pointer<decltype(this)>::type>(pThreadStateBoosting, pFeatureGroup, pMetricOut); \
      }

#endif // LOSS_H
