// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// !!! NOTE: To add a new loss/objective function in C++, follow the steps listed at the top of the "Loss.cpp" file !!!

#ifndef LOSS_H
#define LOSS_H

#include <stddef.h> // size_t, ptrdiff_t
#include <memory> // shared_ptr, unique_ptr

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

class ApplyTrainingData final {
   ThreadStateBoosting * const m_pThreadStateBoosting;
   const FeatureGroup * const m_pFeatureGroup;

public:

   INLINE_ALWAYS ThreadStateBoosting * GetThreadStateBoosting() const noexcept {
      return m_pThreadStateBoosting;
   }
   INLINE_ALWAYS const FeatureGroup * GetFeatureGroup() const noexcept {
      return m_pFeatureGroup;
   }

   INLINE_ALWAYS ApplyTrainingData(
      ThreadStateBoosting * const pThreadStateBoosting, 
      const FeatureGroup * const pFeatureGroup
   ) noexcept :
      m_pThreadStateBoosting(pThreadStateBoosting),
      m_pFeatureGroup(pFeatureGroup) {
   }
};

class ApplyValidationData final {
   ThreadStateBoosting * const m_pThreadStateBoosting;
   const FeatureGroup * const m_pFeatureGroup;
   FloatEbmType m_metric;

public:

   INLINE_ALWAYS ThreadStateBoosting * GetThreadStateBoosting() const noexcept {
      return m_pThreadStateBoosting;
   }
   INLINE_ALWAYS const FeatureGroup * GetFeatureGroup() const noexcept {
      return m_pFeatureGroup;
   }
   INLINE_ALWAYS FloatEbmType GetMetric() const noexcept {
      return m_metric;
   }
   INLINE_ALWAYS void SetMetric(const FloatEbmType metric) noexcept {
      m_metric = metric;
   }

   INLINE_ALWAYS ApplyValidationData(
      ThreadStateBoosting * const pThreadStateBoosting,
      const FeatureGroup * const pFeatureGroup
   ) noexcept :
      m_pThreadStateBoosting(pThreadStateBoosting),
      m_pFeatureGroup(pFeatureGroup),
      m_metric(FloatEbmType { 0 }) {
   }
};

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
   INLINE_RELEASE_TEMPLATED ErrorEbmType CountScoresPreApplyTraining(ApplyTrainingData & data) const {
      return CountScores<TLoss, GetInitialCountScores<TLoss>()>::ApplyTraining(this, data);
   }
   template<typename TLoss>
   INLINE_RELEASE_TEMPLATED ErrorEbmType CountScoresPreApplyValidation(ApplyValidationData & data) const {
      return CountScores<TLoss, GetInitialCountScores<TLoss>()>::ApplyValidation(this, data);
   }
   template<typename TLoss, ptrdiff_t cCompilerScores>
   struct CountScores final {
      INLINE_ALWAYS static ErrorEbmType ApplyTraining(const Loss * const pLoss, ApplyTrainingData & data) {
         if(cCompilerScores == data.GetThreadStateBoosting()->GetBooster()->GetRuntimeLearningTypeOrCountTargetClasses()) {
            return pLoss->BitPackPreApplyTraining<TLoss, cCompilerScores>(data);
         } else {
            return CountScores<TLoss, k_cCompilerOptimizedTargetClassesMax == cCompilerScores ? k_dynamicClassification : cCompilerScores + 1>::ApplyTraining(pLoss, data);
         }
      }
      INLINE_ALWAYS static ErrorEbmType ApplyValidation(const Loss * const pLoss, ApplyValidationData & data) {
         if(cCompilerScores == data.GetThreadStateBoosting()->GetBooster()->GetRuntimeLearningTypeOrCountTargetClasses()) {
            return pLoss->BitPackPreApplyValidation<TLoss, cCompilerScores>(data);
         } else {
            return CountScores<TLoss, k_cCompilerOptimizedTargetClassesMax == cCompilerScores ? k_dynamicClassification : cCompilerScores + 1>::ApplyValidation(pLoss, data);
         }
      }
   };
   template<typename TLoss>
   struct CountScores<TLoss, k_dynamicClassification> final {
      INLINE_ALWAYS static ErrorEbmType ApplyTraining(const Loss * const pLoss, ApplyTrainingData & data) {
         return pLoss->BitPackPreApplyTraining<TLoss, k_dynamicClassification>(data);
      }
      INLINE_ALWAYS static ErrorEbmType ApplyValidation(const Loss * const pLoss, ApplyValidationData & data) {
         return pLoss->BitPackPreApplyValidation<TLoss, k_dynamicClassification>(data);
      }
   };
   template<typename TLoss>
   struct CountScores<TLoss, k_oneScore> final {
      INLINE_ALWAYS static ErrorEbmType ApplyTraining(const Loss * const pLoss, ApplyTrainingData & data) {
         return pLoss->BitPackPreApplyTraining<TLoss, k_oneScore>(data);
      }
      INLINE_ALWAYS static ErrorEbmType ApplyValidation(const Loss * const pLoss, ApplyValidationData & data) {
         return pLoss->BitPackPreApplyValidation<TLoss, k_oneScore>(data);
      }
   };



   template<typename TLoss, ptrdiff_t cCompilerScores>
   INLINE_RELEASE_TEMPLATED ErrorEbmType BitPackPreApplyTraining(ApplyTrainingData & data) const {
      if(k_cItemsPerBitPackNone == data.GetFeatureGroup()->GetBitPack()) {
         return BitPackPostApplyTraining<TLoss, cCompilerScores, k_cItemsPerBitPackNone>(data);
      } else {
         return BitPack<TLoss, cCompilerScores, k_cItemsPerBitPackMax2>::ApplyTraining(this, data);
      }
   }
   template<typename TLoss, ptrdiff_t cCompilerScores>
   INLINE_RELEASE_TEMPLATED ErrorEbmType BitPackPreApplyValidation(ApplyValidationData & data) const {
      if(k_cItemsPerBitPackNone == data.GetFeatureGroup()->GetBitPack()) {
         return BitPackPostApplyValidation<TLoss, cCompilerScores, k_cItemsPerBitPackNone>(data);
      } else {
         return BitPack<TLoss, cCompilerScores, k_cItemsPerBitPackMax2>::ApplyValidation(this, data);
      }
   }
   template<typename TLoss, ptrdiff_t cCompilerScores, ptrdiff_t cCompilerPack>
   struct BitPack final {
      INLINE_ALWAYS static ErrorEbmType ApplyTraining(const Loss * const pLoss, ApplyTrainingData & data) {
         if(cCompilerPack == data.GetFeatureGroup()->GetBitPack()) {
            return pLoss->BitPackPostApplyTraining<TLoss, cCompilerScores, cCompilerPack>(data);
         } else {
            return BitPack<TLoss, cCompilerScores, GetNextBitPack(cCompilerPack)>::ApplyTraining(pLoss, data);
         }
      }
      INLINE_ALWAYS static ErrorEbmType ApplyValidation(const Loss * const pLoss, ApplyValidationData & data) {
         if(cCompilerPack == data.GetFeatureGroup()->GetBitPack()) {
            return pLoss->BitPackPostApplyValidation<TLoss, cCompilerScores, cCompilerPack>(data);
         } else {
            return BitPack<TLoss, cCompilerScores, GetNextBitPack(cCompilerPack)>::ApplyValidation(pLoss, data);
         }
      }
   };
   template<typename TLoss, ptrdiff_t cCompilerScores>
   struct BitPack<TLoss, cCompilerScores, k_cItemsPerBitPackLast> final {
      INLINE_ALWAYS static ErrorEbmType ApplyTraining(const Loss * const pLoss, ApplyTrainingData & data) {
         return pLoss->BitPackPostApplyTraining<TLoss, cCompilerScores, k_cItemsPerBitPackLast>(data);
      }
      INLINE_ALWAYS static ErrorEbmType ApplyValidation(const Loss * const pLoss, ApplyValidationData & data) {
         return pLoss->BitPackPostApplyValidation<TLoss, cCompilerScores, k_cItemsPerBitPackLast>(data);
      }
   };
   template<typename TLoss, ptrdiff_t cCompilerScores, ptrdiff_t cCompilerPack>
   INLINE_RELEASE_TEMPLATED ErrorEbmType BitPackPostApplyTraining(ApplyTrainingData & data) const {
      const TLoss * const pTLoss = static_cast<const TLoss *>(this);
      return pTLoss->template ApplyTrainingTemplated<cCompilerScores, cCompilerPack>(data);
   }
   template<typename TLoss, ptrdiff_t cCompilerScores, ptrdiff_t cCompilerPack>
   INLINE_RELEASE_TEMPLATED ErrorEbmType BitPackPostApplyValidation(ApplyValidationData & data) const {
      const TLoss * const pTLoss = static_cast<const TLoss *>(this);
      return pTLoss->template ApplyValidationTemplated<cCompilerScores, cCompilerPack>(data);
   }

   template<typename TLoss, bool bHessian>
   struct ApplyHessian;

   template<typename TLoss>
   struct ApplyHessian<TLoss, true> final {
      INLINE_ALWAYS static void Func() {
      }
   };

   template<typename TLoss>
   struct ApplyHessian<TLoss, false> final {
      INLINE_ALWAYS static void Func() {
      }
   };

   template<typename TLoss, ptrdiff_t cCompilerScores, ptrdiff_t cCompilerPack, bool bHessian>
   struct Shared final {
      INLINE_ALWAYS static ErrorEbmType ApplyTraining(const Loss * const pLoss, ApplyTrainingData & data) {
         UNUSED(pLoss);
         UNUSED(data);

         ApplyHessian<TLoss, bHessian>::Func(); // TODO: use this

         return Error_None;
      }
      INLINE_ALWAYS static ErrorEbmType ApplyValidation(const Loss * const pLoss, ApplyValidationData & data) {
         UNUSED(pLoss);
         UNUSED(data);

         ApplyHessian<TLoss, bHessian>::Func(); // TODO: use this

         return Error_None;
      }
   };
   template<typename TLoss, ptrdiff_t cCompilerScores, bool bHessian>
   struct Shared<TLoss, cCompilerScores, k_cItemsPerBitPackNone, bHessian> final {
      INLINE_ALWAYS static ErrorEbmType ApplyTraining(const Loss * const pLoss, ApplyTrainingData & data) {
         UNUSED(pLoss);
         UNUSED(data);

         ApplyHessian<TLoss, bHessian>::Func(); // TODO: use this

         return Error_None;
      }
      INLINE_ALWAYS static ErrorEbmType ApplyValidation(const Loss * const pLoss, ApplyValidationData & data) {
         UNUSED(pLoss);
         UNUSED(data);

         ApplyHessian<TLoss, bHessian>::Func(); // TODO: use this

         return Error_None;
      }
   };
   template<typename TLoss, ptrdiff_t cCompilerPack, bool bHessian>
   struct Shared <TLoss, k_oneScore, cCompilerPack, bHessian> final {
      INLINE_ALWAYS static ErrorEbmType ApplyTraining(const Loss * const pLoss, ApplyTrainingData & data) {
         UNUSED(pLoss);
         UNUSED(data);

         ApplyHessian<TLoss, bHessian>::Func(); // TODO: use this

         return Error_None;
      }
      INLINE_ALWAYS static ErrorEbmType ApplyValidation(const Loss * const pLoss, ApplyValidationData & data) {
         UNUSED(pLoss);
         UNUSED(data);

         ApplyHessian<TLoss, bHessian>::Func(); // TODO: use this

         return Error_None;
      }
   };
   template<typename TLoss, bool bHessian>
   struct Shared<TLoss, k_oneScore, k_cItemsPerBitPackNone, bHessian> final {
      INLINE_ALWAYS static ErrorEbmType ApplyTraining(const Loss * const pLoss, ApplyTrainingData & data) {
         UNUSED(pLoss);
         UNUSED(data);

         ApplyHessian<TLoss, bHessian>::Func(); // TODO: use this

         return Error_None;
      }
      INLINE_ALWAYS static ErrorEbmType ApplyValidation(const Loss * const pLoss, ApplyValidationData & data) {
         UNUSED(pLoss);
         UNUSED(data);

         ApplyHessian<TLoss, bHessian>::Func(); // TODO: use this

         return Error_None;
      }
   };

   template<class TLoss>
   struct HasCalculateHessianFunctionInternal {
      // use SFINAE to find out if the target class has the function with the correct signature
      struct TrueStruct {
      };
      struct FalseStruct {
      };

      // under SFINAE, this first version of the NotInvokedCheck function disappears if it fails due to the lack of 
      // the CalculateHessian function with the correct parameters.  It would be better to use a more general
      // version that only looks for the function name, but I don't know if that's possible without taking a pointer
      // to the function, which we don't want to do since our function is meant to be 100% inline, and taking
      // a pointer might create a second non-inlined copy of the function with some compilers.  I think
      // burrying the inline function inside a decltype will prevent it's materialization.
      template<class TCheck>
      static TrueStruct NotInvokedCheck(TCheck const * pCheck,
         typename std::enable_if<
         std::is_same<FloatEbmType, decltype(pCheck->CalculateHessian(FloatEbmType { 0 }, FloatEbmType { 0 }))>::value
         >::type * = nullptr);
      static FalseStruct NotInvokedCheck(...);
      static constexpr bool value = std::is_same<TrueStruct, 
         decltype(HasCalculateHessianFunctionInternal::NotInvokedCheck(static_cast<typename std::remove_reference<TLoss>::type *>(nullptr)))
         >::value;
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
   INLINE_RELEASE_TEMPLATED ErrorEbmType SharedApplyTraining(ApplyTrainingData & data) const {
      static_assert(IsEdgeLoss<TLoss>(), "TLoss must inherit from one of the children of the Loss class");
      return Shared<TLoss, cCompilerScores, cCompilerPack, HasCalculateHessianFunction<TLoss>()>::ApplyTraining(this, data);
   }
   template<typename TLoss, ptrdiff_t cCompilerScores, ptrdiff_t cCompilerPack>
   INLINE_RELEASE_TEMPLATED ErrorEbmType SharedApplyValidation(ApplyValidationData & data) const {
      static_assert(IsEdgeLoss<TLoss>(), "TLoss must inherit from one of the children of the Loss class");
      return Shared<TLoss, cCompilerScores, cCompilerPack, HasCalculateHessianFunction<TLoss>()>::ApplyValidation(this, data);
   }


   template<typename TLoss>
   INLINE_RELEASE_TEMPLATED ErrorEbmType LossApplyTraining(ApplyTrainingData & data) const {
      static_assert(IsEdgeLoss<TLoss>(), "TLoss must inherit from one of the children of the Loss class");
      return CountScoresPreApplyTraining<TLoss>(data);
   }
   template<typename TLoss>
   INLINE_RELEASE_TEMPLATED ErrorEbmType LossApplyValidation(ApplyValidationData & data) const {
      static_assert(IsEdgeLoss<TLoss>(), "TLoss must inherit from one of the children of the Loss class");
      return CountScoresPreApplyValidation<TLoss>(data);
   }


   Loss() = default;

public:

   // TODO: we should wrap all our calls to the potentially re-written client code with try/catch blocks since we don't know if it throws exceptions

   // TODO: we need to actually use this now..
   virtual FloatEbmType GetUpdateMultiple() const;
   virtual bool IsSuperSuperSpecialLossWhereTargetNotNeededOnlyMseLossQualifies() const;

   virtual bool LossHasHessian() const = 0;
   virtual ErrorEbmType ApplyTraining(ApplyTrainingData & data) const = 0;
   virtual ErrorEbmType ApplyValidation(ApplyValidationData & data) const = 0;

   static ErrorEbmType CreateLoss(
      const char * const sLoss,
      const Config * const pConfig,
      const Loss ** const ppLoss
   ) noexcept;
};

// TODO: include ranking
//
// We use the following terminology:
// Target      : the thing we're trying to predict.  For classification this is the label.  For regression this 
//               is what we're predicting directly.  Target and Output seem to be used interchangeably in other 
//               packages.  We choose Target here.
// Score       : the values we use to generate predictions.  For classification these are logits.  For regression these
//               are the predictions themselves.  For multiclass there are N scores per target when there are N classes.
//               For multiclass you could eliminate one score to get N-1 scores, but we don't use that trick in this 
//               package yet.
// Prediction  : the prediction of the model.  We output scores in our model and generate predictions from them.
//               For multiclass the scores are the logits, and the predictions would be the outputs of softmax.
//               We have N scores per target for an N class multiclass problem.
// Binary      : binary classification.  Target is 0 or 1
// Multiclass  : multiclass classification.  Target is 0, 1, 2, ... 
// Regression  : regression
// Multioutput : a model that can predict multiple different things.  A single model could predict binary, 
//               multiclass, regression, etc. different targets.
// Multitask   : A slightly more restricted form of multioutput where training jointly optimizes the targets.
//               The different targets can still be of different types (binary, multiclass, regression, etc), but
//               importantly they share a single loss function.  In C++ we deal only with multitask since otherwise 
//               it would make more sense to train the targets separately.  In higher level languages the models can 
//               either be Multitask or Multioutput depending on how they were generated.
// Multilabel  : A more restricted version of multitask where the tasks are all binary classification.  We use
//               the term MultitaskBinary* here since it fits better into our ontology.
// 
// The most general loss function that we could handle in C++ would be to take a custom loss function that jointly 
// optimizes a multitask problem that contains regression, binary, and multiclass tasks.  This would be: 
// "LossMultitaskCustom"

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


#define LOSS_CLASS_BOILERPLATE_PUT_AT_END_OF_CLASS \
   LOSS_CLASS_TEMPLATE_BOILERPLATE_PUT_AT_END_OF_CLASS \
   LOSS_CLASS_VIRTUAL_BOILERPLATE_PUT_AT_END_OF_CLASS

#define LOSS_CLASS_TEMPLATE_BOILERPLATE_PUT_AT_END_OF_CLASS \
   public: \
      bool LossHasHessian() const override { \
         return Loss::HasCalculateHessianFunction<std::remove_pointer<decltype(this)>::type>(); \
      } \
      template<ptrdiff_t cCompilerScores, ptrdiff_t cCompilerPack> \
      ErrorEbmType ApplyTrainingTemplated(ApplyTrainingData & data) const { \
         return Loss::SharedApplyTraining<std::remove_pointer<decltype(this)>::type, cCompilerScores, cCompilerPack>(data); \
      } \
      template<ptrdiff_t cCompilerScores, ptrdiff_t cCompilerPack> \
      ErrorEbmType ApplyValidationTemplated(ApplyValidationData & data) const { \
         return Loss::SharedApplyValidation<std::remove_pointer<decltype(this)>::type, cCompilerScores, cCompilerPack>(data); \
      }

#define LOSS_CLASS_VIRTUAL_BOILERPLATE_PUT_AT_END_OF_CLASS \
   public: \
      ErrorEbmType ApplyTraining(ApplyTrainingData & data) const override { \
         return Loss::LossApplyTraining<std::remove_pointer<decltype(this)>::type>(data); \
      } \
      ErrorEbmType ApplyValidation(ApplyValidationData & data) const override { \
         return Loss::LossApplyValidation<std::remove_pointer<decltype(this)>::type>(data); \
      }

#endif // LOSS_H
