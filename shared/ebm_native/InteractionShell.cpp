// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "ebm_internal.hpp"

#include "InteractionCore.hpp"
#include "InteractionShell.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

void InteractionShell::Free(InteractionShell * const pInteractionShell) {
   LOG_0(TraceLevelInfo, "Entered InteractionShell::Free");

   if(nullptr != pInteractionShell) {
      free(pInteractionShell->m_aThreadByteBuffer1);
      InteractionCore::Free(pInteractionShell->m_pInteractionCore);
      
      // before we free our memory, indicate it was freed so if our higher level language attempts to use it we have
      // a chance to detect the error
      pInteractionShell->m_handleVerification = k_handleVerificationFreed;
      free(pInteractionShell);
   }

   LOG_0(TraceLevelInfo, "Exited InteractionShell::Free");
}

InteractionShell * InteractionShell::Create() {
   LOG_0(TraceLevelInfo, "Entered InteractionShell::Create");

   InteractionShell * const pNew = EbmMalloc<InteractionShell>();
   if(nullptr != pNew) {
      pNew->InitializeZero();
   }

   LOG_0(TraceLevelInfo, "Exited InteractionShell::Create");

   return pNew;
}

HistogramBucketBase * InteractionShell::GetHistogramBucketBase(size_t cBytesRequired) {
   HistogramBucketBase * aBuffer = m_aThreadByteBuffer1;
   if(UNLIKELY(m_cThreadByteBufferCapacity1 < cBytesRequired)) {
      cBytesRequired <<= 1;
      m_cThreadByteBufferCapacity1 = cBytesRequired;
      LOG_N(TraceLevelInfo, "Growing InteractionShell::ThreadByteBuffer1 to %zu", cBytesRequired);

      free(aBuffer);
      aBuffer = static_cast<HistogramBucketBase *>(EbmMalloc<void>(cBytesRequired));
      m_aThreadByteBuffer1 = aBuffer; // store it before checking it incase it's null so that we don't free old memory
      if(nullptr == aBuffer) {
         LOG_0(TraceLevelWarning, "WARNING InteractionShell::GetHistogramBucketBase OutOfMemory");
      }
   }
   return aBuffer;
}

// a*PredictorScores = logOdds for binary classification
// a*PredictorScores = logWeights for multiclass classification
// a*PredictorScores = predictedValue for regression
static ErrorEbmType CreateInteractionDetector(
   const IntEbmType countFeatures,
   const BoolEbmType * const aFeaturesCategorical,
   const IntEbmType * const aFeaturesBinCount,
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
   const IntEbmType countSamples,
   const void * const targets,
   const IntEbmType * const binnedData,
   const FloatEbmType * const aWeights,
   const FloatEbmType * const predictorScores,
   const FloatEbmType * const optionalTempParams,
   InteractionHandle * interactionHandleOut
) {
   // TODO : give CreateInteractionDetector the same calling parameter order as CreateClassificationInteractionDetector

   EBM_ASSERT(nullptr != interactionHandleOut);
   EBM_ASSERT(nullptr == *interactionHandleOut);

   if(countFeatures < 0) {
      LOG_0(TraceLevelError, "ERROR CreateInteractionDetector countFeatures must be positive");
      return Error_IllegalParamValue;
   }
   if(0 != countFeatures && nullptr == aFeaturesCategorical) {
      // TODO: in the future maybe accept null aFeaturesCategorical and assume there are no missing values
      LOG_0(TraceLevelError, "ERROR CreateInteractionDetector aFeaturesCategorical cannot be nullptr if 0 < countFeatures");
      return Error_IllegalParamValue;
   }
   if(0 != countFeatures && nullptr == aFeaturesBinCount) {
      LOG_0(TraceLevelError, "ERROR CreateInteractionDetector aFeaturesBinCount cannot be nullptr if 0 < countFeatures");
      return Error_IllegalParamValue;
   }
   if(countSamples < 0) {
      LOG_0(TraceLevelError, "ERROR CreateInteractionDetector countSamples must be positive");
      return Error_IllegalParamValue;
   }
   if(0 != countSamples && nullptr == targets) {
      LOG_0(TraceLevelError, "ERROR CreateInteractionDetector targets cannot be nullptr if 0 < countSamples");
      return Error_IllegalParamValue;
   }
   if(0 != countSamples && 0 != countFeatures && nullptr == binnedData) {
      LOG_0(TraceLevelError, "ERROR CreateInteractionDetector binnedData cannot be nullptr if 0 < countSamples AND 0 < countFeatures");
      return Error_IllegalParamValue;
   }
   if(0 != countSamples && nullptr == predictorScores) {
      LOG_0(TraceLevelError, "ERROR CreateInteractionDetector predictorScores cannot be nullptr if 0 < countSamples");
      return Error_IllegalParamValue;
   }
   if(IsConvertError<size_t>(countFeatures)) {
      LOG_0(TraceLevelError, "ERROR CreateInteractionDetector IsConvertError<size_t>(countFeatures)");
      return Error_IllegalParamValue;
   }
   if(IsConvertError<size_t>(countSamples)) {
      LOG_0(TraceLevelError, "ERROR CreateInteractionDetector IsConvertError<size_t>(countSamples)");
      return Error_IllegalParamValue;
   }

   size_t cFeatures = static_cast<size_t>(countFeatures);
   size_t cSamples = static_cast<size_t>(countSamples);

   InteractionShell * const pInteractionShell = InteractionShell::Create();
   if(UNLIKELY(nullptr == pInteractionShell)) {
      LOG_0(TraceLevelWarning, "WARNING CreateInteractionDetector nullptr == pInteractionShell");
      return Error_OutOfMemory;
   }

   const ErrorEbmType error = InteractionCore::Create(
      pInteractionShell,
      runtimeLearningTypeOrCountTargetClasses,
      cFeatures,
      optionalTempParams,
      aFeaturesCategorical,
      aFeaturesBinCount,
      cSamples,
      targets,
      binnedData,
      aWeights,
      predictorScores
   );
   if(Error_None != error) {
      InteractionShell::Free(pInteractionShell);
      LOG_0(TraceLevelWarning, "WARNING CreateInteractionDetector nullptr == pInteractionCore");
      return Error_OutOfMemory;
   }

   *interactionHandleOut = pInteractionShell->GetHandle();
   return Error_None;
}

EBM_NATIVE_IMPORT_EXPORT_BODY ErrorEbmType EBM_NATIVE_CALLING_CONVENTION CreateClassificationInteractionDetector(
   IntEbmType countTargetClasses,
   IntEbmType countFeatures,
   const BoolEbmType * featuresCategorical,
   const IntEbmType * featuresBinCount,
   IntEbmType countSamples,
   const IntEbmType * binnedData,
   const IntEbmType * targets,
   const FloatEbmType * weights,
   const FloatEbmType * predictorScores,
   const FloatEbmType * optionalTempParams,
   InteractionHandle * interactionHandleOut
) {
   LOG_N(
      TraceLevelInfo,
      "Entered CreateClassificationInteractionDetector: "
      "countTargetClasses=%" IntEbmTypePrintf ", "
      "countFeatures=%" IntEbmTypePrintf ", "
      "featuresCategorical=%p, "
      "featuresBinCount=%p, "
      "countSamples=%" IntEbmTypePrintf ", "
      "binnedData=%p, "
      "targets=%p, "
      "weights=%p, "
      "predictorScores=%p, "
      "optionalTempParams=%p, "
      "interactionHandleOut=%p"
      ,
      countTargetClasses,
      countFeatures,
      static_cast<const void *>(featuresCategorical),
      static_cast<const void *>(featuresBinCount),
      countSamples,
      static_cast<const void *>(binnedData),
      static_cast<const void *>(targets),
      static_cast<const void *>(weights),
      static_cast<const void *>(predictorScores),
      static_cast<const void *>(optionalTempParams),
      static_cast<const void *>(interactionHandleOut)
   );

   if(nullptr == interactionHandleOut) {
      LOG_0(TraceLevelError, "ERROR CreateClassificationInteractionDetector nullptr == interactionHandleOut");
      return Error_IllegalParamValue;
   }
   *interactionHandleOut = nullptr; // set this to nullptr as soon as possible so the caller doesn't attempt to free it

   if(countTargetClasses < 0) {
      LOG_0(TraceLevelError, "ERROR CreateClassificationInteractionDetector countTargetClasses can't be negative");
      return Error_IllegalParamValue;
   }
   if(0 == countTargetClasses && 0 != countSamples) {
      LOG_0(TraceLevelError, "ERROR CreateClassificationInteractionDetector countTargetClasses can't be zero unless there are no samples");
      return Error_IllegalParamValue;
   }
   if(IsConvertError<ptrdiff_t>(countTargetClasses)) {
      LOG_0(TraceLevelWarning, "WARNING CreateClassificationInteractionDetector IsConvertError<ptrdiff_t>(countTargetClasses)");
      // we didn't run out of memory, but we will if we accept this and it's not worth making a new error code
      return Error_OutOfMemory;
   }
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = static_cast<ptrdiff_t>(countTargetClasses);
   const ErrorEbmType error = CreateInteractionDetector(
      countFeatures,
      featuresCategorical,
      featuresBinCount,
      runtimeLearningTypeOrCountTargetClasses,
      countSamples,
      targets,
      binnedData,
      weights,
      predictorScores,
      optionalTempParams,
      interactionHandleOut
   );

   LOG_N(TraceLevelInfo, "Exited CreateClassificationInteractionDetector: "
      "*interactionHandleOut=%p, "
      "return=%" ErrorEbmTypePrintf
      ,
      static_cast<void *>(*interactionHandleOut),
      error
   );

   return error;
}

EBM_NATIVE_IMPORT_EXPORT_BODY ErrorEbmType EBM_NATIVE_CALLING_CONVENTION CreateRegressionInteractionDetector(
   IntEbmType countFeatures,
   const BoolEbmType * featuresCategorical,
   const IntEbmType * featuresBinCount,
   IntEbmType countSamples,
   const IntEbmType * binnedData,
   const FloatEbmType * targets,
   const FloatEbmType * weights,
   const FloatEbmType * predictorScores,
   const FloatEbmType * optionalTempParams,
   InteractionHandle * interactionHandleOut
) {
   LOG_N(TraceLevelInfo, "Entered CreateRegressionInteractionDetector: "
      "countFeatures=%" IntEbmTypePrintf ", "
      "featuresCategorical=%p, "
      "featuresBinCount=%p, "
      "countSamples=%" IntEbmTypePrintf ", "
      "binnedData=%p, "
      "targets=%p, "
      "weights=%p, "
      "predictorScores=%p, "
      "optionalTempParams=%p, "
      "interactionHandleOut=%p"
      ,
      countFeatures,
      static_cast<const void *>(featuresCategorical),
      static_cast<const void *>(featuresBinCount),
      countSamples,
      static_cast<const void *>(binnedData),
      static_cast<const void *>(targets),
      static_cast<const void *>(weights),
      static_cast<const void *>(predictorScores),
      static_cast<const void *>(optionalTempParams),
      static_cast<const void *>(interactionHandleOut)
   );

   if(nullptr == interactionHandleOut) {
      LOG_0(TraceLevelError, "ERROR CreateRegressionInteractionDetector nullptr == interactionHandleOut");
      return Error_IllegalParamValue;
   }
   *interactionHandleOut = nullptr; // set this to nullptr as soon as possible so the caller doesn't attempt to free it

   const ErrorEbmType error = CreateInteractionDetector(
      countFeatures,
      featuresCategorical,
      featuresBinCount,
      k_regression,
      countSamples,
      targets,
      binnedData,
      weights,
      predictorScores,
      optionalTempParams,
      interactionHandleOut
   );

   LOG_N(TraceLevelInfo, "Exited CreateRegressionInteractionDetector: "
      "*interactionHandleOut=%p, "
      "return=%" ErrorEbmTypePrintf
      ,
      static_cast<void *>(*interactionHandleOut),
      error
   );

   return error;
}

EBM_NATIVE_IMPORT_EXPORT_BODY void EBM_NATIVE_CALLING_CONVENTION FreeInteractionDetector(
   InteractionHandle interactionHandle
) {
   LOG_N(TraceLevelInfo, "Entered FreeInteractionDetector: interactionHandle=%p", static_cast<void *>(interactionHandle));

   InteractionShell * const pInteractionShell = InteractionShell::GetInteractionShellFromInteractionHandle(interactionHandle);
   // if the conversion above doesn't work, it'll return null, and our free will not in fact free any memory,
   // but it will not crash. We'll leak memory, but at least we'll log that.

   // it's legal to call free on nullptr, just like for free().  This is checked inside InteractionCore::Free()
   InteractionShell::Free(pInteractionShell);

   LOG_0(TraceLevelInfo, "Exited FreeInteractionDetector");
}

} // DEFINED_ZONE_NAME
