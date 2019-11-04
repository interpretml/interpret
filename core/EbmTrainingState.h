// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <ebm@koch.ninja>

#ifndef EBM_TRAINING_STATE_H
#define EBM_TRAINING_STATE_H

#include <stdlib.h> // malloc, realloc, free
#include <stddef.h> // size_t, ptrdiff_t
#include <limits> // numeric_limits

#include "ebmcore.h"
#include "EbmInternal.h"
// very independent includes
#include "Logging.h" // EBM_ASSERT & LOG
#include "SegmentedTensor.h"
// this depends on TreeNode pointers, but doesn't require the full definition of TreeNode
#include "CachedThreadResources.h"
// feature includes
#include "FeatureCore.h"
// FeatureCombination.h depends on FeatureInternal.h
#include "FeatureCombinationCore.h"
// dataset depends on features
#include "DataSetByFeatureCombination.h"
// samples is somewhat independent from datasets, but relies on an indirect coupling with them
#include "SamplingWithReplacement.h"

union CachedThreadResourcesUnion {
   CachedTrainingThreadResources<false> regression;
   CachedTrainingThreadResources<true> classification;

   EBM_INLINE CachedThreadResourcesUnion(const ptrdiff_t runtimeLearningTypeOrCountTargetClasses) {
      LOG_N(TraceLevelInfo, "Entered CachedThreadResourcesUnion: runtimeLearningTypeOrCountTargetClasses=%td", runtimeLearningTypeOrCountTargetClasses);
      const size_t cVectorLength = GetVectorLengthFlatCore(runtimeLearningTypeOrCountTargetClasses);
      if(IsClassification(runtimeLearningTypeOrCountTargetClasses)) {
         // member classes inside a union requre explicit call to constructor
         new(&classification) CachedTrainingThreadResources<true>(cVectorLength);
      } else {
         EBM_ASSERT(IsRegression(runtimeLearningTypeOrCountTargetClasses));
         // member classes inside a union requre explicit call to constructor
         new(&regression) CachedTrainingThreadResources<false>(cVectorLength);
      }
      LOG_0(TraceLevelInfo, "Exited CachedThreadResourcesUnion");
   }

   EBM_INLINE ~CachedThreadResourcesUnion() {
      // TODO: figure out why this is being called, and if that is bad!
      //LOG_0(TraceLevelError, "ERROR ~CachedThreadResourcesUnion called.  It's union destructors should be called explicitly");

      // we don't have enough information here to delete this object, so we do it from our caller
      // we still need this destructor for a technicality that it might be called
      // if there were an excpetion generated in the initializer list which it is constructed in
      // but we have been careful to ensure that the class we are including it in doesn't thow exceptions in the
      // initializer list
   }
};

class EbmTrainingState {
public:
   const ptrdiff_t m_runtimeLearningTypeOrCountTargetClasses;

   const size_t m_cFeatureCombinations;
   FeatureCombinationCore ** const m_apFeatureCombinations;

   // TODO : can we internalize these so that they are not pointers and are therefore subsumed into our class
   DataSetByFeatureCombination * m_pTrainingSet;
   DataSetByFeatureCombination * m_pValidationSet;

   const size_t m_cSamplingSets;

   SamplingMethod ** m_apSamplingSets;
   SegmentedTensor<ActiveDataType, FractionalDataType> ** m_apCurrentModel;
   SegmentedTensor<ActiveDataType, FractionalDataType> ** m_apBestModel;

   FractionalDataType m_bestModelMetric;

   SegmentedTensor<ActiveDataType, FractionalDataType> * const m_pSmallChangeToModelOverwriteSingleSamplingSet;
   SegmentedTensor<ActiveDataType, FractionalDataType> * const m_pSmallChangeToModelAccumulatedFromSamplingSets;

   const size_t m_cFeatures;
   // TODO : in the future, we can allocate this inside a function so that even the objects inside are const
   FeatureCore * const m_aFeatures;

   CachedThreadResourcesUnion m_cachedThreadResourcesUnion;

   EBM_INLINE EbmTrainingState(const ptrdiff_t runtimeLearningTypeOrCountTargetClasses, const size_t cFeatures, const size_t cFeatureCombinations, const size_t cSamplingSets)
      : m_runtimeLearningTypeOrCountTargetClasses(runtimeLearningTypeOrCountTargetClasses)
      , m_cFeatureCombinations(cFeatureCombinations)
      , m_apFeatureCombinations(0 == cFeatureCombinations ? nullptr : FeatureCombinationCore::AllocateFeatureCombinations(cFeatureCombinations))
      , m_pTrainingSet(nullptr)
      , m_pValidationSet(nullptr)
      , m_cSamplingSets(cSamplingSets)
      , m_apSamplingSets(nullptr)
      , m_apCurrentModel(nullptr)
      , m_apBestModel(nullptr)
      , m_bestModelMetric(FractionalDataType { std::numeric_limits<FractionalDataType>::infinity() })
      , m_pSmallChangeToModelOverwriteSingleSamplingSet(SegmentedTensor<ActiveDataType, FractionalDataType>::Allocate(k_cDimensionsMax, GetVectorLengthFlatCore(runtimeLearningTypeOrCountTargetClasses)))
      , m_pSmallChangeToModelAccumulatedFromSamplingSets(SegmentedTensor<ActiveDataType, FractionalDataType>::Allocate(k_cDimensionsMax, GetVectorLengthFlatCore(runtimeLearningTypeOrCountTargetClasses)))
      , m_cFeatures(cFeatures)
      , m_aFeatures(0 == cFeatures || IsMultiplyError(sizeof(FeatureCore), cFeatures) ? nullptr : static_cast<FeatureCore *>(malloc(sizeof(FeatureCore) * cFeatures)))
      // we catch any errors in the constructor, so this should not be able to throw
      , m_cachedThreadResourcesUnion(runtimeLearningTypeOrCountTargetClasses) {
   }

   EBM_INLINE ~EbmTrainingState() {
      LOG_0(TraceLevelInfo, "Entered ~EbmTrainingState");

      if(IsClassification(m_runtimeLearningTypeOrCountTargetClasses)) {
         // member classes inside a union requre explicit call to destructor
         LOG_0(TraceLevelInfo, "~EbmTrainingState identified as classification type");
         m_cachedThreadResourcesUnion.classification.~CachedTrainingThreadResources();
      } else {
         EBM_ASSERT(IsRegression(m_runtimeLearningTypeOrCountTargetClasses));
         // member classes inside a union requre explicit call to destructor
         LOG_0(TraceLevelInfo, "~EbmTrainingState identified as regression type");
         m_cachedThreadResourcesUnion.regression.~CachedTrainingThreadResources();
      }

      SamplingWithReplacement::FreeSamplingSets(m_cSamplingSets, m_apSamplingSets);

      delete m_pTrainingSet;
      delete m_pValidationSet;

      FeatureCombinationCore::FreeFeatureCombinations(m_cFeatureCombinations, m_apFeatureCombinations);

      free(m_aFeatures);

      DeleteSegmentedTensors(m_cFeatureCombinations, m_apCurrentModel);
      DeleteSegmentedTensors(m_cFeatureCombinations, m_apBestModel);
      SegmentedTensor<ActiveDataType, FractionalDataType>::Free(m_pSmallChangeToModelOverwriteSingleSamplingSet);
      SegmentedTensor<ActiveDataType, FractionalDataType>::Free(m_pSmallChangeToModelAccumulatedFromSamplingSets);

      LOG_0(TraceLevelInfo, "Exited ~EbmTrainingState");
   }

   static void DeleteSegmentedTensors(const size_t cFeatureCombinations, SegmentedTensor<ActiveDataType, FractionalDataType> ** const apSegmentedTensors);
   static SegmentedTensor<ActiveDataType, FractionalDataType> ** InitializeSegmentedTensors(const size_t cFeatureCombinations, const FeatureCombinationCore * const * const apFeatureCombinations, const size_t cVectorLength);
   bool Initialize(const IntegerDataType randomSeed, const EbmCoreFeature * const aFeatures, const EbmCoreFeatureCombination * const aFeatureCombinations, const IntegerDataType * featureCombinationIndexes, const size_t cTrainingInstances, const void * const aTrainingTargets, const IntegerDataType * const aTrainingBinnedData, const FractionalDataType * const aTrainingPredictorScores, const size_t cValidationInstances, const void * const aValidationTargets, const IntegerDataType * const aValidationBinnedData, const FractionalDataType * const aValidationPredictorScores);
};

#endif // EBM_TRAINING_STATE_H