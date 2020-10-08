// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <ebm@koch.ninja>

#ifndef EBM_NATIVE_TEST_H
#define EBM_NATIVE_TEST_H

#define UNUSED(x) (void)(x)

enum class TestPriority {
   RandomNumbers,
   SuggestGraphBounds,
   Discretize,
   GenerateUniformBinCuts,
   GenerateWinsorizedBinCuts,
   GenerateQuantileBinCuts,
   BoostingUnusualInputs,
   InteractionUnusualInputs,
   Rehydration,
   BitPackingExtremes
};

class TestCaseHidden;
typedef void (*TestFunctionHidden)(TestCaseHidden & testCaseHidden);

class TestCaseHidden {
public:
   inline TestCaseHidden(TestFunctionHidden pTestFunction, std::string description, TestPriority testPriority) {
      m_pTestFunction = pTestFunction;
      m_description = description;
      m_bPassed = true;
      m_testPriority = testPriority;
   }

   TestFunctionHidden m_pTestFunction;
   std::string m_description;
   bool m_bPassed;
   TestPriority m_testPriority;
};

constexpr inline bool AlwaysFalse() {
   return false;
}

int RegisterTestHidden(const TestCaseHidden & testCaseHidden);

#define CONCATENATE_STRINGS(t1, t2) t1##t2
#define CONCATENATE_TOKENS(t1, t2) CONCATENATE_STRINGS(t1, t2)
#define TEST_CASE(description) \
   static void CONCATENATE_TOKENS(TEST_FUNCTION_HIDDEN_, __LINE__)(TestCaseHidden& testCaseHidden); \
   static int CONCATENATE_TOKENS(UNUSED_INTEGER_HIDDEN_, __LINE__) = \
   RegisterTestHidden(TestCaseHidden(&CONCATENATE_TOKENS(TEST_FUNCTION_HIDDEN_, __LINE__), description, k_filePriority)); \
   static void CONCATENATE_TOKENS(TEST_FUNCTION_HIDDEN_, __LINE__)(TestCaseHidden& testCaseHidden)

void FAILED(TestCaseHidden * const pTestCaseHidden);

bool IsApproxEqual(const double value, const double expected, const double percentage);

// this will ONLY work if used inside the root TEST_CASE function.  The testCaseHidden variable comes from TEST_CASE and should be visible inside the 
// function where CHECK(expression) is called
#define CHECK(expression) \
   do { \
      const bool bFailedHidden = !(expression); \
      if(bFailedHidden) { \
         std::cout << " FAILED on \"" #expression "\""; \
         FAILED(&testCaseHidden); \
      } \
   } while(AlwaysFalse())

// this will ONLY work if used inside the root TEST_CASE function.  The testCaseHidden variable comes from TEST_CASE and should be visible inside the 
// function where CHECK_APPROX(expression) is called
#define CHECK_APPROX(value, expected) \
   do { \
      const double valueHidden = (value); \
      const bool bApproxEqualHidden = IsApproxEqual(valueHidden, static_cast<double>(expected), double { 1e-6 }); \
      if(!bApproxEqualHidden) { \
         std::cout << " FAILED on \"" #value "(" << valueHidden << ") approx " #expected "\""; \
         FAILED(&testCaseHidden); \
      } \
   } while(AlwaysFalse())

// EBM/interpret specific stuff below here!!

constexpr ptrdiff_t k_learningTypeRegression = ptrdiff_t { -1 };
constexpr bool IsClassification(const ptrdiff_t learningTypeOrCountTargetClasses) {
   return 0 <= learningTypeOrCountTargetClasses;
}

constexpr size_t GetVectorLength(const ptrdiff_t learningTypeOrCountTargetClasses) {
#ifdef EXPAND_BINARY_LOGITS
#ifdef REDUCE_MULTICLASS_LOGITS

   // EXPAND_BINARY_LOGITS && REDUCE_MULTICLASS_LOGITS
#error we should not be expanding binary logits while reducing multiclass logits

#else // REDUCE_MULTICLASS_LOGITS

   // EXPAND_BINARY_LOGITS && !REDUCE_MULTICLASS_LOGITS
   return learningTypeOrCountTargetClasses <= ptrdiff_t { 1 } ? 
      size_t { 1 } : 
      static_cast<size_t>(learningTypeOrCountTargetClasses);

#endif // REDUCE_MULTICLASS_LOGITS
#else // EXPAND_BINARY_LOGITS
#ifdef REDUCE_MULTICLASS_LOGITS

   // !EXPAND_BINARY_LOGITS && REDUCE_MULTICLASS_LOGITS
   return learningTypeOrCountTargetClasses <= ptrdiff_t { 2 } ? 
      size_t { 1 } : 
      static_cast<size_t>(learningTypeOrCountTargetClasses) - size_t { 1 };

#else // REDUCE_MULTICLASS_LOGITS

   // !EXPAND_BINARY_LOGITS && !REDUCE_MULTICLASS_LOGITS
   return learningTypeOrCountTargetClasses <= ptrdiff_t { 2 } ? 
      size_t { 1 } : 
      static_cast<size_t>(learningTypeOrCountTargetClasses);

#endif // REDUCE_MULTICLASS_LOGITS
#endif // EXPAND_BINARY_LOGITS
}

constexpr SeedEbmType k_randomSeed = SeedEbmType { -42 };
enum class FeatureType : IntEbmType {
   Ordinal = FeatureTypeOrdinal, Nominal = FeatureTypeNominal
};

class FeatureTest final {
public:

   const FeatureType m_featureType;
   const bool m_hasMissing;
   const IntEbmType m_countBins;

   inline FeatureTest(
      const IntEbmType countBins, 
      const FeatureType featureType = FeatureType::Ordinal, 
      const bool hasMissing = false
   ) :
      m_featureType(featureType),
      m_hasMissing(hasMissing),
      m_countBins(countBins) {
      if(countBins < 0) {
         exit(1);
      }
   }
};

class RegressionSample final {
public:
   const FloatEbmType m_target;
   const std::vector<IntEbmType> m_binnedDataPerFeatureArray;
   const FloatEbmType m_priorPredictorPrediction;
   const bool m_bNullPredictionScores;

   inline RegressionSample(const FloatEbmType target, const std::vector<IntEbmType> binnedDataPerFeatureArray) :
      m_target(target),
      m_binnedDataPerFeatureArray(binnedDataPerFeatureArray),
      m_priorPredictorPrediction(0),
      m_bNullPredictionScores(true) {
   }

   inline RegressionSample(
      const FloatEbmType target, 
      const std::vector<IntEbmType> binnedDataPerFeatureArray, 
      const FloatEbmType priorPredictorPrediction
   ) :
      m_target(target),
      m_binnedDataPerFeatureArray(binnedDataPerFeatureArray),
      m_priorPredictorPrediction(priorPredictorPrediction),
      m_bNullPredictionScores(false) {
   }
};

class ClassificationSample final {
public:
   const IntEbmType m_target;
   const std::vector<IntEbmType> m_binnedDataPerFeatureArray;
   const std::vector<FloatEbmType> m_priorPredictorPerClassLogits;
   const bool m_bNullPredictionScores;

   inline ClassificationSample(const IntEbmType target, const std::vector<IntEbmType> binnedDataPerFeatureArray) :
      m_target(target),
      m_binnedDataPerFeatureArray(binnedDataPerFeatureArray),
      m_bNullPredictionScores(true) {
   }

   inline ClassificationSample(
      const IntEbmType target,
      const std::vector<IntEbmType> binnedDataPerFeatureArray,
      const std::vector<FloatEbmType> priorPredictorPerClassLogits)
      :
      m_target(target),
      m_binnedDataPerFeatureArray(binnedDataPerFeatureArray),
      m_priorPredictorPerClassLogits(priorPredictorPerClassLogits),
      m_bNullPredictionScores(false) {
   }
};

static constexpr ptrdiff_t k_iZeroClassificationLogitDefault = ptrdiff_t { -1 };
static constexpr IntEbmType k_countInnerBagsDefault = IntEbmType { 0 };
static constexpr FloatEbmType k_learningRateDefault = FloatEbmType { 0.01 };
static constexpr IntEbmType k_countTreeSplitsMaxDefault = IntEbmType { 4 };
static constexpr IntEbmType k_countSamplesRequiredForChildSplitMinDefault = IntEbmType { 1 };

class TestApi {
   enum class Stage {
      Beginning, 
      FeaturesAdded, 
      FeatureGroupsAdded, 
      TrainingAdded, 
      ValidationAdded, 
      InitializedBoosting, 
      InteractionAdded, 
      InitializedInteraction
   };

   Stage m_stage;
   const ptrdiff_t m_learningTypeOrCountTargetClasses;
   const ptrdiff_t m_iZeroClassificationLogit;

   std::vector<EbmNativeFeature> m_features;
   std::vector<EbmNativeFeatureGroup> m_featureGroups;
   std::vector<IntEbmType> m_featureGroupIndexes;

   std::vector<std::vector<size_t>> m_countBinsByFeatureGroup;

   std::vector<FloatEbmType> m_trainingRegressionTargets;
   std::vector<IntEbmType> m_trainingClassificationTargets;
   std::vector<IntEbmType> m_trainingBinnedData;
   std::vector<FloatEbmType> m_trainingPredictionScores;
   bool m_bNullTrainingPredictionScores;

   std::vector<FloatEbmType> m_validationRegressionTargets;
   std::vector<IntEbmType> m_validationClassificationTargets;
   std::vector<IntEbmType> m_validationBinnedData;
   std::vector<FloatEbmType> m_validationPredictionScores;
   bool m_bNullValidationPredictionScores;

   PEbmBoosting m_pEbmBoosting;

   std::vector<FloatEbmType> m_interactionRegressionTargets;
   std::vector<IntEbmType> m_interactionClassificationTargets;
   std::vector<IntEbmType> m_interactionBinnedData;
   std::vector<FloatEbmType> m_interactionPredictionScores;
   bool m_bNullInteractionPredictionScores;

   PEbmInteraction m_pEbmInteraction;

   const FloatEbmType * GetPredictorScores(
      const size_t iFeatureGroup,
      const FloatEbmType * const pModelFeatureGroup,
      const std::vector<size_t> perDimensionIndexArrayForBinnedFeatures)
      const;

   FloatEbmType GetPredictorScore(
      const size_t iFeatureGroup,
      const FloatEbmType * const pModelFeatureGroup,
      const std::vector<size_t> perDimensionIndexArrayForBinnedFeatures,
      const size_t iTargetClassOrZero)
      const;

public:

   TestApi(
      const ptrdiff_t learningTypeOrCountTargetClasses, 
      const ptrdiff_t iZeroClassificationLogit = k_iZeroClassificationLogitDefault
   );
   ~TestApi();

   inline size_t GetFeatureGroupsCount() const {
      return m_featureGroups.size();
   }

   void AddFeatures(const std::vector<FeatureTest> features);
   void AddFeatureGroups(const std::vector<std::vector<size_t>> featureGroups);
   void AddTrainingSamples(const std::vector<RegressionSample> samples);
   void AddTrainingSamples(const std::vector<ClassificationSample> samples);
   void AddValidationSamples(const std::vector<RegressionSample> samples);
   void AddValidationSamples(const std::vector<ClassificationSample> samples);
   void InitializeBoosting(const IntEbmType countInnerBags = k_countInnerBagsDefault);
   FloatEbmType Boost(const IntEbmType indexFeatureGroup, 
      const std::vector<FloatEbmType> trainingWeights = {}, 
      const std::vector<FloatEbmType> validationWeights = {}, 
      const FloatEbmType learningRate = k_learningRateDefault, 
      const IntEbmType countTreeSplitsMax = k_countTreeSplitsMaxDefault, 
      const IntEbmType countSamplesRequiredForChildSplitMin = k_countSamplesRequiredForChildSplitMinDefault
   );
   FloatEbmType GetBestModelPredictorScore(
      const size_t iFeatureGroup, 
      const std::vector<size_t> indexes, 
      const size_t iScore
   ) const;
   const FloatEbmType * GetBestModelFeatureGroupRaw(const size_t iFeatureGroup) const;
   FloatEbmType GetCurrentModelPredictorScore(
      const size_t iFeatureGroup,
      const std::vector<size_t> perDimensionIndexArrayForBinnedFeatures,
      const size_t iTargetClassOrZero)
      const;
   const FloatEbmType * GetCurrentModelFeatureGroupRaw(const size_t iFeatureGroup) const;
   void AddInteractionSamples(const std::vector<RegressionSample> samples);
   void AddInteractionSamples(const std::vector<ClassificationSample> samples);
   void InitializeInteraction();
   FloatEbmType InteractionScore(
      const std::vector<IntEbmType> featuresInGroup, 
      const IntEbmType countSamplesRequiredForChildSplitMin = k_countSamplesRequiredForChildSplitMinDefault
   ) const;
};

void DisplayCuts(
   IntEbmType countSamples,
   FloatEbmType * featureValues,
   IntEbmType countBinsMax,
   IntEbmType countSamplesPerBinMin,
   IntEbmType countBinCuts,
   FloatEbmType * binCutsLowerBoundInclusive,
   IntEbmType isMissingPresent,
   FloatEbmType minValue,
   FloatEbmType maxValue
);

#endif // EBM_NATIVE_TEST_H
