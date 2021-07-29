// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <ebm@koch.ninja>

#ifndef EBM_NATIVE_TEST_HPP
#define EBM_NATIVE_TEST_HPP

#define UNUSED(x) (void)(x)

enum class TestPriority {
   Discretize,
   DataSetShared,
   BoostingUnusualInputs,
   InteractionUnusualInputs,
   Rehydration,
   BitPackingExtremes,
   RandomNumbers,
   SuggestGraphBounds,
   CutUniform,
   CutWinsorized,
   CutQuantile
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


int RegisterTestHidden(const TestCaseHidden & testCaseHidden);

#define CONCATENATE_STRINGS(t1, t2) t1##t2
#define CONCATENATE_TOKENS(t1, t2) CONCATENATE_STRINGS(t1, t2)
#define TEST_CASE(description) \
   static void CONCATENATE_TOKENS(TEST_FUNCTION_HIDDEN_, __LINE__)(TestCaseHidden& testCaseHidden); \
   static int CONCATENATE_TOKENS(UNUSED_INTEGER_HIDDEN_, __LINE__) = \
   RegisterTestHidden(TestCaseHidden(&CONCATENATE_TOKENS(TEST_FUNCTION_HIDDEN_, __LINE__), description, k_filePriority)); \
   static void CONCATENATE_TOKENS(TEST_FUNCTION_HIDDEN_, __LINE__)(TestCaseHidden& testCaseHidden)

void FAILED(const double val, TestCaseHidden * const pTestCaseHidden, const std::string message);

bool IsApproxEqual(const double value, const double expected, const double percentage);

// this will ONLY work if used inside the root TEST_CASE function.  The testCaseHidden variable comes from TEST_CASE and should be visible inside the 
// function where CHECK(expression) is called
#define CHECK(expression) \
   do { \
      const bool bFailedHidden = !(expression); \
      if(bFailedHidden) { \
         FAILED(double { 0 }, &testCaseHidden, std::string(" FAILED on \"" #expression "\"")); \
      } \
   } while( (void)0, 0)

// this will ONLY work if used inside the root TEST_CASE function.  The testCaseHidden variable comes from TEST_CASE and should be visible inside the 
// function where CHECK_APPROX(expression) is called
#define CHECK_APPROX(value, expected) \
   do { \
      const double valueHidden = (value); \
      const bool bApproxEqualHidden = IsApproxEqual(valueHidden, static_cast<double>(expected), double { 1e-6 }); \
      if(!bApproxEqualHidden) { \
         FAILED(valueHidden, &testCaseHidden, std::string(" FAILED on \"" #value "(") + std::to_string(valueHidden) + ") approx " #expected "\""); \
      } \
   } while( (void)0, 0)

// this will ONLY work if used inside the root TEST_CASE function.  The testCaseHidden variable comes from TEST_CASE and should be visible inside the 
// function where CHECK_APPROX(expression) is called
#define CHECK_APPROX_TOLERANCE(value, expected, tolerance) \
   do { \
      const double valueHidden = static_cast<double>(value); \
      const bool bApproxEqualHidden = IsApproxEqual(valueHidden, static_cast<double>(expected), static_cast<double>(tolerance)); \
      if(!bApproxEqualHidden) { \
         FAILED(valueHidden, &testCaseHidden, std::string(" FAILED on \"" #value "(") + std::to_string(valueHidden) + ") approx " #expected "\""); \
      } \
   } while( (void)0, 0)

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

class FeatureTest final {
public:

   const bool m_bCategorical;
   const IntEbmType m_countBins;

   inline FeatureTest(
      const IntEbmType countBins, 
      const bool bCategorical = false
   ) :
      m_bCategorical(bCategorical),
      m_countBins(countBins) {
      if(countBins < 0) {
         exit(1);
      }
   }
};

class TestSample final {
public:
   const std::vector<IntEbmType> m_binnedDataPerFeatureArray;
   const FloatEbmType m_target;
   const bool m_bNullWeight;
   const FloatEbmType m_weight;
   const std::vector<FloatEbmType> m_priorScore;

   inline TestSample(const std::vector<IntEbmType> binnedDataPerFeatureArray, const FloatEbmType target) :
      m_binnedDataPerFeatureArray(binnedDataPerFeatureArray),
      m_target(target),
      m_bNullWeight(true),
      m_weight(1) {
   }

   inline TestSample(
      const std::vector<IntEbmType> binnedDataPerFeatureArray, 
      const FloatEbmType target,
      const FloatEbmType weight,
      const std::vector<FloatEbmType> priorScore = {}
   ) :
      m_binnedDataPerFeatureArray(binnedDataPerFeatureArray),
      m_target(target),
      m_bNullWeight(false),
      m_weight(weight),
      m_priorScore(priorScore) {
   }
};

static constexpr ptrdiff_t k_iZeroClassificationLogitDefault = ptrdiff_t { -1 };
static constexpr IntEbmType k_countInnerBagsDefault = IntEbmType { 0 };
static constexpr FloatEbmType k_learningRateDefault = FloatEbmType { 0.01 };
static constexpr IntEbmType k_countSamplesRequiredForChildSplitMinDefault = IntEbmType { 1 };

static constexpr IntEbmType k_leavesMaxFillDefault = 5;
// 64 dimensions is the most we can express with a 64 bit IntEbmType
static const std::vector<IntEbmType> k_leavesMaxDefault = { 
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault },
   IntEbmType { k_leavesMaxFillDefault }
};

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

   std::vector<BoolEbmType> m_featuresCategorical;
   std::vector<IntEbmType> m_featuresBinCount;
   std::vector<IntEbmType> m_featureGroupsDimensionCount;
   std::vector<IntEbmType> m_featureGroupsFeatureIndexes;

   std::vector<std::vector<size_t>> m_countBinsByFeatureGroup;

   std::vector<FloatEbmType> m_trainingRegressionTargets;
   std::vector<IntEbmType> m_trainingClassificationTargets;
   std::vector<IntEbmType> m_trainingBinnedData;
   std::vector<FloatEbmType> m_trainingWeights;
   std::vector<FloatEbmType> m_trainingPredictionScores;
   bool m_bNullTrainingWeights;
   bool m_bNullTrainingPredictionScores;

   std::vector<FloatEbmType> m_validationRegressionTargets;
   std::vector<IntEbmType> m_validationClassificationTargets;
   std::vector<IntEbmType> m_validationBinnedData;
   std::vector<FloatEbmType> m_validationWeights;
   std::vector<FloatEbmType> m_validationPredictionScores;
   bool m_bNullValidationWeights;
   bool m_bNullValidationPredictionScores;

   BoosterHandle m_boosterHandle;

   std::vector<FloatEbmType> m_interactionRegressionTargets;
   std::vector<IntEbmType> m_interactionClassificationTargets;
   std::vector<IntEbmType> m_interactionBinnedData;
   std::vector<FloatEbmType> m_interactionWeights;
   std::vector<FloatEbmType> m_interactionPredictionScores;
   bool m_bNullInteractionWeights;
   bool m_bNullInteractionPredictionScores;

   InteractionHandle m_interactionHandle;

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
      return m_featureGroupsDimensionCount.size();
   }

   void AddFeatures(const std::vector<FeatureTest> features);
   void AddFeatureGroups(const std::vector<std::vector<size_t>> featureGroups);
   void AddTrainingSamples(const std::vector<TestSample> samples);
   void AddValidationSamples(const std::vector<TestSample> samples);
   void InitializeBoosting(const IntEbmType countInnerBags = k_countInnerBagsDefault);
   
   FloatEbmType Boost(
      const IntEbmType indexFeatureGroup,
      const GenerateUpdateOptionsType options = GenerateUpdateOptions_Default,
      const FloatEbmType learningRate = k_learningRateDefault,
      const IntEbmType countSamplesRequiredForChildSplitMin = k_countSamplesRequiredForChildSplitMinDefault,
      const std::vector<IntEbmType> leavesMax = k_leavesMaxDefault
   );

   FloatEbmType GetBestModelPredictorScore(
      const size_t iFeatureGroup, 
      const std::vector<size_t> indexes, 
      const size_t iScore
   ) const;
   
   void GetBestModelFeatureGroupRaw(const size_t iFeatureGroup, FloatEbmType * const aModelValues) const;

   FloatEbmType GetCurrentModelPredictorScore(
      const size_t iFeatureGroup,
      const std::vector<size_t> indexes,
      const size_t iScore
   ) const;

   void GetCurrentModelFeatureGroupRaw(const size_t iFeatureGroup, FloatEbmType * const aModelValues) const;

   void AddInteractionSamples(const std::vector<TestSample> samples);

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
   IntEbmType countCuts,
   FloatEbmType * cutsLowerBoundInclusive,
   IntEbmType isMissingPresent,
   FloatEbmType minValue,
   FloatEbmType maxValue
);

#endif // EBM_NATIVE_TEST_HPP
