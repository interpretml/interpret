// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <ebm@koch.ninja>

#ifndef EBM_NATIVE_TEST_HPP
#define EBM_NATIVE_TEST_HPP

#define UNUSED(x) (void)(x)
#define EBM_ASSERT(x) assert(x)

enum class TestPriority {
   DataSetShared,
   BoostingUnusualInputs,
   InteractionUnusualInputs,
   Rehydration,
   BitPackingExtremes,
   RandomNumbers,
   SuggestGraphBounds,
   CutUniform,
   CutWinsorized,
   CutQuantile,
   Discretize
};


inline static double FloatTickIncrementTest(const double v) noexcept {
   // this function properly handles subnormals by skipping over them on all systems regardless of the FP unit flags.

   assert(!std::isnan(v));
   assert(!std::isinf(v));
   assert(std::numeric_limits<double>::max() != v);

   if(std::numeric_limits<double>::min() <= v || v < -std::numeric_limits<double>::min()) {
      // I have found nextafter fails badly with subnormals.  It doesn't advance!  We disallow all subnormals.
      return std::nextafter(v, std::numeric_limits<double>::max());
   } else if(-std::numeric_limits<double>::min() == v) {
      return double { 0 };
   } else {
      return std::numeric_limits<double>::min();
   }
}
inline static double FloatTickDecrementTest(const double v) noexcept {
   // this function properly handles subnormals by skipping over them on all systems regardless of the FP unit flags.

   assert(!std::isnan(v));
   assert(!std::isinf(v));
   assert(std::numeric_limits<double>::lowest() != v);

   if(v <= -std::numeric_limits<double>::min() || std::numeric_limits<double>::min() < v) {
      // I have found nextafter fails badly with subnormals.  It doesn't advance!  We disallow all subnormals.
      return std::nextafter(v, std::numeric_limits<double>::lowest());
   } else if(std::numeric_limits<double>::min() == v) {
      return double { 0 };
   } else {
      return -std::numeric_limits<double>::min();
   }
}
inline static double DenormalizeTest(const double v) noexcept {
   return v <= -std::numeric_limits<double>::min() ||
      std::numeric_limits<double>::min() <= v ? v : double { 0 };
}

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

bool IsApproxEqual(const double val, const double expected, const double percentage);

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
#define CHECK_APPROX(val, expected) \
   do { \
      const double valHidden = (val); \
      const bool bApproxEqualHidden = IsApproxEqual(valHidden, static_cast<double>(expected), double { 1e-6 }); \
      if(!bApproxEqualHidden) { \
         FAILED(valHidden, &testCaseHidden, std::string(" FAILED on \"" #val "(") + std::to_string(valHidden) + ") approx " #expected "\""); \
      } \
   } while( (void)0, 0)

// this will ONLY work if used inside the root TEST_CASE function.  The testCaseHidden variable comes from TEST_CASE and should be visible inside the 
// function where CHECK_APPROX(expression) is called
#define CHECK_APPROX_TOLERANCE(val, expected, tolerance) \
   do { \
      const double valHidden = static_cast<double>(val); \
      const bool bApproxEqualHidden = IsApproxEqual(valHidden, static_cast<double>(expected), static_cast<double>(tolerance)); \
      if(!bApproxEqualHidden) { \
         FAILED(valHidden, &testCaseHidden, std::string(" FAILED on \"" #val "(") + std::to_string(valHidden) + ") approx " #expected "\""); \
      } \
   } while( (void)0, 0)

// EBM/interpret specific stuff below here!!

static constexpr ptrdiff_t k_learningTypeRegression = ptrdiff_t { -1 };
inline constexpr static bool IsClassification(const ptrdiff_t cClasses) {
   return 0 <= cClasses;
}

inline constexpr static size_t GetCountScores(const ptrdiff_t cClasses) {
#ifdef EXPAND_BINARY_LOGITS
   return ptrdiff_t { 1 } < cClasses ? static_cast<size_t>(cClasses) : (ptrdiff_t { 0 } == cClasses || ptrdiff_t { 1 } == cClasses ? size_t { 0 } : size_t { 1 });
#else // EXPAND_BINARY_LOGITS
   return ptrdiff_t { 2 } < cClasses ? static_cast<size_t>(cClasses) : (ptrdiff_t { 0 } == cClasses || ptrdiff_t { 1 } == cClasses ? size_t { 0 } : size_t { 1 });
#endif // EXPAND_BINARY_LOGITS
}

static constexpr SeedEbm k_seed = SeedEbm { -42 };

class FeatureTest final {
public:

   const bool m_bNominal;
   const IntEbm m_countBins;

   inline FeatureTest(
      const IntEbm countBins, 
      const bool bNominal = false
   ) :
      m_bNominal(bNominal),
      m_countBins(countBins) {
      if(countBins < 0) {
         exit(1);
      }
   }
};

class TestSample final {
public:
   const std::vector<IntEbm> m_sampleBinIndexes;
   const double m_target;
   const bool m_bNullWeight;
   const double m_weight;
   const std::vector<double> m_initScores;

   inline TestSample(const std::vector<IntEbm> sampleBinIndexes, const double target) :
      m_sampleBinIndexes(sampleBinIndexes),
      m_target(target),
      m_bNullWeight(true),
      m_weight(1.0) {
   }

   inline TestSample(
      const std::vector<IntEbm> sampleBinIndexes, 
      const double target,
      const double weight,
      const std::vector<double> initScores = {}
   ) :
      m_sampleBinIndexes(sampleBinIndexes),
      m_target(target),
      m_bNullWeight(false),
      m_weight(weight),
      m_initScores(initScores) {
   }
};

static constexpr ptrdiff_t k_iZeroClassificationLogitDefault = ptrdiff_t { -1 };
static constexpr IntEbm k_countInnerBagsDefault = IntEbm { 0 };
static constexpr double k_learningRateDefault = double { 0.01 };
static constexpr IntEbm k_minSamplesLeafDefault = IntEbm { 1 };

static constexpr IntEbm k_leavesMaxFillDefault = 5;
// 64 dimensions is the most we can express with a 64 bit IntEbm
static const std::vector<IntEbm> k_leavesMaxDefault = { 
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault },
   IntEbm { k_leavesMaxFillDefault }
};

struct BoostRet {
   double gainAvg;
   double validationMetric;
};

class TestApi {
   enum class Stage {
      Beginning, 
      FeaturesAdded, 
      TermsAdded, 
      TrainingAdded, 
      ValidationAdded, 
      InitializedBoosting, 
      InteractionAdded, 
      InitializedInteraction
   };

   std::vector<unsigned char> m_rng;

   Stage m_stage;
   const ptrdiff_t m_cClasses;
   const ptrdiff_t m_iZeroClassificationLogit;

   std::vector<BoolEbm> m_featureNominals;
   std::vector<IntEbm> m_featureBinCounts;
   std::vector<IntEbm> m_dimensionCounts;
   std::vector<IntEbm> m_featureIndexes;

   std::vector<std::vector<size_t>> m_termBinCounts;

   std::vector<double> m_trainingRegressionTargets;
   std::vector<IntEbm> m_trainingClassificationTargets;
   // TODO: make this a vector of vectors.  The first vector being indexed by iFeature
   std::vector<IntEbm> m_trainingBinIndexes;
   std::vector<double> m_trainingWeights;
   std::vector<double> m_trainingInitScores;
   bool m_bNullTrainingWeights;
   bool m_bNullTrainingInitScores;

   std::vector<double> m_validationRegressionTargets;
   std::vector<IntEbm> m_validationClassificationTargets;
   // TODO: make this a vector of vectors.  The first vector being indexed by iFeature
   std::vector<IntEbm> m_validationBinIndexes;
   std::vector<double> m_validationWeights;
   std::vector<double> m_validationInitScores;
   bool m_bNullValidationWeights;
   bool m_bNullValidationInitScores;

   BoosterHandle m_boosterHandle;

   std::vector<double> m_interactionRegressionTargets;
   std::vector<IntEbm> m_interactionClassificationTargets;
   std::vector<IntEbm> m_interactionBinIndexes;
   std::vector<double> m_interactionWeights;
   std::vector<double> m_interactionInitScores;
   bool m_bNullInteractionWeights;
   bool m_bNullInteractionInitScores;

   InteractionHandle m_interactionHandle;

   const double * GetTermScores(
      const size_t iTerm,
      const double * const aTermScores,
      const std::vector<size_t> perDimensionIndexArrayForBinnedFeatures)
      const;

   double GetTermScore(
      const size_t iTerm,
      const double * const aTermScores,
      const std::vector<size_t> perDimensionIndexArrayForBinnedFeatures,
      const size_t iClassOrZero)
      const;

public:

   TestApi(
      const ptrdiff_t cClasses, 
      const ptrdiff_t iZeroClassificationLogit = k_iZeroClassificationLogitDefault
   );
   ~TestApi();

   inline size_t GetCountTerms() const {
      return m_dimensionCounts.size();
   }

   inline BoosterHandle GetBoosterHandle() {
      return m_boosterHandle;
   }

   inline InteractionHandle GetInteractionHandle() {
      return m_interactionHandle;
   }

   void AddFeatures(const std::vector<FeatureTest> features);
   void AddTerms(const std::vector<std::vector<size_t>> termFeatures);
   void AddTrainingSamples(const std::vector<TestSample> samples);
   void AddValidationSamples(const std::vector<TestSample> samples);
   void InitializeBoosting(const IntEbm countInnerBags = k_countInnerBagsDefault);
   
   BoostRet Boost(
      const IntEbm indexTerm,
      const BoostFlags flags = BoostFlags_Default,
      const double learningRate = k_learningRateDefault,
      const IntEbm minSamplesLeaf = k_minSamplesLeafDefault,
      const std::vector<IntEbm> leavesMax = k_leavesMaxDefault
   );

   double GetBestTermScore(
      const size_t iTerm, 
      const std::vector<size_t> indexes, 
      const size_t iScore
   ) const;
   
   void GetBestTermScoresRaw(const size_t iTerm, double * const aTermScores) const;

   double GetCurrentTermScore(
      const size_t iTerm,
      const std::vector<size_t> indexes,
      const size_t iScore
   ) const;

   void GetCurrentTermScoresRaw(const size_t iTerm, double * const aTermScores) const;

   void AddInteractionSamples(const std::vector<TestSample> samples);

   void InitializeInteraction();

   double TestCalcInteractionStrength(
      const std::vector<IntEbm> features, 
      const InteractionFlags flags = InteractionFlags_Default,
      const IntEbm minSamplesLeaf = k_minSamplesLeafDefault
   ) const;
};

void DisplayCuts(
   IntEbm countSamples,
   double * featureVals,
   IntEbm countBinsMax,
   IntEbm minSamplesBin,
   IntEbm countCuts,
   double * cutsLowerBoundInclusive,
   IntEbm isMissingPresent,
   double minFeatureVal,
   double maxFeatureVal
);

#endif // EBM_NATIVE_TEST_HPP
