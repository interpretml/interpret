// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <ebm@koch.ninja>

#ifndef LIBEBM_TEST_HPP
#define LIBEBM_TEST_HPP

#include <stddef.h> // ptrdiff, size_t
#include <limits> // std::numeric_limits
#include <cmath> // std::nextafter
#include <string> // std::string
#include <vector> // std::vector
#include <assert.h> // assert

#include "libebm.h" // IntEbm

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

class TestException final : public std::exception {
   const ErrorEbm m_error;
   const std::string m_message;

public:
   TestException() : m_error(Error_None) {
   }
   TestException(const ErrorEbm error) : m_error(error) {
   }
   TestException(const char * const message) : m_error(Error_None), m_message(message) {
   }
   TestException(const ErrorEbm error, const char * const message) : m_error(error), m_message(message) {
   }

   const std::string & GetMessage() const {
      return m_message;
   }
   ErrorEbm GetError() const {
      return m_error;
   }
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
inline static double DenormalizeTest(const double val) noexcept {
   double deprecisioned[1];
   deprecisioned[0] = val;
   CleanFloats(1, deprecisioned);
   return deprecisioned[0];
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
      const bool bApproxEqualHidden = IsApproxEqual(valHidden, static_cast<double>(expected), double { 1e-4 }); \
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

inline constexpr static bool IsClassification(const OutputType cClasses) {
   return OutputType_GeneralClassification <= cClasses;
}

inline constexpr static size_t GetCountScores(const OutputType cClasses) {
#ifdef EXPAND_BINARY_LOGITS
   return OutputType_BinaryClassification <= cClasses ? static_cast<size_t>(cClasses) : (ptrdiff_t { 0 } == cClasses || ptrdiff_t { 1 } == cClasses ? size_t { 0 } : size_t { 1 });
#else // EXPAND_BINARY_LOGITS
   return OutputType_BinaryClassification < cClasses ? static_cast<size_t>(cClasses) : (ptrdiff_t { 0 } == cClasses || ptrdiff_t { 1 } == cClasses ? size_t { 0 } : size_t { 1 });
#endif // EXPAND_BINARY_LOGITS
}

static constexpr SeedEbm k_seed = SeedEbm { -42 };

class FeatureTest final {
public:

   const IntEbm m_countBins;
   const bool m_bMissing;
   const bool m_bUnknown;
   const bool m_bNominal;

   inline FeatureTest(
      const IntEbm countBins, 
      const bool bMissing = true,
      const bool bUnknown = true,
      const bool bNominal = false
   ) :
      m_countBins(countBins),
      m_bMissing(bMissing),
      m_bUnknown(bUnknown),
      m_bNominal(bNominal)
   {
   }
};

class TestSample final {
public:
   const bool m_bBag;
   const BagEbm m_bagCount;
   const std::vector<IntEbm> m_sampleBinIndexes;
   const double m_target;
   const bool m_bWeight;
   const double m_weight;
   const bool m_bScores;
   const std::vector<double> m_initScores;

   inline TestSample(
      const std::vector<IntEbm> sampleBinIndexes, 
      const double target
   ) :
      m_bBag(false),
      m_bagCount(0),
      m_sampleBinIndexes(sampleBinIndexes),
      m_target(target),
      m_bWeight(false),
      m_weight(1.0),
      m_bScores(false) {
   }

   inline TestSample(
      const std::vector<IntEbm> sampleBinIndexes,
      const double target,
      const double weight
   ) :
      m_bBag(false),
      m_bagCount(0),
      m_sampleBinIndexes(sampleBinIndexes),
      m_target(target),
      m_bWeight(true),
      m_weight(weight),
      m_bScores(false) {
   }

   inline TestSample(
      const std::vector<IntEbm> sampleBinIndexes,
      const double target,
      const std::vector<double> initScores
   ) :
      m_bBag(false),
      m_bagCount(0),
      m_sampleBinIndexes(sampleBinIndexes),
      m_target(target),
      m_bWeight(false),
      m_weight(1.0),
      m_bScores(true),
      m_initScores(initScores) {
   }

   inline TestSample(
      const std::vector<IntEbm> sampleBinIndexes,
      const double target,
      const double weight,
      const std::vector<double> initScores
   ) :
      m_bBag(false),
      m_bagCount(0),
      m_sampleBinIndexes(sampleBinIndexes),
      m_target(target),
      m_bWeight(true),
      m_weight(weight),
      m_bScores(true),
      m_initScores(initScores) {
   }

   inline TestSample(
      BagEbm bagCount, 
      const std::vector<IntEbm> sampleBinIndexes, 
      const double target
   ) :
      m_bBag(true),
      m_bagCount(bagCount),
      m_sampleBinIndexes(sampleBinIndexes),
      m_target(target),
      m_bWeight(false),
      m_weight(1.0),
      m_bScores(false) {
   }

   inline TestSample(
      BagEbm bagCount,
      const std::vector<IntEbm> sampleBinIndexes,
      const double target,
      const double weight
   ) :
      m_bBag(true),
      m_bagCount(bagCount),
      m_sampleBinIndexes(sampleBinIndexes),
      m_target(target),
      m_bWeight(true),
      m_weight(weight),
      m_bScores(false) {
   }

   inline TestSample(
      BagEbm bagCount,
      const std::vector<IntEbm> sampleBinIndexes,
      const double target,
      const std::vector<double> initScores
   ) :
      m_bBag(true),
      m_bagCount(bagCount),
      m_sampleBinIndexes(sampleBinIndexes),
      m_target(target),
      m_bWeight(false),
      m_weight(1.0),
      m_bScores(true),
      m_initScores(initScores) {
   }

   inline TestSample(
      BagEbm bagCount,
      const std::vector<IntEbm> sampleBinIndexes,
      const double target,
      const double weight,
      const std::vector<double> initScores
   ) :
      m_bBag(true),
      m_bagCount(bagCount),
      m_sampleBinIndexes(sampleBinIndexes),
      m_target(target),
      m_bWeight(true),
      m_weight(weight),
      m_bScores(true),
      m_initScores(initScores) {
   }
};

static constexpr ptrdiff_t k_iZeroClassificationLogitDefault = ptrdiff_t { -1 };
static constexpr IntEbm k_countInnerBagsDefault = IntEbm { 0 };
static constexpr double k_learningRateDefault = double { 0.01 };
static constexpr IntEbm k_minSamplesLeafDefault = IntEbm { 1 };
static constexpr CreateBoosterFlags k_testCreateBoosterFlags_Default = CreateBoosterFlags_Default;
static constexpr CreateInteractionFlags k_testCreateInteractionFlags_Default = CreateInteractionFlags_Default;
static constexpr ComputeFlags k_testComputeFlags_Default = ComputeFlags_Default;

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

class TestBoost {
   const OutputType m_cClasses;
   const std::vector<FeatureTest> m_features;
   const std::vector<std::vector<IntEbm>> m_termFeatures;
   const ptrdiff_t m_iZeroClassificationLogit;

   std::vector<unsigned char> m_rng;
   BoosterHandle m_boosterHandle;

   const double * GetTermScores(
      const size_t iTerm,
      const double * const aTermScores,
      const std::vector<size_t> perDimensionIndexArrayForBinnedFeatures
   ) const;

   double GetTermScore(
      const size_t iTerm,
      const double * const aTermScores,
      const std::vector<size_t> perDimensionIndexArrayForBinnedFeatures,
      const size_t iClassOrZero
   ) const;

public:

   TestBoost(
      const OutputType cClasses,
      const std::vector<FeatureTest> features,
      const std::vector<std::vector<IntEbm>> termFeatures,
      const std::vector<TestSample> train,
      const std::vector<TestSample> validation,
      const IntEbm countInnerBags = k_countInnerBagsDefault,
      const CreateBoosterFlags flags = k_testCreateBoosterFlags_Default,
      const ComputeFlags disableCompute = k_testComputeFlags_Default,
      const char * const sObjective = nullptr,
      const ptrdiff_t iZeroClassificationLogit = k_iZeroClassificationLogitDefault
   );
   ~TestBoost();

   inline size_t GetCountTerms() const {
      return m_termFeatures.size();
   }

   inline BoosterHandle GetBoosterHandle() {
      return m_boosterHandle;
   }

   BoostRet Boost(
      const IntEbm indexTerm,
      const TermBoostFlags flags = TermBoostFlags_Default,
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
};


class TestInteraction {
   InteractionHandle m_interactionHandle;

public:

   TestInteraction(
      const OutputType cClasses,
      const std::vector<FeatureTest> features,
      const std::vector<TestSample> samples,
      const CreateInteractionFlags flags = k_testCreateInteractionFlags_Default,
      const ComputeFlags disableCompute = k_testComputeFlags_Default,
      const char * const sObjective = nullptr,
      const ptrdiff_t iZeroClassificationLogit = k_iZeroClassificationLogitDefault
   );
   ~TestInteraction();

   inline InteractionHandle GetInteractionHandle() {
      return m_interactionHandle;
   }

   double TestCalcInteractionStrength(
      const std::vector<IntEbm> features,
      const CalcInteractionFlags flags = CalcInteractionFlags_Default,
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

#endif // LIBEBM_TEST_HPP
