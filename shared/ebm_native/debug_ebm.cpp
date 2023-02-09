// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// We normally only test the external public interface because unit testing in C++ is a bit more complicated
// and would require us to make unit test specific builds.  For our package it seems we can get away with this, 
// yet still have good tests since most of the code is targetable from the outside and we cover a lot of the 
// code with just a few option combinations.  In some special cases though, it's useful to have some 
// glass box testing, so we have a few aspects tested specially here and invoked on startup in DEBUG
// builds.  Don't include significant code here, since the DEBUG build does get included in the package.  Normally
// it'll just eat some filespace though as the RELEASE build is the only one loaded by default

#include "precompiled_header_cpp.hpp"

#include <stddef.h> // size_t, ptrdiff_t
#include <random>

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "approximate_math.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

//#define INCLUDE_TESTS_IN_RELEASE
//#define ENABLE_TEST_LOG_SUM_ERRORS
//#define ENABLE_TEST_EXP_SUM_ERRORS
//#define ENABLE_TEST_SOFTMAX_SUM_ERRORS
//#define ENABLE_PRINTF

#if !defined(NDEBUG) || defined(INCLUDE_TESTS_IN_RELEASE)


#ifdef ENABLE_TEST_LOG_SUM_ERRORS
static double TestLogSumErrors() {
   double debugRet = 0; // this just prevents the optimizer from eliminating this code

   // check that our outputs match that of the std::log function
   EBM_ASSERT(std::isnan(std::log(std::numeric_limits<float>::quiet_NaN())));
   EBM_ASSERT(std::isnan(std::log(-std::numeric_limits<float>::infinity())));
   EBM_ASSERT(std::isnan(std::log(-1.0f))); // should be -NaN
   EBM_ASSERT(-std::numeric_limits<float>::infinity() == std::log(0.0f));
   EBM_ASSERT(-std::numeric_limits<float>::infinity() == std::log(-0.0f));
   EBM_ASSERT(std::numeric_limits<float>::infinity() == std::log(std::numeric_limits<float>::infinity()));

   // our function with the same tests as std::log
   EBM_ASSERT(std::isnan(LogApproxSchraudolph(std::numeric_limits<float>::quiet_NaN())));
   EBM_ASSERT(std::isnan(LogApproxSchraudolph(-std::numeric_limits<float>::infinity())));
   EBM_ASSERT(std::isnan(LogApproxSchraudolph(-1.0f))); // should be -NaN
   EBM_ASSERT(-std::numeric_limits<float>::infinity() == LogApproxSchraudolph(0.0f));
   EBM_ASSERT(-std::numeric_limits<float>::infinity() == LogApproxSchraudolph(-0.0f));
   // -87.3365479f == std::log(std::numeric_limits<float>::min())
   EBM_ASSERT(-87.5f < LogApproxSchraudolph(std::numeric_limits<float>::min()));
   EBM_ASSERT(LogApproxSchraudolph(std::numeric_limits<float>::min()) < -87.25f);
   // -103.278931f == std::log(std::numeric_limits<float>::denorm_min())
   EBM_ASSERT(-88.125f < LogApproxSchraudolph(std::numeric_limits<float>::denorm_min()));
   EBM_ASSERT(LogApproxSchraudolph(std::numeric_limits<float>::denorm_min()) < -87.75f);
   // 88.7228394f == std::log(std::numeric_limits<float>::max())
   EBM_ASSERT(88.5f < LogApproxSchraudolph(std::numeric_limits<float>::max()));
   EBM_ASSERT(LogApproxSchraudolph(std::numeric_limits<float>::max()) < 88.875f);
   EBM_ASSERT(std::numeric_limits<float>::infinity() == LogApproxSchraudolph(std::numeric_limits<float>::infinity()));

   // our exp error has a periodicity of ln(2), so [0, ln(2)) should have the same relative error as 
   // [ln(2), 2 * ln(2)) OR [-ln(2), 0) OR [-ln(2)/2, +ln(2)/2) 
   // BUT, this needs to be evenly distributed, so we can't use nextafter. We need to increment with a constant.
   // boosting will push our exp values to between 1 and 1 + a small number less than e
   static constexpr double k_testLowerInclusiveBound = 1; // this is true lower bound
   static constexpr double k_testUpperExclusiveBound = static_cast<double>(1.0f + 1000 * std::numeric_limits<float>::epsilon()); // 2 would be random guessing. we should optimize for the final rounds.  1.5 gives a log loss about 0.4 which is on the high side of log loss
   static constexpr uint64_t k_cTests = 123513;
   static constexpr bool k_bIsRandom = false;
   static constexpr bool k_bIsRandomFinalFill = true; // if true we choose a random value to randomly fill the space between ticks
   static constexpr float termMid = k_logTermLowerBoundInputCloseToOne;
   static constexpr uint32_t termStepsFromMid = 5;
   static constexpr uint32_t termStepDistance = 1;

   static constexpr ptrdiff_t k_cStats = termStepsFromMid * 2 + 1;
   static constexpr double k_movementTick = (k_testUpperExclusiveBound - k_testLowerInclusiveBound) / k_cTests;

   // uniform_real_distribution includes the lower bound, but not the upper bound, which is good because
   // our window is balanced by not including both ends
   std::uniform_real_distribution<double> testDistribution(k_bIsRandom ? k_testLowerInclusiveBound : double { 0 }, k_bIsRandom ? k_testUpperExclusiveBound : k_movementTick);
   std::mt19937 testRandom(52);


   float addTerm = termMid;
   for(int i = 0; i < termStepDistance * termStepsFromMid; ++i) {
      addTerm = std::nextafter(addTerm, std::numeric_limits<float>::lowest());
   }

   for(ptrdiff_t iStat = 0; iStat < k_cStats; ++iStat) {
      if(k_logTermLowerBound <= addTerm && addTerm <= k_logTermUpperBound) {
         double avgAbsError = 0;
         double avgError = 0;
         double avgSquareError = 0;
         double minError = std::numeric_limits<double>::max();
         double maxError = std::numeric_limits<double>::lowest();

         for(uint64_t iTest = 0; iTest < k_cTests; ++iTest) {
            double val;
            if(k_bIsRandom) {
               val = testDistribution(testRandom);
            } else {
               val = k_testLowerInclusiveBound + iTest * k_movementTick;
               if(k_bIsRandomFinalFill) {
                  val += testDistribution(testRandom);
               }
            }

            double exactVal = std::log(val);
            double approxVal = LogApproxSchraudolph(val, addTerm);
            double error = approxVal - exactVal;
            avgError += error;
            avgAbsError += std::abs(error);
            avgSquareError += error * error;
            minError = std::min(error, minError);
            maxError = std::max(error, maxError);
         }

         avgAbsError /= k_cTests;
         avgError /= k_cTests;
         avgSquareError /= k_cTests;

#ifdef ENABLE_PRINTF
         printf(
            "TextLogApprox: %+.10lf, %+.10lf, %+.10lf, %+.10lf, %+.10lf, %+.8le %s%s\n",
            avgError,
            avgAbsError,
            avgSquareError,
            minError,
            maxError,
            addTerm,
            addTerm == termMid ? "*" : "",
            iStat == k_cStats - 1 ? "\n" : ""
         );
#endif // ENABLE_PRINTF

         // this is just to prevent the compiler for optimizing our code away on release
         debugRet += avgError;
      } else {
#ifdef ENABLE_PRINTF
         if(iStat == k_cStats - 1) {
            printf("\n");
         }
#endif // ENABLE_PRINTF
      }
      for(int i = 0; i < termStepDistance; ++i) {
         addTerm = std::nextafter(addTerm, std::numeric_limits<float>::max());
      }
   }

   // this is just to prevent the compiler for optimizing our code away on release
    return debugRet;
}
// this is just to prevent the compiler for optimizing our code away on release
extern double g_TestLogSumErrors = TestLogSumErrors();
#endif // ENABLE_TEST_LOG_SUM_ERRORS

#ifdef ENABLE_TEST_EXP_SUM_ERRORS
static double TestExpSumErrors() {
   double debugRet = 0; // this just prevents the optimizer from eliminating this code

   // no underflow to denormals
   EBM_ASSERT(!std::isnan(ExpApproxSchraudolph(k_expUnderflowPoint, k_expTermLowerBound)));
   EBM_ASSERT(std::numeric_limits<float>::min() <= ExpApproxSchraudolph(k_expUnderflowPoint, k_expTermLowerBound));

   // no underflow to infinity
   EBM_ASSERT(!std::isnan(ExpApproxSchraudolph(k_expOverflowPoint, k_expTermUpperBound)));
   EBM_ASSERT(ExpApproxSchraudolph(k_expOverflowPoint, k_expTermUpperBound) <= std::numeric_limits<float>::max());


   // no underflow to denormals
   EBM_ASSERT(!std::isnan(ExpApproxBest(k_expUnderflowPoint)));
   EBM_ASSERT(std::numeric_limits<float>::min() <= ExpApproxBest(k_expUnderflowPoint));

   // no underflow to infinity
   EBM_ASSERT(!std::isnan(ExpApproxBest(k_expOverflowPoint)));
   EBM_ASSERT(ExpApproxBest(k_expOverflowPoint) <= std::numeric_limits<float>::max());


   // our exp error has a periodicity of ln(2), so [0, ln(2)) should have the same relative error as 
   // [ln(2), 2 * ln(2)) OR [-ln(2), 0) OR [-ln(2)/2, +ln(2)/2) 
   // BUT, this needs to be evenly distributed, so we can't use nextafter. We need to increment with a constant.
   static constexpr double k_testLowerInclusiveBound = -k_expErrorPeriodicity / 2;
   static constexpr double k_testUpperExclusiveBound = k_expErrorPeriodicity / 2;
   static constexpr uint64_t k_cTests = 10000;
   static constexpr bool k_bIsRandom = false;
   static constexpr bool k_bIsRandomFinalFill = true; // if true we choose a random value to randomly fill the space between ticks
   static constexpr uint32_t termMid = k_expTermZeroMeanRelativeError;
   static constexpr uint32_t termStepsFromMid = 20;
   static constexpr uint32_t termStepDistance = 10;


   static constexpr ptrdiff_t k_cStats = termStepsFromMid * 2 + 1;
   static constexpr double k_movementTick = (k_testUpperExclusiveBound - k_testLowerInclusiveBound) / k_cTests;

   // uniform_real_distribution includes the lower bound, but not the upper bound, which is good because
   // our window is balanced by not including both ends
   std::uniform_real_distribution<double> testDistribution(k_bIsRandom ? k_testLowerInclusiveBound : double { 0 }, k_bIsRandom ? k_testUpperExclusiveBound : k_movementTick);
   std::mt19937 testRandom(52);

   for(ptrdiff_t iStat = 0; iStat < k_cStats; ++iStat) {
      const uint32_t addTerm = termMid - termStepsFromMid * termStepDistance + static_cast<uint32_t>(iStat) * termStepDistance;
      if(k_expTermLowerBound <= addTerm && addTerm <= k_expTermUpperBound) {
         double avgAbsRelativeError = 0;
         double avgRelativeError = 0;
         double avgSquareRelativeError = 0;
         double minRelativeError = std::numeric_limits<double>::max();
         double maxRelativeError = std::numeric_limits<double>::lowest();

         for(uint64_t iTest = 0; iTest < k_cTests; ++iTest) {
            double val;
            if(k_bIsRandom) {
               val = testDistribution(testRandom);
            } else {
               val = k_testLowerInclusiveBound + iTest * k_movementTick;
               if(k_bIsRandomFinalFill) {
                  val += testDistribution(testRandom);
               }
            }

            double exactVal = std::exp(val);
            double approxVal = ExpApproxSchraudolph<false, false, false, false>(val, addTerm);
            double error = approxVal - exactVal;
            double relativeError = error / exactVal;
            avgRelativeError += relativeError;
            avgAbsRelativeError += std::abs(relativeError);
            avgSquareRelativeError += relativeError * relativeError;
            minRelativeError = std::min(relativeError, minRelativeError);
            maxRelativeError = std::max(relativeError, maxRelativeError);
         }

         avgAbsRelativeError /= k_cTests;
         avgRelativeError /= k_cTests;
         avgSquareRelativeError /= k_cTests;

#ifdef ENABLE_PRINTF
         printf(
            "TextExpApprox: %+.10lf, %+.10lf, %+.10lf, %+.10lf, %+.10lf, %d %s%s\n", 
            avgRelativeError, 
            avgAbsRelativeError, 
            avgSquareRelativeError, 
            minRelativeError, 
            maxRelativeError, 
            addTerm, 
            addTerm == termMid ? "*" : "",
            iStat == k_cStats - 1 ? "\n" : ""
         );
#endif // ENABLE_PRINTF

         // this is just to prevent the compiler for optimizing our code away on release
         debugRet += avgRelativeError;
      } else {
#ifdef ENABLE_PRINTF
         if(iStat == k_cStats - 1) {
            printf("\n");
         }
#endif // ENABLE_PRINTF
      }
   }

   // this is just to prevent the compiler for optimizing our code away on release
   return debugRet;
}
// this is just to prevent the compiler for optimizing our code away on release
extern double g_TestExpSumErrors = TestExpSumErrors();
#endif // ENABLE_TEST_EXP_SUM_ERRORS

#ifdef ENABLE_TEST_SOFTMAX_SUM_ERRORS
static double TestSoftmaxSumErrors() {
   double debugRet = 0; // this just prevents the optimizer from eliminating this code

   static constexpr unsigned int seed = 572422;

   static constexpr double k_expWindowSkew = 0;
   static constexpr int expWindowMultiple = 10;
   static_assert(1 <= expWindowMultiple, "window must have a positive non-zero size");

   static constexpr bool k_bIsRandom = true;
   static constexpr bool k_bIsRandomFinalFill = true; // if true we choose a random value to randomly fill the space between ticks
   static constexpr uint64_t k_cTests = uint64_t { 10000000 }; // std::numeric_limits<uint64_t>::max()
   static constexpr uint64_t k_outputPeriodicity = uint64_t { 100000000 };
   static constexpr uint64_t k_cDivisions = 1609; // ideally choose a prime number
   static constexpr ptrdiff_t k_cSoftmaxTerms = 3;
   static_assert(2 <= k_cSoftmaxTerms, "can't have just 1 since that's always 100% chance");
   static constexpr ptrdiff_t iEliminateOneTerm = 0;
   static_assert(iEliminateOneTerm < k_cSoftmaxTerms, "can't eliminate a term above our existing terms");
   static constexpr uint32_t termMid = k_expTermZeroMeanErrorForSoftmaxWithZeroedLogit;
   static constexpr uint32_t termStepsFromMid = 0;
   static constexpr uint32_t termStepDistance = 1;


   // below here are calculated values dependent on the above settings

   // our exp error has a periodicity of ln(2), so [0, ln(2)) should have the same relative error as 
   // [ln(2), 2 * ln(2)) OR [-ln(2), 0) OR [-ln(2)/2, +ln(2)/2) 
   // BUT, this needs to be evenly distributed, so we can't use nextafter. We need to increment with a constant.
   static constexpr double k_testLowerInclusiveBound = 
      k_expWindowSkew - static_cast<double>(expWindowMultiple) * k_expErrorPeriodicity / 2;
   static constexpr double k_testUpperExclusiveBound = 
      k_expWindowSkew + static_cast<double>(expWindowMultiple) * k_expErrorPeriodicity / 2;
   static constexpr ptrdiff_t k_cStats = termStepsFromMid * 2 + 1;
   static constexpr double k_movementTick = (k_testUpperExclusiveBound - k_testLowerInclusiveBound) / k_cDivisions;

   static_assert(k_testLowerInclusiveBound < k_testUpperExclusiveBound, "low must be lower than high");

   static_assert(k_expUnderflowPoint <= k_testLowerInclusiveBound, "outside exp bounds");
   static_assert(k_testLowerInclusiveBound <= k_expOverflowPoint, "outside exp bounds");
   static_assert(k_expUnderflowPoint <= k_testUpperExclusiveBound, "outside exp bounds");
   static_assert(k_testUpperExclusiveBound <= k_expOverflowPoint, "outside exp bounds");

   // uniform_real_distribution includes the lower bound, but not the upper bound, which is good because
   // our window is balanced by not including both ends
   std::uniform_real_distribution<double> testDistribution(
      k_bIsRandom ? k_testLowerInclusiveBound : double { 0 }, k_bIsRandom ? k_testUpperExclusiveBound : k_movementTick);
   std::mt19937 testRandom(seed);

   double softmaxTerms[k_cSoftmaxTerms];

   double avgAbsRelativeError[k_cStats];
   double avgRelativeError[k_cStats];
   double avgSquareRelativeError[k_cStats];
   double minRelativeError[k_cStats];
   double maxRelativeError[k_cStats];

   for(ptrdiff_t iStat = 0; iStat < k_cStats; ++iStat) {
      avgAbsRelativeError[iStat] = 0;
      avgRelativeError[iStat] = 0;
      avgSquareRelativeError[iStat] = 0;
      minRelativeError[iStat] = std::numeric_limits<double>::max();
      maxRelativeError[iStat] = std::numeric_limits<double>::lowest();
   }

   uint64_t aIndexes[k_cSoftmaxTerms];
   for(ptrdiff_t iTerm = 0; iTerm < k_cSoftmaxTerms; ++iTerm) {
      softmaxTerms[iTerm] = 0;
      aIndexes[iTerm] = 0;
   }

   uint64_t iTest = 0;
   while(true) {
      for(ptrdiff_t iStat = 0; iStat < k_cStats; ++iStat) {
         const uint32_t addTerm = termMid - termStepsFromMid * termStepDistance + static_cast<uint32_t>(iStat) * termStepDistance;
         if(k_expTermLowerBound <= addTerm && addTerm <= k_expTermUpperBound) {
            for(ptrdiff_t iTerm = 0; iTerm < k_cSoftmaxTerms; ++iTerm) {
               if(iTerm != iEliminateOneTerm) {
                  if(k_bIsRandom) {
                     softmaxTerms[iTerm] = testDistribution(testRandom);
                  } else {
                     softmaxTerms[iTerm] = k_testLowerInclusiveBound + aIndexes[iTerm] * k_movementTick;
                     if(k_bIsRandomFinalFill) {
                        softmaxTerms[iTerm] += testDistribution(testRandom);
                     }
                  }
               }
            }

            const double exactNumerator = 0 == iEliminateOneTerm ? double { 1 } : std::exp(softmaxTerms[0]);
            double exactDenominator = 0;
            for(ptrdiff_t iTerm = 0; iTerm < k_cSoftmaxTerms; ++iTerm) {
               const double oneTermAdd = iTerm == iEliminateOneTerm ? double { 1 } : std::exp(softmaxTerms[iTerm]);
               exactDenominator += oneTermAdd;
            }
            const double exactVal = exactNumerator / exactDenominator;


            const double approxNumerator = 0 == iEliminateOneTerm ? double { 1 } : ExpApproxBest<false, false, false, false>(softmaxTerms[0]);
            double approxDenominator = 0;
            for(ptrdiff_t iTerm = 0; iTerm < k_cSoftmaxTerms; ++iTerm) {
               const double oneTermAdd = iTerm == iEliminateOneTerm ? double { 1 } : ExpApproxBest<false, false, false, false>(softmaxTerms[iTerm]);
               approxDenominator += oneTermAdd;
            }
            const double approxVal = approxNumerator / approxDenominator;

            const double error = approxVal - exactVal;
            const double relativeError = error / exactVal;
            avgRelativeError[iStat] += relativeError;
            avgAbsRelativeError[iStat] += std::abs(relativeError);
            avgSquareRelativeError[iStat] += relativeError * relativeError;
            minRelativeError[iStat] = std::min(relativeError, minRelativeError[iStat]);
            maxRelativeError[iStat] = std::max(relativeError, maxRelativeError[iStat]);

            // this is just to prevent the compiler for optimizing our code away on release
            debugRet += relativeError;
         }
      }

      bool bDone = true;
      ++iTest;
      if(k_bIsRandom) {
         if(iTest < k_cTests) {
            bDone = false;
         }
      } else {
         for(ptrdiff_t iTerm = 0; iTerm < k_cSoftmaxTerms; ++iTerm) {
            if(iTerm != iEliminateOneTerm) {
               ++aIndexes[iTerm];
               if(aIndexes[iTerm] == k_cDivisions) {
                  aIndexes[iTerm] = 0;
               } else {
                  bDone = false;
                  break;
               }
            }
         }
      }

      if(bDone || (0 < k_outputPeriodicity && 0 == iTest % k_outputPeriodicity)) {
         for(ptrdiff_t iStat = 0; iStat < k_cStats; ++iStat) {
            const uint32_t addTerm = termMid - termStepsFromMid * termStepDistance + static_cast<uint32_t>(iStat) * termStepDistance;
            if(k_expTermLowerBound <= addTerm && addTerm <= k_expTermUpperBound) {
#ifdef ENABLE_PRINTF
               printf(
                  "TextSoftmaxApprox: %+.10lf, %+.10lf, %+.10lf, %+.10lf, %+.10lf, %d %s%s\n",
                  avgRelativeError[iStat] / iTest,
                  avgAbsRelativeError[iStat] / iTest,
                  avgSquareRelativeError[iStat] / iTest,
                  minRelativeError[iStat],
                  maxRelativeError[iStat],
                  addTerm,
                  addTerm == termMid ? "*" : "",
                  iStat == k_cStats - 1 ? "\n" : ""
               );
#endif // ENABLE_PRINTF
            } else {
#ifdef ENABLE_PRINTF
               if(iStat == k_cStats - 1) {
                  printf("\n");
               }
#endif // ENABLE_PRINTF
            }
         }
      }

      if(bDone) {
         break;
      }
   }

   // this is just to prevent the compiler for optimizing our code away on release
   return debugRet;
}
// this is just to prevent the compiler for optimizing our code away on release
extern double g_TestSoftmaxSumErrors = TestSoftmaxSumErrors();
#endif // ENABLE_TEST_SOFTMAX_SUM_ERRORS

#endif // !defined(NDEBUG) || defined(INCLUDE_TESTS_IN_RELEASE)

} // DEFINED_ZONE_NAME
