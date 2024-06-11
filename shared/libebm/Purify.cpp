// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// Purification algorithm from: https://arxiv.org/abs/1911.04974
//@article {lengerich2019purifying,
//  title={Purifying Interaction Effects with the Functional ANOVA: An Efficient Algorithm for Recovering Identifiable Additive Models},
//  author={Lengerich, Benjamin and Tan, Sarah and Chang, Chun-Hao and Hooker, Giles and Caruana, Rich},
//  journal={arXiv preprint arXiv:1911.04974},
//  year={2019}
//}

#include "pch.hpp"

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t
#include <string.h> // memcpy
#include <cmath> // std::abs

#define ZONE_main
#include "zones.h"
#include "bridge.hpp" // k_dynamicScores

#include "RandomDeterministic.hpp" // RandomDeterministic

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

EBM_API_BODY double EBM_CALLING_CONVENTION MeasureImpurity(IntEbm countMultiScores,
      IntEbm indexMultiScore,
      IntEbm countDimensions,
      const IntEbm* dimensionLengths,
      const double* weights,
      const double* scores) {
   LOG_N(Trace_Info,
         "Entered MeasureImpurity: "
         "countMultiScores=%" IntEbmPrintf ", "
         "indexMultiScore=%" IntEbmPrintf ", "
         "countDimensions=%" IntEbmPrintf ", "
         "dimensionLengths=%p, "
         "weights=%p, "
         "scores=%p",
         countMultiScores,
         indexMultiScore,
         countDimensions,
         static_cast<const void*>(dimensionLengths),
         static_cast<const void*>(weights),
         static_cast<const void*>(scores));

   if(countMultiScores <= IntEbm{0}) {
      if(IntEbm{0} == countMultiScores) {
         LOG_0(Trace_Info, "INFO MeasureImpurity zero scores");
         return 0.0;
      } else {
         LOG_0(Trace_Error, "ERROR MeasureImpurity countMultiScores must be positive");
         return double{Error_IllegalParamVal};
      }
   }
   if(IsConvertError<size_t>(countMultiScores)) {
      LOG_0(Trace_Error, "ERROR MeasureImpurity IsConvertError<size_t>(countMultiScores)");
      return double{Error_IllegalParamVal};
   }
   const size_t cScores = static_cast<size_t>(countMultiScores);
   if(IsMultiplyError(sizeof(double), cScores)) {
      LOG_0(Trace_Error, "ERROR MeasureImpurity IsMultiplyError(sizeof(double), cScores)");
      return double{Error_IllegalParamVal};
   }

   if(countMultiScores <= indexMultiScore) {
      LOG_0(Trace_Error, "ERROR MeasureImpurity countMultiScores <= indexMultiScore");
      return double{Error_IllegalParamVal};
   }
   if(indexMultiScore < IntEbm{0}) {
      LOG_0(Trace_Error, "ERROR MeasureImpurity indexMultiScore must be positive");
      return double{Error_IllegalParamVal};
   }
   const size_t iScore = static_cast<size_t>(indexMultiScore);

   if(countDimensions <= IntEbm{0}) {
      if(IntEbm{0} == countDimensions) {
         LOG_0(Trace_Info, "INFO MeasureImpurity zero dimensions");
         return 0.0;
      } else {
         LOG_0(Trace_Error, "ERROR MeasureImpurity countDimensions must be positive");
         return double{Error_IllegalParamVal};
      }
   }
   if(IntEbm{k_cDimensionsMax} < countDimensions) {
      LOG_0(Trace_Warning, "WARNING MeasureImpurity countDimensions too large and would cause out of memory condition");
      return double{Error_IllegalParamVal};
   }
   const size_t cDimensions = static_cast<size_t>(countDimensions);

   if(nullptr == dimensionLengths) {
      LOG_0(Trace_Error, "ERROR MeasureImpurity nullptr == dimensionLengths");
      return double{Error_IllegalParamVal};
   }

   bool bZero = false;
   size_t iDimension = 0;
   do {
      const IntEbm dimensionsLength = dimensionLengths[iDimension];
      if(dimensionsLength <= IntEbm{0}) {
         if(dimensionsLength < IntEbm{0}) {
            LOG_0(Trace_Error, "ERROR MeasureImpurity dimensionsLength value cannot be negative");
            return double{Error_IllegalParamVal};
         }
         bZero = true;
      }
      ++iDimension;
   } while(cDimensions != iDimension);
   if(bZero) {
      LOG_0(Trace_Info, "INFO MeasureImpurity empty tensor");
      return 0.0;
   }

   iDimension = 0;
   size_t cTensorBins = 1;
   size_t aDimensionLengths[k_cDimensionsMax];
   do {
      const IntEbm dimensionsLength = dimensionLengths[iDimension];
      EBM_ASSERT(IntEbm{1} <= dimensionsLength);
      if(IsConvertError<size_t>(dimensionsLength)) {
         // the scores tensor could not exist with this many tensor bins, so it is an error
         LOG_0(Trace_Error, "ERROR MeasureImpurity IsConvertError<size_t>(dimensionsLength)");
         return double{Error_IllegalParamVal};
      }
      const size_t cBins = static_cast<size_t>(dimensionsLength);
      aDimensionLengths[iDimension] = cBins;

      if(IsMultiplyError(cTensorBins, cBins)) {
         // the scores tensor could not exist with this many tensor bins, so it is an error
         LOG_0(Trace_Error, "ERROR MeasureImpurity IsMultiplyError(cTensorBins, cBins)");
         return double{Error_IllegalParamVal};
      }
      cTensorBins *= cBins;

      ++iDimension;
   } while(cDimensions != iDimension);
   EBM_ASSERT(1 <= cTensorBins);

   if(nullptr == weights) {
      LOG_0(Trace_Error, "ERROR MeasureImpurity nullptr == weights");
      return double{Error_IllegalParamVal};
   }

   if(nullptr == scores) {
      LOG_0(Trace_Error, "ERROR MeasureImpurity nullptr == scoresInOut");
      return double{Error_IllegalParamVal};
   }

   // shift to the proper class
   scores = &scores[iScore];

   const size_t* const pDimensionLengthsEnd = &aDimensionLengths[cDimensions];
   const size_t* pDimensionLength = aDimensionLengths;
   size_t cTensorWeightIncrement = sizeof(double);
   double impurityTotal = 0.0;
   size_t iTensorWeight = 0;
   do {
      const size_t cBins = *pDimensionLength;
      size_t iSurfaceBin = 0;

      const size_t cTensorScoreIncrement = cTensorWeightIncrement * cScores;
      const size_t cBytesEnd = cTensorScoreIncrement * cBins;

      EBM_ASSERT(0 == iTensorWeight);
      do {
      next:;
         size_t iTensorWeightCur = iTensorWeight;
         size_t iTensorScoreCur = iTensorWeight * cScores;
         const size_t iTensorEnd = iTensorScoreCur + cBytesEnd;
         double impurityCur = 0.0;
         double weightTotal = 0.0;
         do {
            const double weight = *IndexByte(weights, iTensorWeightCur);
            const double score = *IndexByte(scores, iTensorScoreCur);
            weightTotal += weight;
            impurityCur += weight * score;
            iTensorWeightCur += cTensorWeightIncrement;
            iTensorScoreCur += cTensorScoreIncrement;
         } while(iTensorEnd != iTensorScoreCur);

         impurityCur = impurityCur / weightTotal;
         impurityTotal += std::abs(impurityCur);

         ++iSurfaceBin;
         size_t iSurfaceBinDeconstruct = iSurfaceBin;

         size_t cSurfaceWeightIncrement = sizeof(double);
         const size_t* pDimensionLengthInternal = aDimensionLengths;
         do {
            const size_t cBinsInternal = *pDimensionLengthInternal;
            const size_t cSurfaceWeightIncrementNext = cSurfaceWeightIncrement * cBinsInternal;
            if(pDimensionLengthInternal != pDimensionLength) {
               iTensorWeight += cSurfaceWeightIncrement;
               if(0 != iSurfaceBinDeconstruct % cBinsInternal) {
                  goto next;
               }
               iSurfaceBinDeconstruct /= cBinsInternal;
               iTensorWeight -= cSurfaceWeightIncrementNext;
            }
            cSurfaceWeightIncrement = cSurfaceWeightIncrementNext;
            ++pDimensionLengthInternal;
         } while(pDimensionLengthsEnd != pDimensionLengthInternal);
      } while(false);
      EBM_ASSERT(0 == iTensorWeight);

      cTensorWeightIncrement *= cBins;

      ++pDimensionLength;
   } while(pDimensionLengthsEnd != pDimensionLength);

   return impurityTotal;
}


static ErrorEbm PurifyInternal(RandomDeterministic &rng,
      const double tolerance,
      const size_t cScores,
      const size_t cTensorBins,
      const size_t cSurfaceBins,
      size_t* const aRandomize,
      const size_t* const aDimensionLengths,
      const double* const aWeights,
      double* const aScoresInOut,
      double* const aImpuritiesOut,
      double* const aInterceptOut) {
   EBM_ASSERT(!std::isnan(tolerance));
   EBM_ASSERT(!std::isinf(tolerance));
   EBM_ASSERT(0.0 <= tolerance);
   EBM_ASSERT(1 <= cScores);
   EBM_ASSERT(1 <= cTensorBins);
   EBM_ASSERT(nullptr != aDimensionLengths);
   EBM_ASSERT(nullptr != aWeights);
   EBM_ASSERT(nullptr != aScoresInOut);

   const size_t cBytesScoreClasses = sizeof(double) * cScores;

   if(nullptr != aImpuritiesOut) {
      memset(aImpuritiesOut, 0, cBytesScoreClasses * cSurfaceBins);
   }

   const size_t iScoresEnd = cBytesScoreClasses * cTensorBins;
   const double* const pWeightsEnd = &aWeights[cTensorBins];
   const double* const pScoreMulticlassEnd = &aScoresInOut[cScores];

   double* pScores = aScoresInOut;
   double* pImpurities = aImpuritiesOut;
   double* pIntercept = aInterceptOut;
   do {
      EBM_ASSERT(nullptr == pIntercept || 0.0 == *pIntercept);

      double impurityMax = 0.0;
      double impurityTotalPre = 0.0;
      double weightTotalPre = 0.0;

      double factorPre = 1.0;

      size_t iScorePre = 0;
      const double* pWeightPre = aWeights;
      do {
         const double weight = *pWeightPre;
         if(!(0.0 <= weight)) {
            LOG_0(Trace_Error, "ERROR PurifyInternal weight cannot be negative or NaN");
            return Error_IllegalParamVal;
         }
         EBM_ASSERT(!std::isnan(weight)); // !(0.0 <= weight) above checks for NaN
         if(std::numeric_limits<double>::infinity() == weight) {
            size_t cInfWeights;
            goto skip_multiply_intercept;
            do {
               factorPre *= 0.5;
               // there should be a factor that will allow us to succeed before this
               EBM_ASSERT(std::numeric_limits<double>::min() <= factorPre);
            skip_multiply_intercept:;

               size_t iScoreInterior = iScorePre;
               const double* pWeightInterior = pWeightPre;
               impurityTotalPre = 0.0;
               cInfWeights = 0;
               do {
                  const double weightInterior = *pWeightInterior;
                  if(!(0.0 <= weightInterior)) {
                     LOG_0(Trace_Error, "ERROR PurifyInternal weight cannot be negative or NaN");
                     return Error_IllegalParamVal;
                  }
                  EBM_ASSERT(!std::isnan(weightInterior)); // !(0.0 <= weightInterior) above checks for NaN
                  if(std::numeric_limits<double>::infinity() == weightInterior) {
                     const double scoreInterior = *IndexByte(pScores, iScoreInterior);
                     if(!std::isnan(scoreInterior) && !std::isinf(scoreInterior)) {
                        ++cInfWeights;
                        impurityTotalPre += factorPre * scoreInterior;

                        // impurityTotalPre can reach -+inf, but once it gets there it cannot
                        // escape that value because everything we add subsequently is non-inf.
                        EBM_ASSERT(!std::isnan(impurityTotalPre));
                     }
                  }
                  iScoreInterior += cBytesScoreClasses;
                  ++pWeightInterior;
               } while(pWeightsEnd != pWeightInterior);
            } while(std::isinf(impurityTotalPre));
            // turn off early exiting based on tolerance
            impurityMax = 0.0;
            weightTotalPre = static_cast<double>(cInfWeights);
            // cInfWeights can be zero if all the +inf weights are +-inf or NaN scores, so goto a check for this
            goto pre_intercept;
         }
         const double score = *IndexByte(pScores, iScorePre);
         if(!std::isnan(score) && !std::isinf(score)) {
            weightTotalPre += weight;
            const double impurity = weight * score;
            impurityTotalPre += impurity;
            impurityMax += std::abs(impurity);
         }
         iScorePre += cBytesScoreClasses;
         ++pWeightPre;
      } while(pWeightsEnd != pWeightPre);
      EBM_ASSERT(!std::isnan(weightTotalPre));
      EBM_ASSERT(0.0 <= weightTotalPre);

      // even if score and weight are never NaN or infinity, the product of both can be +-inf and if +inf is added
      // to -inf it will be NaN, so impurityTotalPre can be NaN even if weight and score are well formed
      // impurityMax cannot be NaN though here since abs(impurity) cannot be -inf. If score was zero and weight
      // was +inf then impurityMax could be NaN, but we've excluded it at this point through bInfWeight.
      EBM_ASSERT(!std::isnan(impurityMax));
      EBM_ASSERT(0.0 <= impurityMax);

      if(std::numeric_limits<double>::min() <= impurityMax) {
         // impurityMax could be zero because the scores are zero or the weights are zero.
         // Either way, it's already pure, so no work needed.
         
         while(std::isnan(impurityTotalPre) || std::isinf(impurityTotalPre) || std::isinf(weightTotalPre)) {
            // If impurity is NaN, it means that score * weight overflowed to +inf once and -inf another time

            // In IEEE-754 this is an exact operation and should loose no information unless it underflows
            factorPre *= 0.5;
            // there should be a factor that will allow us to succeed before this
            EBM_ASSERT(std::numeric_limits<double>::min() <= factorPre);
            impurityTotalPre = 0.0;
            weightTotalPre = 0.0;
            size_t iScoreRetry = 0;
            const double* pWeightRetry = aWeights;
            do {
               const double weight = *pWeightRetry;
               EBM_ASSERT(!std::isnan(weight) && 0.0 <= weight && weight != std::numeric_limits<double>::infinity());
               const double score = *IndexByte(pScores, iScoreRetry);
               if(!std::isnan(score) && !std::isinf(score)) {
                  const double weightTimesFactor = factorPre * weight;
                  weightTotalPre += weightTimesFactor;
                  const double scoreTimesFactor = factorPre * score;
                  impurityTotalPre += scoreTimesFactor * weightTimesFactor;
               }
               iScoreRetry += cBytesScoreClasses;
               ++pWeightRetry;
            } while(pWeightsEnd != pWeightRetry);
         }
         EBM_ASSERT(!std::isnan(weightTotalPre));
         EBM_ASSERT(0.0 <= weightTotalPre);

      pre_intercept:;
         if(std::numeric_limits<double>::min() <= weightTotalPre) {
            if(std::numeric_limits<double>::infinity() == impurityMax) {
               // handle this by just turning off early exit and let the algorithm exit when it cannot improve
               impurityMax = 0.0;
            } else {
               impurityMax = impurityMax * tolerance * factorPre / weightTotalPre;
               // at this location:
               //   0.0 < impurityMax < +inf
               //   0.0 <= tolerance < +inf
               //   0.0 < factorPre <= 1.0
               //   0.0 < weightTotalPre < +inf
               //   impurityMax * tolerance can overflow to +inf, but dividing by a non-NaN, non-inf, non-zero number is
               //   non-NaN
               EBM_ASSERT(!std::isnan(impurityMax));
               if(std::numeric_limits<double>::infinity() == impurityMax) {
                  // handle this by just turning off early exit and let the algorithm exit when it cannot improve
                  impurityMax = 0.0;
               }
            }

            if(nullptr != pIntercept) {
               // pull out the intercept early since this will make purification easier
               double intercept = impurityTotalPre / weightTotalPre / factorPre;
               EBM_ASSERT(!std::isnan(intercept));
               if(std::isinf(intercept)) {
                  // the intercept is the weighted average of numbers, so it cannot mathematically
                  // be larger than the largest number, and we checked that the sum of those numbers
                  // did not overflow, so this shouldn't be possible without floating point noise
                  // If it does happen, the real value must be very very close to +-max_float,
                  // so use that instead

                  if(std::numeric_limits<double>::infinity() == intercept) {
                     intercept = std::numeric_limits<double>::max();
                  } else {
                     EBM_ASSERT(-std::numeric_limits<double>::infinity() == intercept);
                     intercept = -std::numeric_limits<double>::max();
                  }
               }

               *pIntercept = intercept;
               const double interceptNeg = -intercept;
               size_t iScoreUpdate = 0;
               do {
                  // this can create new +-inf values, but not NaN since we limited intercept to non-NaN, non-inf
                  double* const pScoreUpdate = IndexByte(pScores, iScoreUpdate);
                  const double scoreOld = *pScoreUpdate;
                  const double scoreNew = scoreOld + interceptNeg;
                  EBM_ASSERT(std::isnan(scoreOld) || !std::isnan(scoreNew));
                  *pScoreUpdate = scoreNew;
                  iScoreUpdate += cBytesScoreClasses;
               } while(iScoresEnd != iScoreUpdate);
            }

            if(size_t{0} != cSurfaceBins) {
               // this only happens for 1 dimensional inputs. Exit after finding the intercept

               // this prevents impurityCur from overflowing since the individual terms we add cannot sum to infinity
               // so as long as we multiply by 1/number_of_terms_summed we can guarantee the sum will not overflow
               // start from 0.5 instead of 1.0 to allow for floating point error.
               const double impuritySumOverflowPreventer = 0.5 / static_cast<double>(cSurfaceBins);
               double impurityPrev;
               double impurityCur = std::numeric_limits<double>::infinity();
               bool bRetry;
               do {
                  // if any non-infinite value was flipped to an infinite value, it could increase the impurity
                  // so we set impurityCur to NaN. We need to reset it to +inf to avoid stopping early
                  impurityCur = std::isnan(impurityCur) ? std::numeric_limits<double>::infinity() : impurityCur;
                  impurityPrev = impurityCur;
                  impurityCur = 0.0;
                  bRetry = false;

                  if(nullptr != aRandomize) {
                     // TODO: We're currently generating different randomized ordering for each class when doing multiclass
                     // This might cause problems during boosting, especially if we use a non-zero tolerance because then there 
                     // are small impurities that creep in and would change the multiclass predictions on a repetitive process 
                     // like boosting. We could change this to use the same random order for all classes by looping over 
                     // the classes after choosing the random order, like we do in PurifyNormalizedMulticlass.

                     size_t cRemaining = cSurfaceBins;
                     do {
                        --cRemaining;
                        aRandomize[cRemaining] = cRemaining;
                     } while(size_t{0} != cRemaining);

                     cRemaining = cSurfaceBins;
                     do {
                        const size_t iSwap = rng.NextFast(cRemaining);
                        const size_t iOriginal = aRandomize[iSwap];
                        --cRemaining;
                        aRandomize[iSwap] = aRandomize[cRemaining];
                        aRandomize[cRemaining] = iOriginal;
                     } while(size_t{0} != cRemaining);
                  }

                  size_t iRandom = 0;
                  do {
                     size_t iSurfaceBin = iRandom;
                     if(nullptr != aRandomize) {
                        iSurfaceBin = aRandomize[iRandom];
                     }

                     size_t cTensorWeightIncrement = sizeof(double);
                     size_t cSweepBins;
                     size_t iDimensionSurfaceBin = iSurfaceBin;
                     const size_t* pSweepingDimensionLength = aDimensionLengths;
                     while(true) {
                        cSweepBins = *pSweepingDimensionLength;
                        EBM_ASSERT(1 <= cSweepBins);
                        EBM_ASSERT(0 == cTensorBins % cSweepBins);
                        size_t cSurfaceBinsExclude = cTensorBins / cSweepBins;
                        if(iDimensionSurfaceBin < cSurfaceBinsExclude) {
                           // we've found it
                           break;
                        }
                        iDimensionSurfaceBin -= cSurfaceBinsExclude;
                        cTensorWeightIncrement *= cSweepBins;
                        ++pSweepingDimensionLength;
                     }
                     size_t cTensorScoreIncrement = cTensorWeightIncrement * cScores;

                     if(size_t{1} != cSweepBins && cTensorBins != cSweepBins) {
                        // If cSweepBins is cTensorBins, then all other dimensions are length 1 and the
                        // surface is the same as the intercept.
                        //
                        // If cSweepBins is 1, then the entire tensor could be pushed down to a lower reduced
                        // dimension however, we should keep the scores in this current tensor because we could
                        // have more than 1 dimension of length 1, and then it would be ambiguous
                        // and random which dimension we should push scores to.  Also, this should only occur
                        // when the user specifies an interaction since our automatic interaction detection
                        // will not select interactions with a feature of 1 bin. If the user specifies an
                        // interaction with a useless bin length of 1, then that is what they'll get back.

                        size_t iTensorWeight = 0;
                        size_t multiple = sizeof(double);
                        const size_t* pDimensionLength = aDimensionLengths;
                        while(size_t{0} != iDimensionSurfaceBin) {
                           const size_t cBins = *pDimensionLength;
                           EBM_ASSERT(1 <= cBins);
                           if(pDimensionLength != pSweepingDimensionLength) {
                              const size_t iBin = iDimensionSurfaceBin % cBins;
                              iDimensionSurfaceBin /= cBins;
                              iTensorWeight += iBin * multiple;
                           }
                           multiple *= cBins;
                           ++pDimensionLength;
                        }
                        size_t iTensorScore = iTensorWeight * cScores;

                        double factor = 1.0;
                        const size_t iTensorEnd = iTensorScore + cTensorScoreIncrement * cSweepBins;
                        double impurity = 0.0;
                        double weightTotal = 0.0;
                        size_t iTensorWeightCur = iTensorWeight;
                        size_t iTensorScoreCur = iTensorScore;
                        do {
                           const double weight = *IndexByte(aWeights, iTensorWeightCur);
                           EBM_ASSERT(!std::isnan(weight) && 0.0 <= weight);
                           if(std::numeric_limits<double>::infinity() == weight) {
                              size_t cInfWeights;
                              goto skip_multiply;
                              do {
                                 factor *= 0.5;
                                 // there should be a factor that will allow us to succeed before this
                                 EBM_ASSERT(std::numeric_limits<double>::min() <= factor);
                              skip_multiply:;
                                 size_t iTensorWeightCurInterior = iTensorWeightCur;
                                 size_t iTensorScoreCurInterior = iTensorScoreCur;
                                 impurity = 0.0;
                                 cInfWeights = 0;
                                 do {
                                    const double weightInterior = *IndexByte(aWeights, iTensorWeightCurInterior);
                                    EBM_ASSERT(!std::isnan(weightInterior) && 0.0 <= weightInterior);
                                    if(std::numeric_limits<double>::infinity() == weightInterior) {
                                       const double scoreInterior = *IndexByte(pScores, iTensorScoreCurInterior);
                                       if(!std::isnan(scoreInterior) && !std::isinf(scoreInterior)) {
                                          ++cInfWeights;
                                          impurity += factor * scoreInterior;

                                          // impurity can reach -+inf, but once it gets there it cannot
                                          // escape that value because everything we add subsequently is non-inf.
                                          EBM_ASSERT(!std::isnan(impurity));
                                       }
                                    }
                                    iTensorWeightCurInterior += cTensorWeightIncrement;
                                    iTensorScoreCurInterior += cTensorScoreIncrement;
                                 } while(iTensorEnd != iTensorScoreCurInterior);
                              } while(std::isinf(impurity));
                              weightTotal = static_cast<double>(cInfWeights);
                              // cInfWeights can be zero if all the +inf weights are +-inf or NaN scores, so goto a
                              // check for this
                              goto do_impurity;
                           }
                           const double score = *IndexByte(pScores, iTensorScoreCur);
                           if(!std::isnan(score) && !std::isinf(score)) {
                              weightTotal += weight;
                              impurity += weight * score;
                           }
                           iTensorWeightCur += cTensorWeightIncrement;
                           iTensorScoreCur += cTensorScoreIncrement;
                        } while(iTensorEnd != iTensorScoreCur);

                        while(std::isnan(impurity) || std::isinf(impurity) || std::isinf(weightTotal)) {
                           // if impurity is NaN, it means that score * weight overflowed to +inf once and -inf another
                           // time

                           // in IEEE-754 this is an exact operation and should loose no information unless it
                           // underflows
                           factor *= 0.5;
                           // there should be a factor that will allow us to succeed before this
                           EBM_ASSERT(std::numeric_limits<double>::min() <= factor);
                           impurity = 0.0;
                           weightTotal = 0.0;
                           iTensorWeightCur = iTensorWeight;
                           iTensorScoreCur = iTensorScore;
                           do {
                              const double weight = *IndexByte(aWeights, iTensorWeightCur);
                              EBM_ASSERT(!std::isnan(weight) && 0.0 <= weight &&
                                    weight != std::numeric_limits<double>::infinity());
                              const double score = *IndexByte(pScores, iTensorScoreCur);
                              if(!std::isnan(score) && !std::isinf(score)) {
                                 const double weightTimesFactor = factor * weight;
                                 weightTotal += weightTimesFactor;
                                 const double scoreTimesFactor = factor * score;
                                 impurity += scoreTimesFactor * weightTimesFactor;
                              }
                              iTensorWeightCur += cTensorWeightIncrement;
                              iTensorScoreCur += cTensorScoreIncrement;
                           } while(iTensorEnd != iTensorScoreCur);
                        }

                     do_impurity:;

                        if(std::numeric_limits<double>::min() <= weightTotal) {
                           impurity = impurity / weightTotal / factor;
                           EBM_ASSERT(!std::isnan(impurity));
                           if(std::isinf(impurity)) {
                              // impurity is the weighted average of numbers, so it cannot mathematically
                              // be larger than the largest number, and we checked that the sum of those numbers
                              // did not overflow, so this shouldn't be possible without floating point noise.
                              // If it does happen, the real value must be very very close to +-max_float,
                              // so use that instead.

                              if(std::numeric_limits<double>::infinity() == impurity) {
                                 impurity = std::numeric_limits<double>::max();
                              } else {
                                 EBM_ASSERT(-std::numeric_limits<double>::infinity() == impurity);
                                 impurity = -std::numeric_limits<double>::max();
                              }
                           }

                           if(nullptr != pImpurities) {
                              double* const pImpurity = IndexByte(pImpurities, iSurfaceBin * cBytesScoreClasses);
                              const double oldImpurity = *pImpurity;
                              // we prevent impurities from reaching NaN or an infinity
                              EBM_ASSERT(!std::isnan(oldImpurity));
                              EBM_ASSERT(!std::isinf(oldImpurity));
                              double newImpurity = oldImpurity + impurity;
                              if(std::isinf(newImpurity)) {
                                 // There should be a solution that allows any tensor to be purified
                                 // without overflowing any of the impurity cells, however due to the
                                 // random ordering that we process cells, they could temporarily overflow
                                 // to +-inf, so limit the change to something that doesn't overflow to
                                 // allow the weight to be transfered elsewhere.

                                 if(std::numeric_limits<double>::infinity() == newImpurity) {
                                    EBM_ASSERT(0.0 < oldImpurity);
                                    impurity = std::numeric_limits<double>::max() - oldImpurity;
                                    newImpurity = std::numeric_limits<double>::max();
                                 } else {
                                    EBM_ASSERT(-std::numeric_limits<double>::infinity() == newImpurity);
                                    EBM_ASSERT(oldImpurity < 0.0);
                                    impurity = -std::numeric_limits<double>::max() - oldImpurity;
                                    newImpurity = -std::numeric_limits<double>::max();
                                 }
                              }
                              *pImpurity = newImpurity;
                           }

                           const double absImpurity = std::abs(impurity);
                           bRetry |= impurityMax < absImpurity;
                           impurityCur += absImpurity * impuritySumOverflowPreventer;

                           impurity = -impurity;

                           size_t iScoreUpdate = iTensorScore;
                           do {
                              // this can create new +-inf values in the tensor
                              double* const pScoreUpdate = IndexByte(pScores, iScoreUpdate);
                              double score = *pScoreUpdate;
                              if(!std::isinf(score)) {
                                 score += impurity;
                                 if(std::isinf(score)) {
                                    // we transitioned a score to an infinity. This can dramatically increase the
                                    // impurityCur value of the next iteration since we might have balanced a big
                                    // value against an opposite sign big value and now with one of them as infinite
                                    // the other has no counterbalance, so we need to reset our impurity checker
                                    impurityCur = std::numeric_limits<double>::quiet_NaN();
                                 }
                                 *pScoreUpdate = score;
                              }
                              iScoreUpdate += cTensorScoreIncrement;
                           } while(iTensorEnd != iScoreUpdate);
                        }
                     }
                     ++iRandom;
                  } while(cSurfaceBins != iRandom);
                  // this loops on std::isnan(impurityCur)
               } while(bRetry && !(impurityPrev <= impurityCur));
               EBM_ASSERT(!std::isnan(impurityCur));

               if(nullptr != pIntercept) {
                  double factorPost = 1.0;

                  size_t iScorePost = 0;
                  const double* pWeightPost = aWeights;
                  double impurityTotalPost = 0.0;
                  double weightTotalPost = 0.0;
                  do {
                     const double weight = *pWeightPost;
                     EBM_ASSERT(!std::isnan(weight) && 0.0 <= weight);
                     if(std::numeric_limits<double>::infinity() == weight) {
                        size_t cInfWeights;
                        goto skip_multiply_intercept2;
                        do {
                           factorPost *= 0.5;
                           // there should be a factor that will allow us to succeed before this
                           EBM_ASSERT(std::numeric_limits<double>::min() <= factorPost);
                        skip_multiply_intercept2:;

                           size_t iScoreInterior = iScorePost;
                           const double* pWeightInterior = pWeightPost;
                           impurityTotalPost = 0.0;
                           cInfWeights = 0;
                           do {
                              const double weightInterior = *pWeightInterior;
                              EBM_ASSERT(!std::isnan(weightInterior) && 0.0 <= weightInterior);
                              if(std::numeric_limits<double>::infinity() == weightInterior) {
                                 const double scoreInterior = *IndexByte(pScores, iScoreInterior);
                                 if(!std::isnan(scoreInterior) && !std::isinf(scoreInterior)) {
                                    ++cInfWeights;
                                    impurityTotalPost += factorPost * scoreInterior;

                                    // impurityTotalPost can reach -+inf, but once it gets there it cannot
                                    // escape that value because everything we add subsequently is non-inf.
                                    EBM_ASSERT(!std::isnan(impurityTotalPost));
                                 }
                              }
                              iScoreInterior += cBytesScoreClasses;
                              ++pWeightInterior;
                           } while(pWeightsEnd != pWeightInterior);
                        } while(std::isinf(impurityTotalPost));
                        weightTotalPost = static_cast<double>(cInfWeights);
                        // cInfWeights can be zero if all the +inf weights are +-inf or NaN scores, so goto a check for
                        // this
                        goto post_intercept;
                     }
                     const double score = *IndexByte(pScores, iScorePost);
                     if(!std::isnan(score) && !std::isinf(score)) {
                        weightTotalPost += weight;
                        const double impurity = weight * score;
                        impurityTotalPost += impurity;
                     }
                     iScorePost += cBytesScoreClasses;
                     ++pWeightPost;
                  } while(pWeightsEnd != pWeightPost);
                  EBM_ASSERT(!std::isnan(weightTotalPost));
                  EBM_ASSERT(0.0 <= weightTotalPost);

                  while(std::isnan(impurityTotalPost) || std::isinf(impurityTotalPost) || std::isinf(weightTotalPost)) {
                     // If impurity is NaN, it means that score * weight overflowed to +inf once and -inf another time

                     // In IEEE-754 this is an exact operation and should loose no information unless it underflows
                     factorPost *= 0.5;
                     // there should be a factor that will allow us to succeed before this
                     EBM_ASSERT(std::numeric_limits<double>::min() <= factorPost);
                     impurityTotalPost = 0.0;
                     weightTotalPost = 0.0;
                     size_t iScoreRetry = 0;
                     const double* pWeightRetry = aWeights;
                     do {
                        const double weight = *pWeightRetry;
                        EBM_ASSERT(!std::isnan(weight) && 0.0 <= weight &&
                              weight != std::numeric_limits<double>::infinity());
                        const double score = *IndexByte(pScores, iScoreRetry);
                        if(!std::isnan(score) && !std::isinf(score)) {
                           const double weightTimesFactor = factorPost * weight;
                           weightTotalPost += weightTimesFactor;
                           const double scoreTimesFactor = factorPost * score;
                           impurityTotalPost += scoreTimesFactor * weightTimesFactor;
                        }
                        iScoreRetry += cBytesScoreClasses;
                        ++pWeightRetry;
                     } while(pWeightsEnd != pWeightRetry);
                  }
                  EBM_ASSERT(!std::isnan(weightTotalPost));
                  EBM_ASSERT(0.0 <= weightTotalPost);

               post_intercept:;
                  if(std::numeric_limits<double>::min() <= weightTotalPost) {
                     // pull out the intercept early since this will make purification easier
                     double interceptChange = impurityTotalPost / weightTotalPost / factorPost;
                     EBM_ASSERT(!std::isnan(interceptChange));
                     if(std::isinf(interceptChange)) {
                        // the intercept is the weighted average of numbers, so it cannot mathematically
                        // be larger than the largest number, and we checked that the sum of those numbers
                        // did not overflow, so this shouldn't be possible without floating point noise
                        // If it does happen, the real value must be very very close to +-max_float,
                        // so use that instead

                        if(std::numeric_limits<double>::infinity() == interceptChange) {
                           interceptChange = std::numeric_limits<double>::max();
                        } else {
                           EBM_ASSERT(-std::numeric_limits<double>::infinity() == interceptChange);
                           interceptChange = -std::numeric_limits<double>::max();
                        }
                     }

                     double newIntercept = *pIntercept + interceptChange;
                     if(std::isinf(newIntercept)) {
                        // It should be pretty difficult, or perhaps even impossible, for the impurity,
                        // which starts from 0.0 to reach +-infinity since it comes from the weighted averaged
                        // values in the original tensor. Allowing the impurity to reach +-inf creates more
                        // problems I think than limiting it to the maximum non-inf float value. Even if we do
                        // get an overflow to +inf, I think it should be pretty close to the max float.
                        // Checking here allows us to give a guarantee that the impurities are normal floats.

                        if(std::numeric_limits<double>::infinity() == newIntercept) {
                           newIntercept = std::numeric_limits<double>::max();
                        } else {
                           EBM_ASSERT(-std::numeric_limits<double>::infinity() == newIntercept);
                           newIntercept = -std::numeric_limits<double>::max();
                        }
                     }
                     *pIntercept = newIntercept;

                     const double interceptChangeNeg = -interceptChange;
                     size_t iScoreUpdate = 0;
                     do {
                        // this can create new +-inf values, but not NaN since we limited intercept to non-NaN, non-inf
                        double* const pScoreUpdate = IndexByte(pScores, iScoreUpdate);
                        const double scoreOld = *pScoreUpdate;
                        const double scoreNew = scoreOld + interceptChangeNeg;
                        EBM_ASSERT(std::isnan(scoreOld) || !std::isnan(scoreNew));
                        *pScoreUpdate = scoreNew;
                        iScoreUpdate += cBytesScoreClasses;
                     } while(iScoresEnd != iScoreUpdate);
                  }
               }
            }
         }
      }
      ++pScores;
      if(nullptr != pImpurities) {
         ++pImpurities;
      }
      if(nullptr != pIntercept) {
         ++pIntercept;
      }
   } while(pScoreMulticlassEnd != pScores);

   return Error_None;
}

static void NormalizeClasses(const size_t cScores, double* const aScores) {
   // TODO: this function propagates NaN values to all the other multiclass positions
   // So, on any 2nd call we can use the knowlege that the 1st item in the array will be NaN if there are any NaN values

   // Respond asymetrically to +inf and -inf.  -inf means
   // that a particular class is impossible. +inf means that
   // the class is always selected.  Having mutliple -inf
   // values makes sense because then there are just a bunch
   // of impossible classes, but if there are mutliple +inf values,
   // it is ambiguous. In that case keep the multiple +inf values
   // but when any value is +inf then all the rest can be shifted
   // to -inf since they are impossible. NaN means a prediction
   // is illegal, so propagate the NaN to all classes.
   // Example responses look like:
   // -1.1, -2.2,  NaN, +3.1 =>  NaN,  NaN,  NaN,  NaN
   // -2.0, -3.0, -inf, +2.0 => -1.0, -2.0, -inf, +3.0
   // -inf, -2.0, -inf, +3.0 => -inf, -2.5, -inf, +2.5
   // -1.1, -2.2, +inf, +3.1 => -inf, -inf, +inf, -inf
   // +inf, -2.2, +inf, +3.1 => +inf, -inf, +inf, -inf
   // +inf, -2.2, -inf, +3.1 => +inf, -inf, -inf, -inf

   // Do not overflow to +inf, or underflow to -inf:
   //  low, high,  low,  0.0 =>  low, high,  low,  0.0
   // high,  low, high,  0.0 => high,  low, high,  0.0
   //
   // This restriction means that in some rare cases the
   // sum of the class scores will not be zero because
   // we would otherwise generate a +-inf value.

   EBM_ASSERT(1 <= cScores);
   EBM_ASSERT(nullptr != aScores);
   const double* const pScoresEnd = &aScores[cScores];
   double* pScore;

   double avg = 0.0;
   double valMax = -std::numeric_limits<double>::infinity();
   double valMin = std::numeric_limits<double>::infinity();
   pScore = aScores;
   do {
      const double score = *pScore;
      valMax = valMax < score ? score : valMax;
      valMin = score < valMin ? score : valMin;
      avg += score;
      ++pScore;
   } while(pScoresEnd != pScore);
   EBM_ASSERT(!std::isnan(valMax));
   EBM_ASSERT(!std::isnan(valMin));

   const double normalize = 1.0 / static_cast<double>(cScores);
   avg *= normalize;

   if(std::isnan(avg)) {
      if(-std::numeric_limits<double>::infinity() == valMax || std::numeric_limits<double>::infinity() == valMin) {
         // valMax can only be -inf if all numbers are -inf or NaN
         // valMin can only be +inf if all numbers are +inf or NaN
         // But they summed to NaN, which all +-inf values cannot do, so there is a NaN
         pScore = aScores;
         do {
            *pScore = std::numeric_limits<double>::quiet_NaN();
            ++pScore;
         } while(pScoresEnd != pScore);
         return;
      }
      if(std::numeric_limits<double>::infinity() == valMax) {
         pScore = aScores;
         do {
            double score = *pScore;
            if(std::isnan(score)) {
               pScore = aScores;
               do {
                  *pScore = std::numeric_limits<double>::quiet_NaN();
                  ++pScore;
               } while(pScoresEnd != pScore);
               return;
            }
            score = std::numeric_limits<double>::infinity() == score ? score : -std::numeric_limits<double>::infinity();
            *pScore = score;
            ++pScore;
         } while(pScoresEnd != pScore);
         return;
      }
      if(-std::numeric_limits<double>::infinity() != valMin) {
         EBM_ASSERT(!std::isinf(valMax));
         EBM_ASSERT(!std::isinf(valMin));
         // there are no +-inf values since neither valMax nor valMin have an infinity
         // But, we summed to a NaN value. By summing numbers we could overflow to either +inf or -inf
         // but then once there our sum cannot escale to create the opposite sign infinite value,
         // so the only way that a NaN could exist is if there was alrady a NaN in the data.
         pScore = aScores;
         do {
            *pScore = std::numeric_limits<double>::quiet_NaN();
            ++pScore;
         } while(pScoresEnd != pScore);
         return;
      }
      // At this point there are no +inf values in the data, and we have at least one -inf value. Our sum was
      // NaN, which could be caused by a NaN in the data or we could sum big numbres that overflowed to +inf and
      // subsequently added a -inf value. We also know that there is at least one non-NaN, non-inf value in the
      // data since valMax is set to such a number. We need to find the valMin in the non-inf data.

      EBM_ASSERT(!std::isinf(valMax));
      EBM_ASSERT(-std::numeric_limits<double>::infinity() == valMin);

      avg = 0.0;
      valMin = std::numeric_limits<double>::infinity();
      size_t cNormal = 0;
      pScore = aScores;
      do {
         const double score = *pScore;
         EBM_ASSERT(std::numeric_limits<double>::infinity() != score);
         if(-std::numeric_limits<double>::infinity() != score) {
            if(std::isnan(score)) {
               pScore = aScores;
               do {
                  *pScore = std::numeric_limits<double>::quiet_NaN();
                  ++pScore;
               } while(pScoresEnd != pScore);
               return;
            }
            valMin = score < valMin ? score : valMin;
            avg += score * normalize; // multiply by normalize to prevent overflowing
            ++cNormal;
         }
         ++pScore;
      } while(pScoresEnd != pScore);
      EBM_ASSERT(1 <= cNormal); // since valMax is non-nan, non-inf

      const double refactor = static_cast<double>(cScores) / static_cast<double>(cNormal);
      avg *= refactor;

      if(std::isinf(avg)) {
         // avg cannot mathematically be larger than the largest number so if it overflowed it was
         // due to floating point noise, so restore to non-inf
         if(std::numeric_limits<double>::infinity() == avg) {
            avg = std::numeric_limits<double>::max();
         } else {
            EBM_ASSERT(-std::numeric_limits<double>::infinity() == avg);
            avg = -std::numeric_limits<double>::max();
         }
      }
   } else if(std::isinf(avg)) {
      if(-std::numeric_limits<double>::infinity() == valMax || std::numeric_limits<double>::infinity() == valMin) {
         // valMax can only be -inf if all numbers are -inf or NaN
         // valMin can only be +inf if all numbers are +inf or NaN
         // But they summed to a non-NaN value, so there are no NaNs
         // and all the values are already what they should be
         return;
      }
      if(std::numeric_limits<double>::infinity() == valMax) {
         pScore = aScores;
         do {
            double score = *pScore;
            EBM_ASSERT(!std::isnan(score));
            score = std::numeric_limits<double>::infinity() == score ? score : -std::numeric_limits<double>::infinity();
            *pScore = score;
            ++pScore;
         } while(pScoresEnd != pScore);
         return;
      }
      if(-std::numeric_limits<double>::infinity() == valMin) {
         // valMax is known, but valMin was overwritten by -inf, so find it again
         EBM_ASSERT(!std::isnan(valMax));
         EBM_ASSERT(!std::isinf(valMax));
         valMin = std::numeric_limits<double>::infinity();
         avg = 0.0;
         size_t cNormal = 0;
         pScore = aScores;
         do {
            const double score = *pScore;
            EBM_ASSERT(!std::isnan(score));
            EBM_ASSERT(std::numeric_limits<double>::infinity() != score);
            if(-std::numeric_limits<double>::infinity() != score) {
               valMin = score < valMin ? score : valMin;
               avg += score * normalize; // multiply by normalize to prevent overflowing
               ++cNormal;
            }
            ++pScore;
         } while(pScoresEnd != pScore);
         EBM_ASSERT(1 <= cNormal); // since valMax is non-nan, non-inf

         const double refactor = static_cast<double>(cScores) / static_cast<double>(cNormal);
         avg *= refactor;
      } else {
         // there are no NaN, +-inf values, but there was an overflow, so prevent that
         avg = 0.0;
         pScore = aScores;
         do {
            const double score = *pScore;
            avg += score * normalize; // multiply by normalize to prevent overflowing
            ++pScore;
         } while(pScoresEnd != pScore);
      }
      if(std::isinf(avg)) {
         // avg cannot mathematically be larger than the largest number so if it overflowed it was
         // due to floating point noise, so restore to non-inf
         if(std::numeric_limits<double>::infinity() == avg) {
            avg = std::numeric_limits<double>::max();
         } else {
            EBM_ASSERT(-std::numeric_limits<double>::infinity() == avg);
            avg = -std::numeric_limits<double>::max();
         }
      }
   }

   EBM_ASSERT(!std::isnan(valMax));
   EBM_ASSERT(!std::isinf(valMax));
   EBM_ASSERT(!std::isnan(valMin));
   EBM_ASSERT(!std::isinf(valMin));
   EBM_ASSERT(!std::isnan(avg));
   EBM_ASSERT(!std::isinf(avg));

   double shift = -avg;
   if(0.0 <= shift) {
      if(std::numeric_limits<double>::max() - shift < valMax) {
         shift = std::numeric_limits<double>::max() - valMax;
      }
   } else {
      if(valMin < -std::numeric_limits<double>::max() - shift) {
         shift = -std::numeric_limits<double>::max() - valMin;
      }
   }

   pScore = aScores;
   do {
      double score = *pScore;
      EBM_ASSERT(!std::isnan(score));
      EBM_ASSERT(std::numeric_limits<double>::infinity() != score);
      if(-std::numeric_limits<double>::infinity() != score) {
         score += shift;
         if(std::isinf(score)) {
            // we limited shift to not overflow, so this overflow was due to floating point numeracy
            if(std::numeric_limits<double>::infinity() == score) {
               score = std::numeric_limits<double>::max();
            } else {
               EBM_ASSERT(-std::numeric_limits<double>::infinity() == score);
               score = -std::numeric_limits<double>::max();
            }
         }
         *pScore = score;
      }
      ++pScore;
   } while(pScoresEnd != pScore);
}

static ErrorEbm PurifyNormalizedMulticlass(RandomDeterministic &rng, 
      const size_t cScores,
      const size_t cTensorBins,
      const size_t cSurfaceBins,
      size_t* const aRandomize,
      const size_t* const aDimensionLengths,
      const double* const aWeights,
      double* const aScoresInOut,
      double* const aImpuritiesOut,
      double* const aInterceptOut) {
   EBM_ASSERT(1 <= cScores);
   EBM_ASSERT(1 <= cTensorBins);
   EBM_ASSERT(nullptr != aDimensionLengths);
   EBM_ASSERT(nullptr != aWeights);
   EBM_ASSERT(nullptr != aScoresInOut);

   const size_t cBytesScoreClasses = sizeof(double) * cScores;

   if(nullptr != aImpuritiesOut) {
      memset(aImpuritiesOut, 0, cBytesScoreClasses * cSurfaceBins);
   }

   const size_t iScoresEnd = cBytesScoreClasses * cTensorBins;
   const double* const pWeightsEnd = &aWeights[cTensorBins];
   const double* const pScoreMulticlassEnd = &aScoresInOut[cScores];
   const double* const pScoreEnd = &aScoresInOut[cScores * cTensorBins];

   double* pScores;
   double* pImpurities;
   double* pIntercept;

   pScores = aScoresInOut;
   do {
      NormalizeClasses(cScores, pScores);
      pScores += cScores;
   } while(pScoreEnd != pScores);

   if(nullptr != aInterceptOut) {
      pScores = aScoresInOut;
      pImpurities = aImpuritiesOut;
      pIntercept = aInterceptOut;
      do {
         EBM_ASSERT(nullptr == pIntercept || 0.0 == *pIntercept);

         double impurityTotalPre = 0.0;
         double weightTotalPre = 0.0;

         double factorPre = 1.0;

         size_t iScorePre = 0;
         const double* pWeightPre = aWeights;
         do {
            const double weight = *pWeightPre;
            if(!(0.0 <= weight)) {
               LOG_0(Trace_Error, "ERROR PurifyNormalizedMulticlass weight cannot be negative or NaN");
               return Error_IllegalParamVal;
            }
            EBM_ASSERT(!std::isnan(weight)); // !(0.0 <= weight) above checks for NaN
            if(std::numeric_limits<double>::infinity() == weight) {
               size_t cInfWeights;
               goto skip_multiply_intercept;
               do {
                  factorPre *= 0.5;
                  // there should be a factor that will allow us to succeed before this
                  EBM_ASSERT(std::numeric_limits<double>::min() <= factorPre);
               skip_multiply_intercept:;

                  size_t iScoreInterior = iScorePre;
                  const double* pWeightInterior = pWeightPre;
                  impurityTotalPre = 0.0;
                  cInfWeights = 0;
                  do {
                     const double weightInterior = *pWeightInterior;
                     if(!(0.0 <= weightInterior)) {
                        LOG_0(Trace_Error, "ERROR PurifyNormalizedMulticlass weight cannot be negative or NaN");
                        return Error_IllegalParamVal;
                     }
                     EBM_ASSERT(!std::isnan(weightInterior)); // !(0.0 <= weightInterior) above checks for NaN
                     if(std::numeric_limits<double>::infinity() == weightInterior) {
                        const double scoreInterior = *IndexByte(pScores, iScoreInterior);
                        if(!std::isnan(scoreInterior) && !std::isinf(scoreInterior)) {
                           ++cInfWeights;
                           impurityTotalPre += factorPre * scoreInterior;

                           // impurityTotalPre can reach -+inf, but once it gets there it cannot
                           // escape that value because everything we add subsequently is non-inf.
                           EBM_ASSERT(!std::isnan(impurityTotalPre));
                        }
                     }
                     iScoreInterior += cBytesScoreClasses;
                     ++pWeightInterior;
                  } while(pWeightsEnd != pWeightInterior);
               } while(std::isinf(impurityTotalPre));
               weightTotalPre = static_cast<double>(cInfWeights);
               // cInfWeights can be zero if all the +inf weights are +-inf or NaN scores, so goto a check for this
               goto pre_intercept;
            }
            const double score = *IndexByte(pScores, iScorePre);
            if(!std::isnan(score) && !std::isinf(score)) {
               weightTotalPre += weight;
               const double impurity = weight * score;
               impurityTotalPre += impurity;
            }
            iScorePre += cBytesScoreClasses;
            ++pWeightPre;
         } while(pWeightsEnd != pWeightPre);
         EBM_ASSERT(!std::isnan(weightTotalPre));
         EBM_ASSERT(0.0 <= weightTotalPre);

         while(std::isnan(impurityTotalPre) || std::isinf(impurityTotalPre) || std::isinf(weightTotalPre)) {
            // If impurity is NaN, it means that score * weight overflowed to +inf once and -inf another time

            // In IEEE-754 this is an exact operation and should loose no information unless it underflows
            factorPre *= 0.5;
            // there should be a factor that will allow us to succeed before this
            EBM_ASSERT(std::numeric_limits<double>::min() <= factorPre);
            impurityTotalPre = 0.0;
            weightTotalPre = 0.0;
            size_t iScoreRetry = 0;
            const double* pWeightRetry = aWeights;
            do {
               const double weight = *pWeightRetry;
               EBM_ASSERT(!std::isnan(weight) && 0.0 <= weight && weight != std::numeric_limits<double>::infinity());
               const double score = *IndexByte(pScores, iScoreRetry);
               if(!std::isnan(score) && !std::isinf(score)) {
                  const double weightTimesFactor = factorPre * weight;
                  weightTotalPre += weightTimesFactor;
                  const double scoreTimesFactor = factorPre * score;
                  impurityTotalPre += scoreTimesFactor * weightTimesFactor;
               }
               iScoreRetry += cBytesScoreClasses;
               ++pWeightRetry;
            } while(pWeightsEnd != pWeightRetry);
         }
         EBM_ASSERT(!std::isnan(weightTotalPre));
         EBM_ASSERT(0.0 <= weightTotalPre);

      pre_intercept:;
         if(std::numeric_limits<double>::min() <= weightTotalPre) {
            // pull out the intercept early since this will make purification easier
            double intercept = impurityTotalPre / weightTotalPre / factorPre;
            EBM_ASSERT(!std::isnan(intercept));
            if(std::isinf(intercept)) {
               // the intercept is the weighted average of numbers, so it cannot mathematically
               // be larger than the largest number, and we checked that the sum of those numbers
               // did not overflow, so this shouldn't be possible without floating point noise
               // If it does happen, the real value must be very very close to +-max_float,
               // so use that instead

               if(std::numeric_limits<double>::infinity() == intercept) {
                  intercept = std::numeric_limits<double>::max();
               } else {
                  EBM_ASSERT(-std::numeric_limits<double>::infinity() == intercept);
                  intercept = -std::numeric_limits<double>::max();
               }
            }

            *pIntercept = intercept;
            const double interceptNeg = -intercept;
            size_t iScoreUpdate = 0;
            do {
               // this can create new +-inf values, but not NaN since we limited intercept to non-NaN, non-inf
               double* const pScoreUpdate = IndexByte(pScores, iScoreUpdate);
               const double scoreOld = *pScoreUpdate;
               const double scoreNew = scoreOld + interceptNeg;
               EBM_ASSERT(std::isnan(scoreOld) || !std::isnan(scoreNew));
               *pScoreUpdate = scoreNew;
               iScoreUpdate += cBytesScoreClasses;
            } while(iScoresEnd != iScoreUpdate);
         }
         ++pScores;
         if(nullptr != pImpurities) {
            ++pImpurities;
         }
         if(nullptr != pIntercept) {
            ++pIntercept;
         }
      } while(pScoreMulticlassEnd != pScores);

      pScores = aScoresInOut;
      do {
         NormalizeClasses(cScores, pScores);
         pScores += cScores;
      } while(pScoreEnd != pScores);
   }

   if(size_t{0} != cSurfaceBins) {
      // this only happens for 1 dimensional inputs. Exit after finding the intercept

      // this prevents impurityCur from overflowing since the individual terms we add cannot sum to infinity
      // so as long as we multiply by 1/number_of_terms_summed we can guarantee the sum will not overflow
      // start from 0.5 instead of 1.0 to allow for floating point error.
      const double impuritySumOverflowPreventer = 0.5 / static_cast<double>(cScores * cSurfaceBins);
      double impurityPrev;
      double impurityCur = std::numeric_limits<double>::infinity();
      do {
         // if any non-infinite value was flipped to an infinite value, it could increase the impurity
         // so we set impurityCur to NaN. We need to reset it to +inf to avoid stopping early
         impurityCur = std::isnan(impurityCur) ? std::numeric_limits<double>::infinity() : impurityCur;
         impurityPrev = impurityCur;
         impurityCur = 0.0;

         if(nullptr != aRandomize) {
            size_t cRemaining = cSurfaceBins;
            do {
               --cRemaining;
               aRandomize[cRemaining] = cRemaining;
            } while(size_t{0} != cRemaining);

            cRemaining = cSurfaceBins;
            do {
               const size_t iSwap = rng.NextFast(cRemaining);
               const size_t iOriginal = aRandomize[iSwap];
               --cRemaining;
               aRandomize[iSwap] = aRandomize[cRemaining];
               aRandomize[cRemaining] = iOriginal;
            } while(size_t{0} != cRemaining);
         }

         size_t iRandom = 0;
         do {
            size_t iSurfaceBin = iRandom;
            if(nullptr != aRandomize) {
               iSurfaceBin = aRandomize[iRandom];
            }

            size_t cTensorWeightIncrement = sizeof(double);
            size_t cSweepBins;
            size_t iDimensionSurfaceBin = iSurfaceBin;
            const size_t* pSweepingDimensionLength = aDimensionLengths;
            while(true) {
               cSweepBins = *pSweepingDimensionLength;
               EBM_ASSERT(1 <= cSweepBins);
               EBM_ASSERT(0 == cTensorBins % cSweepBins);
               size_t cSurfaceBinsExclude = cTensorBins / cSweepBins;
               if(iDimensionSurfaceBin < cSurfaceBinsExclude) {
                  // we've found it
                  break;
               }
               iDimensionSurfaceBin -= cSurfaceBinsExclude;
               cTensorWeightIncrement *= cSweepBins;
               ++pSweepingDimensionLength;
            }

            if(size_t{1} != cSweepBins && cTensorBins != cSweepBins) {
               // If cSweepBins is cTensorBins, then all other dimensions are length 1 and the
               // surface is the same as the intercept.
               //
               // If cSweepBins is 1, then the entire tensor could be pushed down to a lower reduced
               // dimension however, we should keep the scores in this current tensor because we could
               // have more than 1 dimension of length 1, and then it would be ambiguous
               // and random which dimension we should push scores to.  Also, this should only occur
               // when the user specifies an interaction since our automatic interaction detection
               // will not select interactions with a feature of 1 bin. If the user specifies an
               // interaction with a useless bin length of 1, then that is what they'll get back.

               size_t iTensorWeight = 0;
               size_t multiple = sizeof(double);
               const size_t* pDimensionLength = aDimensionLengths;
               while(size_t{0} != iDimensionSurfaceBin) {
                  const size_t cBins = *pDimensionLength;
                  EBM_ASSERT(1 <= cBins);
                  if(pDimensionLength != pSweepingDimensionLength) {
                     const size_t iBin = iDimensionSurfaceBin % cBins;
                     iDimensionSurfaceBin /= cBins;
                     iTensorWeight += iBin * multiple;
                  }
                  multiple *= cBins;
                  ++pDimensionLength;
               }
               const size_t iTensorScore = iTensorWeight * cScores;
               const size_t cTensorScoreIncrement = cTensorWeightIncrement * cScores;
               const size_t iTensorEnd = iTensorScore + cTensorScoreIncrement * cSweepBins;

               pScores = aScoresInOut;
               pImpurities = aImpuritiesOut;
               do {
                  double factor = 1.0;
                  double impurity = 0.0;
                  double weightTotal = 0.0;
                  size_t iTensorWeightCur = iTensorWeight;
                  size_t iTensorScoreCur = iTensorScore;
                  do {
                     const double weight = *IndexByte(aWeights, iTensorWeightCur);
                     EBM_ASSERT(!std::isnan(weight) && 0.0 <= weight);
                     if(std::numeric_limits<double>::infinity() == weight) {
                        size_t cInfWeights;
                        goto skip_multiply;
                        do {
                           factor *= 0.5;
                           // there should be a factor that will allow us to succeed before this
                           EBM_ASSERT(std::numeric_limits<double>::min() <= factor);
                        skip_multiply:;
                           size_t iTensorWeightCurInterior = iTensorWeightCur;
                           size_t iTensorScoreCurInterior = iTensorScoreCur;
                           impurity = 0.0;
                           cInfWeights = 0;
                           do {
                              const double weightInterior = *IndexByte(aWeights, iTensorWeightCurInterior);
                              EBM_ASSERT(!std::isnan(weightInterior) && 0.0 <= weightInterior);
                              if(std::numeric_limits<double>::infinity() == weightInterior) {
                                 const double scoreInterior = *IndexByte(pScores, iTensorScoreCurInterior);
                                 if(!std::isnan(scoreInterior) && !std::isinf(scoreInterior)) {
                                    ++cInfWeights;
                                    impurity += factor * scoreInterior;

                                    // impurity can reach -+inf, but once it gets there it cannot
                                    // escape that value because everything we add subsequently is non-inf.
                                    EBM_ASSERT(!std::isnan(impurity));
                                 }
                              }
                              iTensorWeightCurInterior += cTensorWeightIncrement;
                              iTensorScoreCurInterior += cTensorScoreIncrement;
                           } while(iTensorEnd != iTensorScoreCurInterior);
                        } while(std::isinf(impurity));
                        weightTotal = static_cast<double>(cInfWeights);
                        // cInfWeights can be zero if all the +inf weights are +-inf or NaN scores, so goto a check for
                        // this
                        goto do_impurity;
                     }
                     const double score = *IndexByte(pScores, iTensorScoreCur);
                     if(!std::isnan(score) && !std::isinf(score)) {
                        weightTotal += weight;
                        impurity += weight * score;
                     }
                     iTensorWeightCur += cTensorWeightIncrement;
                     iTensorScoreCur += cTensorScoreIncrement;
                  } while(iTensorEnd != iTensorScoreCur);

                  while(std::isnan(impurity) || std::isinf(impurity) || std::isinf(weightTotal)) {
                     // if impurity is NaN, it means that score * weight overflowed to +inf once and -inf another
                     // time

                     // in IEEE-754 this is an exact operation and should loose no information unless it underflows
                     factor *= 0.5;
                     // there should be a factor that will allow us to succeed before this
                     EBM_ASSERT(std::numeric_limits<double>::min() <= factor);
                     impurity = 0.0;
                     weightTotal = 0.0;
                     iTensorWeightCur = iTensorWeight;
                     iTensorScoreCur = iTensorScore;
                     do {
                        const double weight = *IndexByte(aWeights, iTensorWeightCur);
                        EBM_ASSERT(!std::isnan(weight) && 0.0 <= weight &&
                              weight != std::numeric_limits<double>::infinity());
                        const double score = *IndexByte(pScores, iTensorScoreCur);
                        if(!std::isnan(score) && !std::isinf(score)) {
                           const double weightTimesFactor = factor * weight;
                           weightTotal += weightTimesFactor;
                           const double scoreTimesFactor = factor * score;
                           impurity += scoreTimesFactor * weightTimesFactor;
                        }
                        iTensorWeightCur += cTensorWeightIncrement;
                        iTensorScoreCur += cTensorScoreIncrement;
                     } while(iTensorEnd != iTensorScoreCur);
                  }

               do_impurity:;

                  if(std::numeric_limits<double>::min() <= weightTotal) {
                     impurity = impurity / weightTotal / factor;
                     EBM_ASSERT(!std::isnan(impurity));
                     if(std::isinf(impurity)) {
                        // impurity is the weighted average of numbers, so it cannot mathematically
                        // be larger than the largest number, and we checked that the sum of those numbers
                        // did not overflow, so this shouldn't be possible without floating point noise.
                        // If it does happen, the real value must be very very close to +-max_float,
                        // so use that instead.

                        if(std::numeric_limits<double>::infinity() == impurity) {
                           impurity = std::numeric_limits<double>::max();
                        } else {
                           EBM_ASSERT(-std::numeric_limits<double>::infinity() == impurity);
                           impurity = -std::numeric_limits<double>::max();
                        }
                     }

                     if(nullptr != pImpurities) {
                        double* const pImpurity = IndexByte(pImpurities, iSurfaceBin * cBytesScoreClasses);
                        const double oldImpurity = *pImpurity;
                        // we prevent impurities from reaching NaN or an infinity
                        EBM_ASSERT(!std::isnan(oldImpurity));
                        EBM_ASSERT(!std::isinf(oldImpurity));
                        double newImpurity = oldImpurity + impurity;
                        if(std::isinf(newImpurity)) {
                           // There should be a solution that allows any tensor to be purified
                           // without overflowing any of the impurity cells, however due to the
                           // random ordering that we process cells, they could temporarily overflow
                           // to +-inf, so limit the change to something that doesn't overflow to
                           // allow the weight to be transfered elsewhere.

                           if(std::numeric_limits<double>::infinity() == newImpurity) {
                              EBM_ASSERT(0.0 < oldImpurity);
                              impurity = std::numeric_limits<double>::max() - oldImpurity;
                              newImpurity = std::numeric_limits<double>::max();
                           } else {
                              EBM_ASSERT(-std::numeric_limits<double>::infinity() == newImpurity);
                              EBM_ASSERT(oldImpurity < 0.0);
                              impurity = -std::numeric_limits<double>::max() - oldImpurity;
                              newImpurity = -std::numeric_limits<double>::max();
                           }
                        }
                        *pImpurity = newImpurity;
                     }

                     const double absImpurity = std::abs(impurity);
                     impurityCur += absImpurity * impuritySumOverflowPreventer;

                     impurity = -impurity;

                     size_t iScoreUpdate = iTensorScore;
                     do {
                        // this can create new +-inf values in the tensor
                        double* const pScoreUpdate = IndexByte(pScores, iScoreUpdate);
                        double score = *pScoreUpdate;
                        if(!std::isinf(score)) {
                           score += impurity;
                           if(std::isinf(score)) {
                              // we transitioned a score to an infinity. This can dramatically increase the
                              // impurityCur value of the next iteration since we might have balanced a big
                              // value against an opposite sign big value and now with one of them as infinite
                              // the other has no counterbalance, so we need to reset our impurity checker
                              impurityCur = std::numeric_limits<double>::quiet_NaN();
                           }
                           *pScoreUpdate = score;
                        }
                        iScoreUpdate += cTensorScoreIncrement;
                     } while(iTensorEnd != iScoreUpdate);
                  }

                  ++pScores;
                  if(nullptr != pImpurities) {
                     ++pImpurities;
                  }
               } while(pScoreMulticlassEnd != pScores);

               size_t iTensorScoreNorm = iTensorScore;
               do {
                  double* const pScore = IndexByte(aScoresInOut, iTensorScoreNorm);
                  NormalizeClasses(cScores, pScore);
                  iTensorScoreNorm += cTensorScoreIncrement;
               } while(iTensorEnd != iTensorScoreNorm);
            }
            ++iRandom;
         } while(cSurfaceBins != iRandom);
         // this loops on std::isnan(impurityCur)
      } while(!(impurityPrev <= impurityCur));
      EBM_ASSERT(!std::isnan(impurityCur));

      if(nullptr != aInterceptOut) {
         pScores = aScoresInOut;
         pImpurities = aImpuritiesOut;
         pIntercept = aInterceptOut;
         do {
            double factorPost = 1.0;

            size_t iScorePost = 0;
            const double* pWeightPost = aWeights;
            double impurityTotalPost = 0.0;
            double weightTotalPost = 0.0;
            do {
               const double weight = *pWeightPost;
               EBM_ASSERT(!std::isnan(weight) && 0.0 <= weight);
               if(std::numeric_limits<double>::infinity() == weight) {
                  size_t cInfWeights;
                  goto skip_multiply_intercept2;
                  do {
                     factorPost *= 0.5;
                     // there should be a factor that will allow us to succeed before this
                     EBM_ASSERT(std::numeric_limits<double>::min() <= factorPost);
                  skip_multiply_intercept2:;

                     size_t iScoreInterior = iScorePost;
                     const double* pWeightInterior = pWeightPost;
                     impurityTotalPost = 0.0;
                     cInfWeights = 0;
                     do {
                        const double weightInterior = *pWeightInterior;
                        EBM_ASSERT(!std::isnan(weightInterior) && 0.0 <= weightInterior);
                        if(std::numeric_limits<double>::infinity() == weightInterior) {
                           const double scoreInterior = *IndexByte(pScores, iScoreInterior);
                           if(!std::isnan(scoreInterior) && !std::isinf(scoreInterior)) {
                              ++cInfWeights;
                              impurityTotalPost += factorPost * scoreInterior;

                              // impurityTotalPost can reach -+inf, but once it gets there it cannot
                              // escape that value because everything we add subsequently is non-inf.
                              EBM_ASSERT(!std::isnan(impurityTotalPost));
                           }
                        }
                        iScoreInterior += cBytesScoreClasses;
                        ++pWeightInterior;
                     } while(pWeightsEnd != pWeightInterior);
                  } while(std::isinf(impurityTotalPost));
                  weightTotalPost = static_cast<double>(cInfWeights);
                  // cInfWeights can be zero if all the +inf weights are +-inf or NaN scores, so goto a check for this
                  goto post_intercept;
               }
               const double score = *IndexByte(pScores, iScorePost);
               if(!std::isnan(score) && !std::isinf(score)) {
                  weightTotalPost += weight;
                  const double impurity = weight * score;
                  impurityTotalPost += impurity;
               }
               iScorePost += cBytesScoreClasses;
               ++pWeightPost;
            } while(pWeightsEnd != pWeightPost);
            EBM_ASSERT(!std::isnan(weightTotalPost));
            EBM_ASSERT(0.0 <= weightTotalPost);

            while(std::isnan(impurityTotalPost) || std::isinf(impurityTotalPost) || std::isinf(weightTotalPost)) {
               // If impurity is NaN, it means that score * weight overflowed to +inf once and -inf another time

               // In IEEE-754 this is an exact operation and should loose no information unless it underflows
               factorPost *= 0.5;
               // there should be a factor that will allow us to succeed before this
               EBM_ASSERT(std::numeric_limits<double>::min() <= factorPost);
               impurityTotalPost = 0.0;
               weightTotalPost = 0.0;
               size_t iScoreRetry = 0;
               const double* pWeightRetry = aWeights;
               do {
                  const double weight = *pWeightRetry;
                  EBM_ASSERT(!std::isnan(weight) && 0.0 <= weight && weight != std::numeric_limits<double>::infinity());
                  const double score = *IndexByte(pScores, iScoreRetry);
                  if(!std::isnan(score) && !std::isinf(score)) {
                     const double weightTimesFactor = factorPost * weight;
                     weightTotalPost += weightTimesFactor;
                     const double scoreTimesFactor = factorPost * score;
                     impurityTotalPost += scoreTimesFactor * weightTimesFactor;
                  }
                  iScoreRetry += cBytesScoreClasses;
                  ++pWeightRetry;
               } while(pWeightsEnd != pWeightRetry);
            }
            EBM_ASSERT(!std::isnan(weightTotalPost));
            EBM_ASSERT(0.0 <= weightTotalPost);

         post_intercept:;
            if(std::numeric_limits<double>::min() <= weightTotalPost) {
               // pull out the intercept early since this will make purification easier
               double interceptChange = impurityTotalPost / weightTotalPost / factorPost;
               EBM_ASSERT(!std::isnan(interceptChange));
               if(std::isinf(interceptChange)) {
                  // the intercept is the weighted average of numbers, so it cannot mathematically
                  // be larger than the largest number, and we checked that the sum of those numbers
                  // did not overflow, so this shouldn't be possible without floating point noise
                  // If it does happen, the real value must be very very close to +-max_float,
                  // so use that instead

                  if(std::numeric_limits<double>::infinity() == interceptChange) {
                     interceptChange = std::numeric_limits<double>::max();
                  } else {
                     EBM_ASSERT(-std::numeric_limits<double>::infinity() == interceptChange);
                     interceptChange = -std::numeric_limits<double>::max();
                  }
               }

               double newIntercept = *pIntercept + interceptChange;
               if(std::isinf(newIntercept)) {
                  // It should be pretty difficult, or perhaps even impossible, for the impurity,
                  // which starts from 0.0 to reach +-infinity since it comes from the weighted averaged
                  // values in the original tensor. Allowing the impurity to reach +-inf creates more
                  // problems I think than limiting it to the maximum non-inf float value. Even if we do
                  // get an overflow to +inf, I think it should be pretty close to the max float.
                  // Checking here allows us to give a guarantee that the impurities are normal floats.

                  if(std::numeric_limits<double>::infinity() == newIntercept) {
                     newIntercept = std::numeric_limits<double>::max();
                  } else {
                     EBM_ASSERT(-std::numeric_limits<double>::infinity() == newIntercept);
                     newIntercept = -std::numeric_limits<double>::max();
                  }
               }
               *pIntercept = newIntercept;

               const double interceptChangeNeg = -interceptChange;
               size_t iScoreUpdate = 0;
               do {
                  // this can create new +-inf values, but not NaN since we limited intercept to non-NaN, non-inf
                  double* const pScoreUpdate = IndexByte(pScores, iScoreUpdate);
                  const double scoreOld = *pScoreUpdate;
                  const double scoreNew = scoreOld + interceptChangeNeg;
                  EBM_ASSERT(std::isnan(scoreOld) || !std::isnan(scoreNew));
                  *pScoreUpdate = scoreNew;
                  iScoreUpdate += cBytesScoreClasses;
               } while(iScoresEnd != iScoreUpdate);
            }
            ++pScores;
            if(nullptr != pImpurities) {
               ++pImpurities;
            }
            if(nullptr != pIntercept) {
               ++pIntercept;
            }
         } while(pScoreMulticlassEnd != pScores);

         pScores = aScoresInOut;
         do {
            NormalizeClasses(cScores, pScores);
            pScores += cScores;
         } while(pScoreEnd != pScores);
      }
   }
   return Error_None;
}

EBM_API_BODY ErrorEbm EBM_CALLING_CONVENTION Purify(double tolerance,
      BoolEbm isRandomized,
      BoolEbm isMulticlassNormalization,
      IntEbm countMultiScores,
      IntEbm countDimensions,
      const IntEbm* dimensionLengths,
      const double* weights,
      double* scoresInOut,
      double* impuritiesOut,
      double* interceptOut) {
   LOG_N(Trace_Info,
         "Entered Purify: "
         "tolerance=%le, "
         "isRandomized=%s, "
         "isMulticlassNormalization=%s, "
         "countMultiScores=%" IntEbmPrintf ", "
         "countDimensions=%" IntEbmPrintf ", "
         "dimensionLengths=%p, "
         "weights=%p, "
         "scoresInOut=%p, "
         "impuritiesOut=%p, "
         "interceptOut=%p",
         tolerance,
         ObtainTruth(isRandomized),
         ObtainTruth(isMulticlassNormalization),
         countMultiScores,
         countDimensions,
         static_cast<const void*>(dimensionLengths),
         static_cast<const void*>(weights),
         static_cast<const void*>(scoresInOut),
         static_cast<const void*>(impuritiesOut),
         static_cast<const void*>(interceptOut));

   ErrorEbm error;

   if(countMultiScores <= IntEbm{0}) {
      if(IntEbm{0} == countMultiScores) {
         LOG_0(Trace_Info, "INFO Purify zero scores");
         return Error_None;
      } else {
         LOG_0(Trace_Error, "ERROR Purify countMultiScores must be positive");
         return Error_IllegalParamVal;
      }
   }
   if(IsConvertError<size_t>(countMultiScores)) {
      LOG_0(Trace_Error, "ERROR Purify IsConvertError<size_t>(countMultiScores)");
      return Error_IllegalParamVal;
   }
   const size_t cScores = static_cast<size_t>(countMultiScores);

   if(IsMultiplyError(sizeof(*interceptOut), cScores)) {
      LOG_0(Trace_Error, "ERROR Purify IsMultiplyError(sizeof(*interceptOut), cScores)");
      return Error_IllegalParamVal;
   }

   if(nullptr != interceptOut) {
      memset(interceptOut, 0, sizeof(*interceptOut) * cScores);
   }

   if(countDimensions <= IntEbm{0}) {
      if(IntEbm{0} == countDimensions) {
         LOG_0(Trace_Info, "INFO Purify zero dimensions");
         return Error_None;
      } else {
         LOG_0(Trace_Error, "ERROR Purify countDimensions must be positive");
         return Error_IllegalParamVal;
      }
   }
   if(IntEbm{k_cDimensionsMax} < countDimensions) {
      LOG_0(Trace_Warning, "WARNING Purify countDimensions too large and would cause out of memory condition");
      return Error_OutOfMemory;
   }
   const size_t cDimensions = static_cast<size_t>(countDimensions);

   if(nullptr == dimensionLengths) {
      LOG_0(Trace_Error, "ERROR Purify nullptr == dimensionLengths");
      return Error_IllegalParamVal;
   }

   bool bZero = false;
   size_t iDimension = 0;
   do {
      const IntEbm dimensionsLength = dimensionLengths[iDimension];
      if(dimensionsLength <= IntEbm{0}) {
         if(dimensionsLength < IntEbm{0}) {
            LOG_0(Trace_Error, "ERROR Purify dimensionsLength value cannot be negative");
            return Error_IllegalParamVal;
         }
         bZero = true;
      }
      ++iDimension;
   } while(cDimensions != iDimension);
   if(bZero) {
      LOG_0(Trace_Info, "INFO Purify empty tensor");
      return Error_None;
   }

   iDimension = 0;
   size_t cTensorBins = 1;
   size_t aDimensionLengths[k_cDimensionsMax];
   do {
      const IntEbm dimensionsLength = dimensionLengths[iDimension];
      EBM_ASSERT(IntEbm{1} <= dimensionsLength);
      if(IsConvertError<size_t>(dimensionsLength)) {
         // the scores tensor could not exist with this many tensor bins, so it is an error
         LOG_0(Trace_Error, "ERROR Purify IsConvertError<size_t>(dimensionsLength)");
         return Error_IllegalParamVal;
      }
      const size_t cBins = static_cast<size_t>(dimensionsLength);
      aDimensionLengths[iDimension] = cBins;

      if(IsMultiplyError(cTensorBins, cBins)) {
         // the scores tensor could not exist with this many tensor bins, so it is an error
         LOG_0(Trace_Error, "ERROR Purify IsMultiplyError(cTensorBins, cBins)");
         return Error_IllegalParamVal;
      }
      cTensorBins *= cBins;

      ++iDimension;
   } while(cDimensions != iDimension);
   EBM_ASSERT(1 <= cTensorBins);

   if(nullptr == weights) {
      LOG_0(Trace_Error, "ERROR Purify nullptr == weights");
      return Error_IllegalParamVal;
   }

   if(nullptr == scoresInOut) {
      LOG_0(Trace_Error, "ERROR Purify nullptr == scoresInOut");
      return Error_IllegalParamVal;
   }

   if(std::isnan(tolerance) || std::isinf(tolerance) || tolerance < 0.0) {
      LOG_0(Trace_Error, "ERROR Purify std::isnan(tolerance) || std::isinf(tolerance) || tolerance < 0.0)");
      return Error_IllegalParamVal;
   }

   size_t cSurfaceBins = 0;
   if(1 < cDimensions) {
      // if there is only 1 dimension, then push all weight to the intercept and have no surface bins
      size_t iExclude = 0;
      do {
         const size_t cBins = aDimensionLengths[iExclude];
         EBM_ASSERT(0 == cTensorBins % cBins);
         const size_t cSurfaceBinsExclude = cTensorBins / cBins;
         cSurfaceBins += cSurfaceBinsExclude;
         ++iExclude;
      } while(cDimensions != iExclude);
   }

   if(IsMultiplyError(sizeof(double), cScores, cSurfaceBins)) {
      LOG_0(Trace_Error, "ERROR Purify IsMultiplyError(sizeof(double), cScores, cSurfaceBins)");
      return Error_IllegalParamVal;
   }

   RandomDeterministic rng;
   size_t* aRandomize = nullptr;
   if(EBM_FALSE != isRandomized) {
      if(IsMultiplyError(sizeof(*aRandomize), cSurfaceBins)) {
         LOG_0(Trace_Warning, "WARNING Purify IsMultiplyError(sizeof(*aRandomize), cSurfaceBins)");
         return Error_OutOfMemory;
      }
      aRandomize = static_cast<size_t*>(malloc(sizeof(*aRandomize) * cSurfaceBins));
      if(nullptr == aRandomize) {
         LOG_0(Trace_Warning, "WARNING Purify nullptr != aRandomize");
         return Error_OutOfMemory;
      }
      static constexpr uint64_t seed = 9271049328402875910u;
      rng.Initialize(seed);
   }

   // NOTES about generating new infinities during purification:
   // - We can't prevent new infinities from being created in the purified output. An easy example is
   //   [-DBL_MAX, -DBL_MAX, DBL_MAX] which purifies to [2/3 * -DBL_MAX, 2/3 * -DBL_MAX, 4/3 * DBL_MAX]
   // - On the first purification, we cannot overflow the intercept or purification surface cell since we're taking
   //   a weighted average the average cannot be larger than any value
   // - On subsequent purification steps, we can overflow the intercept or purification surface cell, however I 
   //   think if the algorithm was allowed to converge and there were no floating point noise issues the
   //   purified cell and/or intercept would not overflow to an infinity
   // - we can prevent the impurities and/or intercept from overflowing by limiting the amount of purification in
   //   the step to a value that does not overflow and I think the algorithm is still guaranteed to make forward 
   //   progress
   // - eventually though, even the impurities can get infinities since we later purified the impurities until we reach
   //   the intercept, so the only guarantee we could get in theory was that we don't overflow the intercept
   // - But even for the intercept, since there is an existing intercept we can't guarantee that the purified intercept
   //   will not overflow to an infinity, so there can be no guarantees in the EBM as a whole
   // 
   // We do take the following precautions:
   // - when we move impurity from the original tensor to the impurity tensor, we limit the purification at that step
   //   to a number that will not overflow the impurity cell
   // - when we normalize multiclass scores, we have a rule that we never adjust them to overflow a value that was not
   //   already an infinity. To do this we loose the guarantee that the numbers sum to zero within a bin
   // - we DO NOT guarantee that the intercept avoids overflowing since we add to the intercept at the start and end,
   //   but the caller can get this guarantee by passing NULL for the intercept pointer since we guarantee that the
   //   impurities are non-overflowing

   if(1 != cScores && EBM_FALSE != isMulticlassNormalization) {
      error = PurifyNormalizedMulticlass(rng,
            cScores,
            cTensorBins,
            cSurfaceBins,
            aRandomize,
            aDimensionLengths,
            weights,
            scoresInOut,
            impuritiesOut,
            interceptOut);
   } else {
      error = PurifyInternal(rng,
            tolerance,
            cScores,
            cTensorBins,
            cSurfaceBins,
            aRandomize,
            aDimensionLengths,
            weights,
            scoresInOut,
            impuritiesOut,
            interceptOut);
   }

   free(aRandomize);

   LOG_0(Trace_Info, "Exited Purify");

   return error;
}

} // namespace DEFINED_ZONE_NAME
