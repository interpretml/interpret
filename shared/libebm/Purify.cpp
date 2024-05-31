// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "pch.hpp"

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t
#include <string.h> // memcpy
#include <cmath> // std::abs

#define ZONE_main
#include "zones.h"

#include "RandomDeterministic.hpp" // RandomDeterministic

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

extern ErrorEbm PurifyInternal(const double tolerance,
      const size_t cTensorBins,
      const size_t cSurfaceBins,
      const size_t* const aDimensionLengths,
      const double* const aWeights,
      double* const aScores,
      double* const aImpurities,
      double* const pInterceptOut) {
   EBM_ASSERT(!std::isnan(tolerance));
   EBM_ASSERT(!std::isinf(tolerance));
   EBM_ASSERT(0.0 <= tolerance);
   EBM_ASSERT(1 <= cTensorBins);
   // we ignore surfaces of length 1 and anything equal to the original tensor, so 2x2 is smallest
   EBM_ASSERT(0 == cSurfaceBins || 4 <= cSurfaceBins);
   EBM_ASSERT(nullptr != aDimensionLengths);
   EBM_ASSERT(nullptr != aWeights);
   EBM_ASSERT(nullptr != aScores);

   if(nullptr != aImpurities) {
      memset(aImpurities, 0, cSurfaceBins * sizeof(*aImpurities));
   }

   const double* pScore = aScores;
   const double* pWeight = aWeights;
   const double* const aScoresEnd = aScores + cTensorBins;
   double impurityMax = 0.0;
   double impurityTotalAll = 0.0;
   double weightTotalAll = 0.0;
   do {
      const double weight = *pWeight;
      if(!(0.0 <= weight)) {
         LOG_0(Trace_Error, "ERROR PurifyInternal weight cannot be negative or NaN");
         return Error_IllegalParamVal;
      }
      EBM_ASSERT(!std::isnan(weight));
      if(std::numeric_limits<double>::infinity() == weight) {
         // TODO: handle +inf weight by putting all the "weight" on +inf bins and ignoring non-inf bins.
         // we also need to have a fallback incase the impurityTotalAll overflows when only considering inf bins.
         // For now though, just return an error
         LOG_0(Trace_Error, "ERROR PurifyInternal weight cannot be +inf");
         return Error_IllegalParamVal;
      }
      const double score = *pScore;
      if(!std::isnan(score) && !std::isinf(score)) {
         weightTotalAll += weight;
         const double impurity = weight * score;
         impurityTotalAll += impurity;
         impurityMax += std::abs(impurity);
      }
      ++pScore;
      ++pWeight;
   } while(aScoresEnd != pScore);
   EBM_ASSERT(!std::isnan(weightTotalAll));
   EBM_ASSERT(0.0 <= weightTotalAll);

   // even if score and weight are never NaN or infinity, the product of both can be +-inf and if +inf is added
   // to -inf it will be NaN, so impurityTotalAll can be NaN even if weight and score are well formed
   // impurityMax cannot be NaN though here since abs(impurity) cannot be -inf. If score was zero and weight
   // was +inf then impurityMax could be NaN, but we've excluded it at this point through bInfWeight.
   EBM_ASSERT(!std::isnan(impurityMax));
   EBM_ASSERT(0.0 <= impurityMax);
   
   double factorIntercept = 1.0;
   while(std::isnan(impurityTotalAll) || std::isinf(impurityTotalAll) || std::isinf(weightTotalAll)) {
      // If impurity is NaN, it means that score * weight overflowed to +inf once and -inf another time

      // In IEEE-754 this is an exact operation and should loose no information unless it underflows
      factorIntercept *= 0.5;
      // there should be a factor that will allow us to succeed before this
      EBM_ASSERT(std::numeric_limits<double>::min() <= factorIntercept);
      impurityTotalAll = 0.0;
      weightTotalAll = 0.0;
      pScore = aScores;
      pWeight = aWeights;
      do {
         const double weight = *pWeight;
         EBM_ASSERT(!std::isnan(weight) && 0.0 <= weight && weight != std::numeric_limits<double>::infinity());
         const double score = *pScore;
         if(!std::isnan(score) && !std::isinf(score)) {
            const double weightTimesFactor = factorIntercept * weight;
            weightTotalAll += weightTimesFactor;
            const double scoreTimesFactor = factorIntercept * score;
            impurityTotalAll += scoreTimesFactor * weightTimesFactor;
         }
         ++pScore;
         ++pWeight;
      } while(aScoresEnd != pScore);
   }
   EBM_ASSERT(!std::isnan(weightTotalAll));
   EBM_ASSERT(0.0 <= weightTotalAll);

   if(impurityMax < std::numeric_limits<double>::min() || weightTotalAll < std::numeric_limits<double>::min()) {
      // subnormal numbers are considered to be zero by us
      // impurityMax could be zero because the scores are zero or the weights are zero. Either way, it's pure.
      if(nullptr != pInterceptOut) {
         *pInterceptOut = 0.0;
      }
      return Error_None;
   }
   if(std::numeric_limits<double>::infinity() == impurityMax) {
      // handle this by just turning off early exit and let the algorithm exit when it cannot improve
      impurityMax = 0.0;
   } else {
      impurityMax = impurityMax * tolerance * factorIntercept / weightTotalAll;
      // at this location: 
      //   0.0 < impurityMax < +inf
      //   0.0 <= tolerance < +inf
      //   0.0 < factorIntercept <= 1.0
      //   0.0 < weightTotalAll < +inf
      //   impurityMax * tolerance can overflow to +inf, but dividing by a non-NaN, non-inf, non-zero number is non-NaN
      EBM_ASSERT(!std::isnan(impurityMax));
      if(std::numeric_limits<double>::infinity() == impurityMax) {
         // handle this by just turning off early exit and let the algorithm exit when it cannot improve
         impurityMax = 0.0;
      }
   }

   if(nullptr != pInterceptOut) {
      // pull out the intercept early since this will make purification easier
      double intercept = impurityTotalAll / weightTotalAll / factorIntercept;
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

      *pInterceptOut = intercept;
      const double interceptNeg = -intercept;
      double* pScore2 = aScores;
      do {
         // this can create new +-inf values, but not NaN since we limited intercept to non-NaN, non-inf
         const double scoreOld = *pScore2;
         const double scoreNew = scoreOld + interceptNeg;
         EBM_ASSERT(std::isnan(scoreOld) || !std::isnan(scoreNew));
         *pScore2 = scoreNew;
         ++pScore2;
      } while(aScoresEnd != pScore2);
   }

   // this prevents impurityCur from overflowing since the individual terms we add are not infinity
   // so as long as we multiply by 1/number_of_terms_summed we can guarantee the sum will not overflow
   // start from 0.5 instead of 1.0 to allow for floating point noise
   const double impuritySumOverflowPreventer = 0.5 / static_cast<double>(cSurfaceBins);
   double impurityPrev = std::numeric_limits<double>::infinity();
   double impurityCur;
   bool bRetry;
   do {
      impurityCur = 0.0;
      bRetry = false;

      // TODO: do a card shuffle of the surface bin indexes to process them in random order

      size_t iAllSurfaceBin = 0;
      do {
         size_t cTensorIncrement = sizeof(double);
         size_t cSweepBins;
         size_t iDimensionSurfaceBin = iAllSurfaceBin;
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
            cTensorIncrement *= cSweepBins;
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

            size_t iTensor = 0;
            size_t multiple = sizeof(double);
            const size_t* pDimensionLength = aDimensionLengths;
            while(size_t{0} != iDimensionSurfaceBin) {
               const size_t cBins = *pDimensionLength;
               EBM_ASSERT(1 <= cBins);
               if(pDimensionLength != pSweepingDimensionLength) {
                  const size_t iBin = iDimensionSurfaceBin % cBins;
                  iDimensionSurfaceBin /= cBins;
                  iTensor += iBin * multiple;
               }
               multiple *= cBins;
               ++pDimensionLength;
            }

            const size_t iTensorEnd = iTensor + cTensorIncrement * cSweepBins;
            double impurity = 0.0;
            double weightTotal = 0.0;
            size_t iTensorCur = iTensor;
            do {
               const double weight = *IndexByte(aWeights, iTensorCur);
               EBM_ASSERT(!std::isnan(weight) && 0.0 <= weight);
               if(std::numeric_limits<double>::infinity() == weight) {
                  EBM_ASSERT(false); // TODO: our current code above disallows +inf weight, but allow it in the future
               }
               const double score = *IndexByte(aScores, iTensorCur);
               if(!std::isnan(score) && !std::isinf(score)) {
                  weightTotal += weight;
                  impurity += weight * score;
               }
               iTensorCur += cTensorIncrement;
            } while(iTensorEnd != iTensorCur);

            double factor = 1.0;
            while(std::isnan(impurity) || std::isinf(impurity) || std::isinf(weightTotal)) {
               // if impurity is NaN, it means that score * weight overflowed to +inf once and -inf another time

               // in IEEE-754 this is an exact operation and should loose no information unless it underflows
               factor *= 0.5;
               // there should be a factor that will allow us to succeed before this
               EBM_ASSERT(std::numeric_limits<double>::min() <= factor); 
               impurity = 0.0;
               weightTotal = 0.0;
               iTensorCur = iTensor;
               do {
                  const double weight = *IndexByte(aWeights, iTensorCur);
                  EBM_ASSERT(!std::isnan(weight) && 0.0 <= weight &&
                        weight != std::numeric_limits<double>::infinity());
                  const double score = *IndexByte(aScores, iTensorCur);
                  if(!std::isnan(score) && !std::isinf(score)) {
                     const double weightTimesFactor = factor * weight;
                     weightTotal += weightTimesFactor;
                     const double scoreTimesFactor = factor * score;
                     impurity += scoreTimesFactor * weightTimesFactor;
                  }
                  iTensorCur += cTensorIncrement;
               } while(iTensorEnd != iTensorCur);
            }

            if(weightTotal < std::numeric_limits<double>::min()) {
               impurity = 0.0;
            } else {
               impurity = impurity / weightTotal / factor;
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
            }

            const double absImpurity = std::abs(impurity);
            bRetry |= impurityMax < absImpurity;
            impurityCur += absImpurity * impuritySumOverflowPreventer;

            // this can create new +-inf values in the surfaces
            if(nullptr != aImpurities) {
               aImpurities[iAllSurfaceBin] += impurity;
            }
            impurity = -impurity;

            size_t iTensorAdd = iTensor;
            do {
               // this can create new +-inf values in the tensor
               double score = *IndexByte(aScores, iTensorAdd) + impurity;
               if(std::isinf(score)) {
                  // It should be pretty difficult, or perhaps even impossible, for the impurity score, 
                  // which starts from 0.0 to reach +-infinity since it comes from the weighted averaged 
                  // values in the original tensor. Allowing the impurity to reach +-inf creates more
                  // problems I think than limiting it to the maximum non-inf float value. Even if we do
                  // get an overflow to +inf, I think it should be pretty close to the max float.
                  // Checking here allows us to give a guarantee that the impurities are normal floats.

                  if(std::numeric_limits<double>::infinity() == score) {
                     score = std::numeric_limits<double>::max();
                  } else {
                     EBM_ASSERT(-std::numeric_limits<double>::infinity() == score);
                     score = -std::numeric_limits<double>::max();
                  }
               }
               *IndexByte(aScores, iTensorAdd) = score;
               iTensorAdd += cTensorIncrement;
            } while(iTensorEnd != iTensorAdd);
         }
         ++iAllSurfaceBin;
      } while(cSurfaceBins != iAllSurfaceBin);

      if(impurityPrev <= impurityCur) {
         // To ensure that we exit even with floating point noise, exit when things do not improve.
         break;
      }
      impurityPrev = impurityCur;
   } while(bRetry);

   return Error_None;
}


EBM_API_BODY ErrorEbm EBM_CALLING_CONVENTION Purify(double tolerance,
      IntEbm countDimensions,
      const IntEbm* dimensionLengths,
      const double* weights,
      double* scores,
      double* impurities,
      double* interceptOut) {
   LOG_N(Trace_Info,
         "Entered Purify: "
         "tolerance=%le, "
         "countDimensions=%" IntEbmPrintf ", "
         "dimensionLengths=%p, "
         "weights=%p, "
         "scores=%p, "
         "impurities=%p, "
         "interceptOut=%p",
         tolerance,
         countDimensions,
         static_cast<const void*>(dimensionLengths),
         static_cast<const void*>(weights),
         static_cast<const void*>(scores),
         static_cast<const void*>(impurities),
         static_cast<const void*>(interceptOut));

   ErrorEbm error;

   if(nullptr != interceptOut) {
      *interceptOut = 0.0;
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
   do {
      const IntEbm dimensionsLength = dimensionLengths[iDimension];
      EBM_ASSERT(IntEbm{1} <= dimensionsLength);
      if(IsConvertError<size_t>(dimensionsLength)) {
         // the scores tensor could not exist with this many tensor bins, so it is an error
         LOG_0(Trace_Error, "ERROR Purify IsConvertError<size_t>(dimensionsLength)");
         return Error_OutOfMemory;
      }
      const size_t cBins = static_cast<size_t>(dimensionsLength);

      if(IsMultiplyError(cTensorBins, cBins)) {
         // the scores tensor could not exist with this many tensor bins, so it is an error
         LOG_0(Trace_Error, "ERROR Purify IsMultiplyError(cTensorBins, cBins)");
         return Error_OutOfMemory;
      }
      cTensorBins *= cBins;

      ++iDimension;
   } while(cDimensions != iDimension);
   EBM_ASSERT(1 <= cTensorBins);

   if(nullptr == weights) {
      LOG_0(Trace_Error, "ERROR Purify nullptr == weights");
      return Error_IllegalParamVal;
   }

   if(nullptr == scores) {
      LOG_0(Trace_Error, "ERROR Purify nullptr == scores");
      return Error_IllegalParamVal;
   }

   if(std::isnan(tolerance) || std::isinf(tolerance) || tolerance < 0.0) {
      LOG_0(Trace_Error, "ERROR Purify std::isnan(tolerance) || std::isinf(tolerance) || tolerance < 0.0)");
      return Error_IllegalParamVal;
   }

   size_t aDimensionLengths[k_cDimensionsMax];
   size_t cSurfaceBins = 0;
   size_t iExclude = 0;
   do {
      const size_t cBins = static_cast<size_t>(dimensionLengths[iExclude]);
      aDimensionLengths[iExclude] = cBins;
      EBM_ASSERT(0 == cTensorBins % cBins);
      const size_t cSurfaceBinsExclude = cTensorBins / cBins;
      cSurfaceBins += cSurfaceBinsExclude;
      ++iExclude;
   } while(cDimensions != iExclude);

   error = PurifyInternal(
         tolerance, cTensorBins, cSurfaceBins, aDimensionLengths, weights, scores, impurities, interceptOut);

   LOG_0(Trace_Info, "Exited Purify");

   return error;
}

} // namespace DEFINED_ZONE_NAME
