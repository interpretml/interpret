// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

// TODO: use noexcept throughout our codebase (exception extern "C" functions) !  The compiler can optimize functions better if it knows there are no exceptions
// TODO: review all the C++ library calls, including things like std::abs and verify that none of them throw exceptions, otherwise use the C versions that provide this guarantee

#include <stddef.h> // size_t, ptrdiff_t
#include <limits> // std::numeric_limits

#include "ebm_native.h"
#include "EbmInternal.h"
#include "Logging.h" // EBM_ASSERT & LOG

// python binning types
//quantile
//quantile_humanized
//uniform
//winsorized

EBM_NATIVE_IMPORT_EXPORT_BODY IntEbmType EBM_NATIVE_CALLING_CONVENTION ConvertLogitsToProbabilities(
   IntEbmType countTargetClasses,
   IntEbmType countSamples,
   const FloatEbmType * logits,
   FloatEbmType * probabilitiesOut
) {
   UNUSED(countTargetClasses); // TODO: use this
   for(size_t i = 0; i < static_cast<size_t>(countSamples); ++i) {
      const FloatEbmType odds = std::exp(logits[i]);
      probabilitiesOut[i] = odds / (FloatEbmType { 1 } + odds);
   }
   return IntEbmType { 0 };
}

// Plan:
//   - when making predictions, in the great majority of cases, we should serially determine the logits of each
//     sample per feature and then later add those logits.  It's tempting to want to process more than one feature
//     at a time, but that's a red-hearing:
//     - data typically gets passed to us as C ordered data, so feature0 and feature1 are in adjacent memory
//       cells, and sample0 and sample1 are distant.  It's less costly to read the data per feature for our pure input
//       data.  It wouldn't do us much good though if we striped just two features at a time, so we'd want to
//       process all N features in order to take advantage of this property.  But if you do that, then we'd need
//       to do binary searches on a single sample for a single feature, then fetch into cache the next feature's
//       cut "definition".  The cost of constantly bringing into L1 cache the cut points and logits for each feature
//       would entail more memory movement than either processing the matrix out of order or transposing it beforehand
//     - it's tempting to then consider striping just 2 features or some limited subset.  We get limited speed benefits
//       when processing two features at a time since at best it halves the time to access the matrix, but we still
//       then need to keep two cut point arrays that we do unpredictable branches on and it potentially pushes some
//       of our cut point and logit arrays out from L1 cache into L2 or beyond
//     - we get benefits by having special case algorithms based on the number of cut points (see below where we
//       do linear searches for small numbers of cut points, and pad cut point arrays for slightly larger numbers of
//       cut points).  And it's hard to see how we could combine these together and say have a special loop to handle
//       when one feature has 3 cut points, and the other has 50 cut points
//     - one of the benefits of doing 2 features at once would be that we could add the logits together and write
//       the sum to memory instead of writing both logits and later reading those again and summing them and writing
//       them back to memory, but since we'd be doing this with correcly ordered memory, we'd be able to stream
//       the reads and the writes such that they'd take approx 1 clock cycle each, so in reality we don't gain much
//       from combining the logits at binary search time
//     - in theory we might gain something if we had two single cut features because we could load the 2 values we're
//       cutting into 2 registers, then have the cut points in 2 persistent registers, and have 4 registers for the
//       logit results.  We can overwrite one of the two registers loaded with the sum of the resulting logits.  
//       That's a total of 8 registers.  For 2 cuts, we'd need 2 for loading, 4 for cuts, 6 for logits, so 12 registers
//       Which is also doable.  Beyond that, we'd need to use or access memory when combining processing for 2 features
//       and I think it would be better to pay the streaming to memory cost than to fetch somewhat unpredictably
//       the cut points or logits
//     - even if we did write special case code for handling two binary features, it won't help us if the matrix the
//       user passes us doesn't put the binary features adjacent to eachother.  We can't re-arrange the columsn for
//       less than the cost of partial transposes, so we'd rather just go with partial transposes
//     - doing a partial striped transpose is 32% faster in my tests than reading 2 columns at once, so we'd be
//       better off transposing the two columns than process them.  This is because we are limited to reading just
//       two values efficiently at a time, rather than reading a full stripe efficiently.
//   - we can get data from the user as fortran ordered.  If it comes to us fortran ordered
//     then great, because our accessing that data per feature is very efficient (approx 1 clock cycle per read)
//   - we can get data from the user as C ordered (this is more common).  We could read the matrix in poor memory
//     order, but then we're still loading in a complete cache line at a time.  It makes more sense to read in data
//     in a stripe and transpose it that way.  I did some perfs, and reading stripes of 64 doubles was fastest
//     We pay the cost of having 64 write streams, but our reads are very fast.  That's the break even point though
//   - transposing the complete matrix would double our memory requirements.  Since transposing is fastest with 64
//     doubles though, we can extract and transpose our original C ordered data in 64 feature groupings
//   - we can use SIMD easily enough by loading the next 2/4/8 doubles at a time and re-using the same cut definition
//     within a single processor
//   - we can use threading efficiently in one of two ways.  We can subdivide the samples up by the number of CPUs
//     and have each CPU process those ranges.  This allows all the CPUs to utilize the same cut point definitions
//     but they have smaller batches.  Alternatively, we can give each CPU one feature and have it load the cut
//     point and logit definitions into it's L1 cache which isn't likely to be shared.  If some of the cut points
//     or logits need to be in L2 though, there might be bad contention.
//   - hyper-threads would probably benefit from having the same cut points and logits since both hyper-threads share
//     the L1 cahce, so the "best" solution is probably use thread afinity to keep CPUs working on the same feature
//     and dividing up the samples between the hyper-threads, but then benefit from larger batch sizes by putting
//     different features on different CPUs
//   - the exact threading solution will probably depend on exact numbers of samples and threads and machine 
//     architecture
//   - whether dividing the work by samples or features or a mix, if we make multiple calls into our discritize
//     function, we would want to preserve our threads since they are costly to make, so we'd want to have a
//     thread allocation object that we'd free after discretization
//   - for fortran ordered arrays, the user might as well pass us the entire array and we'll process it directly
//   - for C ordered data, either the 64 stride transpose happens in our higher level caller, or they just pass
//     us the C ordered data, and we do the partial transposes inside C++ from the badly ordered original data
//   - in the entire dataset gets passed to us, then we don't need a thread allocation object since we just do it once
//   - if the original array is in pandas, it seems to be stored internally as a numpy array if the datatypes are all
//     the same, so we can pass that direclty into our function
//   - if the original array is in pandas, and consists of strings or integers or anything heterogenious, then
//     the data appears to be fortran ordered.  In that case we'd like to pass the data in that bare format
//   - but we're not sure that pandas stores these as 2-D matricies or multiple 1-D arrays.  If the ladder, then
//     we either need to process it one array at a time, or copy the data together.
//   - handling strings can either be done with python vectorized functions or in cython (try pure python first)
//   - after our per-feature logit arrays have been written, we can load in several at a time and add them together
//     and write out the result, and we can parallelize that operation until all the logits have been added
//   - SIMD reads and writes are better on certain boundaries.  We don't control the data passed to us from the user
//     so we might want to read the first few instances with a special binary search function and then start
//     on the SIMD on a memory aligned boundary, then also use the special binary search function for the last few
//   - one complication is that for pairs we need to have both feature in memory to evaluate.  If the pairs are
//     not in the same stripe we need to preserve them until they are.  In most cases we can probably just hold the
//     features we need or organize which stripes we load at which times, but in the worst case we may want
//     to re-discretize some features, or in the worst case discretize all features (preserving in a compressed 
//     format?).  This really needs to be threshed out.
//
//   - Table of matrix access speeds (for summing cells in a matrix):
//       bad_order = 7.43432
//       stride_1 = 7.27575
//       stride_2 = 4.08857
//       stride_16384 = 0.431882
//       transpose_1 = 10.4326
//       transpose_2 = 6.49787
//       transpose_4 = 4.54615
//       transpose_8 = 3.42918
//       transpose_16 = 3.04755
//       transpose_32 = 2.80757
//       transpose_64 = 2.75464
//       transpose_128 = 2.79845
//       transpose_256 = 2.8748
//       transpose_512 = 2.96725
//       transpose_1024 = 3.17072
//       transpose_2048 = 6.04042
//       transpose_4096 = 6.1348
//       transpose_8192 = 6.26907
//       transpose_16384 = 7.73406

// don't bother using a lock here.  We don't care if an extra log message is written out due to thread parallism
static int g_cLogEnterDiscretizeParametersMessages = 25;
static int g_cLogExitDiscretizeParametersMessages = 25;

EBM_NATIVE_IMPORT_EXPORT_BODY IntEbmType EBM_NATIVE_CALLING_CONVENTION Discretize(
   IntEbmType countSamples,
   const FloatEbmType * featureValues,
   IntEbmType countBinCuts,
   const FloatEbmType * binCutsLowerBoundInclusive,
   IntEbmType * discretizedOut
) {
   LOG_COUNTED_N(
      &g_cLogEnterDiscretizeParametersMessages,
      TraceLevelInfo,
      TraceLevelVerbose,
      "Entered Discretize: "
      "countSamples=%" IntEbmTypePrintf ", "
      "featureValues=%p, "
      "countBinCuts=%" IntEbmTypePrintf ", "
      "binCutsLowerBoundInclusive=%p, "
      "discretizedOut=%p"
      ,
      countSamples,
      static_cast<const void *>(featureValues),
      countBinCuts,
      static_cast<const void *>(binCutsLowerBoundInclusive),
      static_cast<void *>(discretizedOut)
   );

   IntEbmType ret;
   if(UNLIKELY(countSamples <= IntEbmType { 0 })) {
      if(UNLIKELY(countSamples < IntEbmType { 0 })) {
         LOG_0(TraceLevelError, "ERROR Discretize countSamples cannot be negative");
         ret = IntEbmType { 1 };
         goto exit_with_log;
      } else {
         EBM_ASSERT(IntEbmType { 0 } == countSamples);
         ret = IntEbmType { 0 };
         goto exit_with_log;
      }
   } else {
      if(UNLIKELY(!IsNumberConvertable<size_t>(countSamples))) {
         // this needs to point to real memory, otherwise it's invalid
         LOG_0(TraceLevelError, "ERROR Discretize countSamples was too large to fit into memory");
         ret = IntEbmType { 1 };
         goto exit_with_log;
      }

      const size_t cSamples = static_cast<size_t>(countSamples);

      if(IsMultiplyError(sizeof(*featureValues), cSamples)) {
         LOG_0(TraceLevelError, "ERROR Discretize countSamples was too large to fit into featureValues");
         ret = IntEbmType { 1 };
         goto exit_with_log;
      }

      if(IsMultiplyError(sizeof(*discretizedOut), cSamples)) {
         LOG_0(TraceLevelError, "ERROR Discretize countSamples was too large to fit into discretizedOut");
         ret = IntEbmType { 1 };
         goto exit_with_log;
      }

      if(UNLIKELY(nullptr == featureValues)) {
         LOG_0(TraceLevelError, "ERROR Discretize featureValues cannot be null");
         ret = IntEbmType { 1 };
         goto exit_with_log;
      }

      if(UNLIKELY(nullptr == discretizedOut)) {
         LOG_0(TraceLevelError, "ERROR Discretize discretizedOut cannot be null");
         ret = IntEbmType { 1 };
         goto exit_with_log;
      }

      const FloatEbmType * pValue = featureValues;
      const FloatEbmType * const pValueEnd = featureValues + cSamples;
      IntEbmType * pDiscretized = discretizedOut;

      if(UNLIKELY(countBinCuts <= IntEbmType { 0 })) {
         if(UNLIKELY(countBinCuts < IntEbmType { 0 })) {
            LOG_0(TraceLevelError, "ERROR Discretize countBinCuts cannot be negative");
            ret = IntEbmType { 1 };
            goto exit_with_log;
         }
         EBM_ASSERT(IntEbmType { 0 } == countBinCuts);

         do {
            const FloatEbmType val = *pValue;
            IntEbmType result;
            result = UNPREDICTABLE(std::isnan(val)) ? IntEbmType { 1 } : IntEbmType { 0 };
            *pDiscretized = result;
            ++pDiscretized;
            ++pValue;
         } while(LIKELY(pValueEnd != pValue));
         ret = IntEbmType { 0 };
         goto exit_with_log;
      }

      if(UNLIKELY(nullptr == binCutsLowerBoundInclusive)) {
         LOG_0(TraceLevelError, "ERROR Discretize binCutsLowerBoundInclusive cannot be null");
         ret = IntEbmType { 1 };
         goto exit_with_log;
      }

#ifndef NDEBUG
      if(IsNumberConvertable<size_t>(countBinCuts)) {
         const size_t cBinCuts = static_cast<size_t>(countBinCuts);
         size_t iDebug = 0;
         while(true) {
            EBM_ASSERT(!std::isnan(binCutsLowerBoundInclusive[iDebug]));
            EBM_ASSERT(!std::isinf(binCutsLowerBoundInclusive[iDebug]));

            size_t iDebugInc = iDebug + 1;
            if(cBinCuts <= iDebugInc) {
               break;
            }
            // if the values aren't increasing, we won't crash, but we'll return non-sensical bins.  That's a tollerable
            // failure though given that this check might be expensive if cBinCuts was large compared to cSamples
            EBM_ASSERT(binCutsLowerBoundInclusive[iDebug] < binCutsLowerBoundInclusive[iDebugInc]);
            iDebug = iDebugInc;
         }
      }
# endif // NDEBUG

      if(PREDICTABLE(IntEbmType { 1 } == countBinCuts)) {
         const FloatEbmType cut0 = binCutsLowerBoundInclusive[0];
         do {
            const FloatEbmType val = *pValue;
            IntEbmType result;

            result = UNPREDICTABLE(cut0 <= val) ? IntEbmType { 1 } : IntEbmType { 0 };
            result = UNPREDICTABLE(std::isnan(val)) ? IntEbmType { 2 } : result;

            *pDiscretized = result;
            ++pDiscretized;
            ++pValue;
         } while(LIKELY(pValueEnd != pValue));
         ret = IntEbmType { 0 };
         goto exit_with_log;
      }

      if(PREDICTABLE(IntEbmType { 2 } == countBinCuts)) {
         const FloatEbmType cut0 = binCutsLowerBoundInclusive[0];
         const FloatEbmType cut1 = binCutsLowerBoundInclusive[1];
         do {
            const FloatEbmType val = *pValue;
            IntEbmType result;

            result = UNPREDICTABLE(cut0 <= val) ? IntEbmType { 1 } : IntEbmType { 0 };
            result = UNPREDICTABLE(cut1 <= val) ? IntEbmType { 2 } : result;
            result = UNPREDICTABLE(std::isnan(val)) ? IntEbmType { 3 } : result;

            *pDiscretized = result;
            ++pDiscretized;
            ++pValue;
         } while(LIKELY(pValueEnd != pValue));
         ret = IntEbmType { 0 };
         goto exit_with_log;
      }

      if(PREDICTABLE(IntEbmType { 3 } == countBinCuts)) {
         const FloatEbmType cut0 = binCutsLowerBoundInclusive[0];
         const FloatEbmType cut1 = binCutsLowerBoundInclusive[1];
         const FloatEbmType cut2 = binCutsLowerBoundInclusive[2];
         do {
            const FloatEbmType val = *pValue;
            IntEbmType result;

            result = UNPREDICTABLE(cut0 <= val) ? IntEbmType { 1 } : IntEbmType { 0 };
            result = UNPREDICTABLE(cut1 <= val) ? IntEbmType { 2 } : result;
            result = UNPREDICTABLE(cut2 <= val) ? IntEbmType { 3 } : result;
            result = UNPREDICTABLE(std::isnan(val)) ? IntEbmType { 4 } : result;

            *pDiscretized = result;
            ++pDiscretized;
            ++pValue;
         } while(LIKELY(pValueEnd != pValue));
         ret = IntEbmType { 0 };
         goto exit_with_log;
      }

      if(PREDICTABLE(IntEbmType { 4 } == countBinCuts)) {
         const FloatEbmType cut0 = binCutsLowerBoundInclusive[0];
         const FloatEbmType cut1 = binCutsLowerBoundInclusive[1];
         const FloatEbmType cut2 = binCutsLowerBoundInclusive[2];
         const FloatEbmType cut3 = binCutsLowerBoundInclusive[3];
         do {
            const FloatEbmType val = *pValue;
            IntEbmType result;

            result = UNPREDICTABLE(cut0 <= val) ? IntEbmType { 1 } : IntEbmType { 0 };
            result = UNPREDICTABLE(cut1 <= val) ? IntEbmType { 2 } : result;
            result = UNPREDICTABLE(cut2 <= val) ? IntEbmType { 3 } : result;
            result = UNPREDICTABLE(cut3 <= val) ? IntEbmType { 4 } : result;
            result = UNPREDICTABLE(std::isnan(val)) ? IntEbmType { 5 } : result;

            *pDiscretized = result;
            ++pDiscretized;
            ++pValue;
         } while(LIKELY(pValueEnd != pValue));
         ret = IntEbmType { 0 };
         goto exit_with_log;
      }

      if(PREDICTABLE(IntEbmType { 5 } == countBinCuts)) {
         const FloatEbmType cut0 = binCutsLowerBoundInclusive[0];
         const FloatEbmType cut1 = binCutsLowerBoundInclusive[1];
         const FloatEbmType cut2 = binCutsLowerBoundInclusive[2];
         const FloatEbmType cut3 = binCutsLowerBoundInclusive[3];
         const FloatEbmType cut4 = binCutsLowerBoundInclusive[4];
         do {
            const FloatEbmType val = *pValue;
            IntEbmType result;

            result = UNPREDICTABLE(cut0 <= val) ? IntEbmType { 1 } : IntEbmType { 0 };
            result = UNPREDICTABLE(cut1 <= val) ? IntEbmType { 2 } : result;
            result = UNPREDICTABLE(cut2 <= val) ? IntEbmType { 3 } : result;
            result = UNPREDICTABLE(cut3 <= val) ? IntEbmType { 4 } : result;
            result = UNPREDICTABLE(cut4 <= val) ? IntEbmType { 5 } : result;
            result = UNPREDICTABLE(std::isnan(val)) ? IntEbmType { 6 } : result;

            *pDiscretized = result;
            ++pDiscretized;
            ++pValue;
         } while(LIKELY(pValueEnd != pValue));
         ret = IntEbmType { 0 };
         goto exit_with_log;
      }

      if(PREDICTABLE(IntEbmType { 6 } == countBinCuts)) {
         const FloatEbmType cut0 = binCutsLowerBoundInclusive[0];
         const FloatEbmType cut1 = binCutsLowerBoundInclusive[1];
         const FloatEbmType cut2 = binCutsLowerBoundInclusive[2];
         const FloatEbmType cut3 = binCutsLowerBoundInclusive[3];
         const FloatEbmType cut4 = binCutsLowerBoundInclusive[4];
         const FloatEbmType cut5 = binCutsLowerBoundInclusive[5];
         do {
            const FloatEbmType val = *pValue;
            IntEbmType result;

            result = UNPREDICTABLE(cut0 <= val) ? IntEbmType { 1 } : IntEbmType { 0 };
            result = UNPREDICTABLE(cut1 <= val) ? IntEbmType { 2 } : result;
            result = UNPREDICTABLE(cut2 <= val) ? IntEbmType { 3 } : result;
            result = UNPREDICTABLE(cut3 <= val) ? IntEbmType { 4 } : result;
            result = UNPREDICTABLE(cut4 <= val) ? IntEbmType { 5 } : result;
            result = UNPREDICTABLE(cut5 <= val) ? IntEbmType { 6 } : result;
            result = UNPREDICTABLE(std::isnan(val)) ? IntEbmType { 7 } : result;

            *pDiscretized = result;
            ++pDiscretized;
            ++pValue;
         } while(LIKELY(pValueEnd != pValue));
         ret = IntEbmType { 0 };
         goto exit_with_log;
      }

      if(PREDICTABLE(IntEbmType { 7 } == countBinCuts)) {
         // for digitization during training this all fits into registers, but if we evaluate the logits at the same
         // time for mains it doesn't quite fit into 16 registers.  We have 1 value that we load from memory to
         // process, 7 cut points, 8 logits bin cut logits (for prediction), and 1 logit for the missing value bin
         // (for prediction), which is 17 registers.  We need to load some things from memory therefore.  We can
         // do the final load for the missing bin logit after the comparison operator of isnan, so we could probably
         // just load 1 value, otherwise we'd need to load 2 values since loading them would overwrite one of them

         // TODO: since we can't fit everything into registers, perhaps we'd get better performance doing a full
         // binary search using memory, thus minimizing the number of comparisons?

         const FloatEbmType cut0 = binCutsLowerBoundInclusive[0];
         const FloatEbmType cut1 = binCutsLowerBoundInclusive[1];
         const FloatEbmType cut2 = binCutsLowerBoundInclusive[2];
         const FloatEbmType cut3 = binCutsLowerBoundInclusive[3];
         const FloatEbmType cut4 = binCutsLowerBoundInclusive[4];
         const FloatEbmType cut5 = binCutsLowerBoundInclusive[5];
         const FloatEbmType cut6 = binCutsLowerBoundInclusive[6];
         do {
            const FloatEbmType val = *pValue;
            IntEbmType result;

            result = UNPREDICTABLE(cut0 <= val) ? IntEbmType { 1 } : IntEbmType { 0 };
            result = UNPREDICTABLE(cut1 <= val) ? IntEbmType { 2 } : result;
            result = UNPREDICTABLE(cut2 <= val) ? IntEbmType { 3 } : result;
            result = UNPREDICTABLE(cut3 <= val) ? IntEbmType { 4 } : result;
            result = UNPREDICTABLE(cut4 <= val) ? IntEbmType { 5 } : result;
            result = UNPREDICTABLE(cut5 <= val) ? IntEbmType { 6 } : result;
            result = UNPREDICTABLE(cut6 <= val) ? IntEbmType { 7 } : result;
            result = UNPREDICTABLE(std::isnan(val)) ? IntEbmType { 8 } : result;

            *pDiscretized = result;
            ++pDiscretized;
            ++pValue;
         } while(LIKELY(pValueEnd != pValue));
         ret = IntEbmType { 0 };
         goto exit_with_log;
      }

      if(PREDICTABLE(IntEbmType { 8 } == countBinCuts)) {
         // TODO: for 8 specifically, we could probably do a special case of binary search with everything in
         //       registers.  It would take just 3 comparisons then instead of 8.  7 is less benefit since we'd then 
         //       still have 3 comparisons required plus one to handle the padding case at the end, so we'd have 
         //       4 total and more register moves.  8 is probably the only sweet spot special case.

         const FloatEbmType cut0 = binCutsLowerBoundInclusive[0];
         const FloatEbmType cut1 = binCutsLowerBoundInclusive[1];
         const FloatEbmType cut2 = binCutsLowerBoundInclusive[2];
         const FloatEbmType cut3 = binCutsLowerBoundInclusive[3];
         const FloatEbmType cut4 = binCutsLowerBoundInclusive[4];
         const FloatEbmType cut5 = binCutsLowerBoundInclusive[5];
         const FloatEbmType cut6 = binCutsLowerBoundInclusive[6];
         const FloatEbmType cut7 = binCutsLowerBoundInclusive[7];
         do {
            const FloatEbmType val = *pValue;
            IntEbmType result;

            result = UNPREDICTABLE(cut0 <= val) ? IntEbmType { 1 } : IntEbmType { 0 };
            result = UNPREDICTABLE(cut1 <= val) ? IntEbmType { 2 } : result;
            result = UNPREDICTABLE(cut2 <= val) ? IntEbmType { 3 } : result;
            result = UNPREDICTABLE(cut3 <= val) ? IntEbmType { 4 } : result;
            result = UNPREDICTABLE(cut4 <= val) ? IntEbmType { 5 } : result;
            result = UNPREDICTABLE(cut5 <= val) ? IntEbmType { 6 } : result;
            result = UNPREDICTABLE(cut6 <= val) ? IntEbmType { 7 } : result;
            result = UNPREDICTABLE(cut7 <= val) ? IntEbmType { 8 } : result;
            result = UNPREDICTABLE(std::isnan(val)) ? IntEbmType { 9 } : result;

            *pDiscretized = result;
            ++pDiscretized;
            ++pValue;
         } while(LIKELY(pValueEnd != pValue));
         ret = IntEbmType { 0 };
         goto exit_with_log;
      }

      FloatEbmType binCutsLowerBoundInclusiveCopy[1023];
      if(PREDICTABLE(countBinCuts <= IntEbmType { 15 })) {
         constexpr size_t cPower = 16;
         if(cPower * 4 <= cSamples) {
            static_assert(cPower - 1 <= sizeof(binCutsLowerBoundInclusiveCopy) /
               sizeof(binCutsLowerBoundInclusiveCopy[0]), "binCutsLowerBoundInclusiveCopy buffer not large enough");

            const size_t cBinCuts = static_cast<size_t>(countBinCuts);
            const size_t cSkip = cPower - 1 - cBinCuts;

            for(size_t i = 0; i < cSkip; ++i) {
               binCutsLowerBoundInclusiveCopy[i] = -std::numeric_limits<FloatEbmType>::infinity();
            }

            char * const pBaseValid = reinterpret_cast<char *>(binCutsLowerBoundInclusiveCopy + cSkip);
            memcpy(pBaseValid, binCutsLowerBoundInclusive, sizeof(*binCutsLowerBoundInclusive) * cBinCuts);

            const size_t missingVal = cBinCuts + size_t { 1 };
            const FloatEbmType firstComparison = binCutsLowerBoundInclusiveCopy[cPower / 2 - 1];
            do {
               const FloatEbmType val = *pValue;
               char * pResult = reinterpret_cast<char *>(binCutsLowerBoundInclusiveCopy);

               pResult += UNPREDICTABLE(firstComparison <= val) ? size_t { cPower / 2 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 3 } * sizeof(FloatEbmType)) <= val) ? size_t { 4 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 1 } * sizeof(FloatEbmType)) <= val) ? size_t { 2 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult) <= val) ? size_t { 1 } * sizeof(FloatEbmType) : size_t { 0 };

               size_t result = (pResult - pBaseValid) / sizeof(FloatEbmType);
               result = UNPREDICTABLE(std::isnan(val)) ? missingVal : result;

               *pDiscretized = static_cast<IntEbmType>(result);
               ++pDiscretized;
               ++pValue;
            } while(LIKELY(pValueEnd != pValue));
            ret = IntEbmType { 0 };
            goto exit_with_log;
         }
      } else if(PREDICTABLE(countBinCuts <= IntEbmType { 31 })) {
         constexpr size_t cPower = 32;
         if(cPower * 4 <= cSamples) {
            static_assert(cPower - 1 <= sizeof(binCutsLowerBoundInclusiveCopy) /
               sizeof(binCutsLowerBoundInclusiveCopy[0]), "binCutsLowerBoundInclusiveCopy buffer not large enough");

            const size_t cBinCuts = static_cast<size_t>(countBinCuts);
            const size_t cSkip = cPower - 1 - cBinCuts;

            for(size_t i = 0; i < cSkip; ++i) {
               binCutsLowerBoundInclusiveCopy[i] = -std::numeric_limits<FloatEbmType>::infinity();
            }

            char * const pBaseValid = reinterpret_cast<char *>(binCutsLowerBoundInclusiveCopy + cSkip);
            memcpy(pBaseValid, binCutsLowerBoundInclusive, sizeof(*binCutsLowerBoundInclusive) * cBinCuts);

            const size_t missingVal = cBinCuts + size_t { 1 };
            const FloatEbmType firstComparison = binCutsLowerBoundInclusiveCopy[cPower / 2 - 1];
            do {
               const FloatEbmType val = *pValue;
               char * pResult = reinterpret_cast<char *>(binCutsLowerBoundInclusiveCopy);

               pResult += UNPREDICTABLE(firstComparison <= val) ? size_t { cPower / 2 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 7 } * sizeof(FloatEbmType)) <= val) ? size_t { 8 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 3 } * sizeof(FloatEbmType)) <= val) ? size_t { 4 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 1 } * sizeof(FloatEbmType)) <= val) ? size_t { 2 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult) <= val) ? size_t { 1 } * sizeof(FloatEbmType) : size_t { 0 };

               size_t result = (pResult - pBaseValid) / sizeof(FloatEbmType);
               result = UNPREDICTABLE(std::isnan(val)) ? missingVal : result;

               *pDiscretized = static_cast<IntEbmType>(result);
               ++pDiscretized;
               ++pValue;
            } while(LIKELY(pValueEnd != pValue));
            ret = IntEbmType { 0 };
            goto exit_with_log;
         }
      } else if(PREDICTABLE(countBinCuts <= IntEbmType { 63 })) {
         constexpr size_t cPower = 64;
         if(cPower * 4 <= cSamples) {
            static_assert(cPower - 1 <= sizeof(binCutsLowerBoundInclusiveCopy) /
               sizeof(binCutsLowerBoundInclusiveCopy[0]), "binCutsLowerBoundInclusiveCopy buffer not large enough");

            const size_t cBinCuts = static_cast<size_t>(countBinCuts);
            const size_t cSkip = cPower - 1 - cBinCuts;

            for(size_t i = 0; i < cSkip; ++i) {
               binCutsLowerBoundInclusiveCopy[i] = -std::numeric_limits<FloatEbmType>::infinity();
            }

            char * const pBaseValid = reinterpret_cast<char *>(binCutsLowerBoundInclusiveCopy + cSkip);
            memcpy(pBaseValid, binCutsLowerBoundInclusive, sizeof(*binCutsLowerBoundInclusive) * cBinCuts);

            const size_t missingVal = cBinCuts + size_t { 1 };
            const FloatEbmType firstComparison = binCutsLowerBoundInclusiveCopy[cPower / 2 - 1];
            do {
               const FloatEbmType val = *pValue;
               char * pResult = reinterpret_cast<char *>(binCutsLowerBoundInclusiveCopy);

               pResult += UNPREDICTABLE(firstComparison <= val) ? size_t { cPower / 2 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 15 } * sizeof(FloatEbmType)) <= val) ? size_t { 16 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 7 } * sizeof(FloatEbmType)) <= val) ? size_t { 8 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 3 } * sizeof(FloatEbmType)) <= val) ? size_t { 4 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 1 } * sizeof(FloatEbmType)) <= val) ? size_t { 2 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult) <= val) ? size_t { 1 } * sizeof(FloatEbmType) : size_t { 0 };

               size_t result = (pResult - pBaseValid) / sizeof(FloatEbmType);
               result = UNPREDICTABLE(std::isnan(val)) ? missingVal : result;

               *pDiscretized = static_cast<IntEbmType>(result);
               ++pDiscretized;
               ++pValue;
            } while(LIKELY(pValueEnd != pValue));
            ret = IntEbmType { 0 };
            goto exit_with_log;
         }
      } else if(PREDICTABLE(countBinCuts <= IntEbmType { 127 })) {
         constexpr size_t cPower = 128;
         if(cPower * 4 <= cSamples) {
            static_assert(cPower - 1 <= sizeof(binCutsLowerBoundInclusiveCopy) /
               sizeof(binCutsLowerBoundInclusiveCopy[0]), "binCutsLowerBoundInclusiveCopy buffer not large enough");

            const size_t cBinCuts = static_cast<size_t>(countBinCuts);
            const size_t cSkip = cPower - 1 - cBinCuts;

            for(size_t i = 0; i < cSkip; ++i) {
               binCutsLowerBoundInclusiveCopy[i] = -std::numeric_limits<FloatEbmType>::infinity();
            }

            char * const pBaseValid = reinterpret_cast<char *>(binCutsLowerBoundInclusiveCopy + cSkip);
            memcpy(pBaseValid, binCutsLowerBoundInclusive, sizeof(*binCutsLowerBoundInclusive) * cBinCuts);

            const size_t missingVal = cBinCuts + size_t { 1 };
            const FloatEbmType firstComparison = binCutsLowerBoundInclusiveCopy[cPower / 2 - 1];
            do {
               const FloatEbmType val = *pValue;
               char * pResult = reinterpret_cast<char *>(binCutsLowerBoundInclusiveCopy);

               pResult += UNPREDICTABLE(firstComparison <= val) ? size_t { cPower / 2 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 31 } * sizeof(FloatEbmType)) <= val) ? size_t { 32 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 15 } * sizeof(FloatEbmType)) <= val) ? size_t { 16 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 7 } * sizeof(FloatEbmType)) <= val) ? size_t { 8 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 3 } * sizeof(FloatEbmType)) <= val) ? size_t { 4 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 1 } * sizeof(FloatEbmType)) <= val) ? size_t { 2 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult) <= val) ? size_t { 1 } * sizeof(FloatEbmType) : size_t { 0 };

               size_t result = (pResult - pBaseValid) / sizeof(FloatEbmType);
               result = UNPREDICTABLE(std::isnan(val)) ? missingVal : result;

               *pDiscretized = static_cast<IntEbmType>(result);
               ++pDiscretized;
               ++pValue;
            } while(LIKELY(pValueEnd != pValue));
            ret = IntEbmType { 0 };
            goto exit_with_log;
         }
      } else if(PREDICTABLE(countBinCuts <= IntEbmType { 255 })) {
         constexpr size_t cPower = 256;
         if(cPower * 4 <= cSamples) {
            static_assert(cPower - 1 <= sizeof(binCutsLowerBoundInclusiveCopy) /
               sizeof(binCutsLowerBoundInclusiveCopy[0]), "binCutsLowerBoundInclusiveCopy buffer not large enough");

            const size_t cBinCuts = static_cast<size_t>(countBinCuts);
            const size_t cSkip = cPower - 1 - cBinCuts;

            for(size_t i = 0; i < cSkip; ++i) {
               binCutsLowerBoundInclusiveCopy[i] = -std::numeric_limits<FloatEbmType>::infinity();
            }

            char * const pBaseValid = reinterpret_cast<char *>(binCutsLowerBoundInclusiveCopy + cSkip);
            memcpy(pBaseValid, binCutsLowerBoundInclusive, sizeof(*binCutsLowerBoundInclusive) * cBinCuts);

            const size_t missingVal = cBinCuts + size_t { 1 };
            const FloatEbmType firstComparison = binCutsLowerBoundInclusiveCopy[cPower / 2 - 1];
            do {
               const FloatEbmType val = *pValue;
               char * pResult = reinterpret_cast<char *>(binCutsLowerBoundInclusiveCopy);

               pResult += UNPREDICTABLE(firstComparison <= val) ? size_t { cPower / 2 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 63 } * sizeof(FloatEbmType)) <= val) ? size_t { 64 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 31 } * sizeof(FloatEbmType)) <= val) ? size_t { 32 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 15 } * sizeof(FloatEbmType)) <= val) ? size_t { 16 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 7 } * sizeof(FloatEbmType)) <= val) ? size_t { 8 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 3 } * sizeof(FloatEbmType)) <= val) ? size_t { 4 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 1 } * sizeof(FloatEbmType)) <= val) ? size_t { 2 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult) <= val) ? size_t { 1 } * sizeof(FloatEbmType) : size_t { 0 };

               size_t result = (pResult - pBaseValid) / sizeof(FloatEbmType);
               result = UNPREDICTABLE(std::isnan(val)) ? missingVal : result;

               *pDiscretized = static_cast<IntEbmType>(result);
               ++pDiscretized;
               ++pValue;
            } while(LIKELY(pValueEnd != pValue));
            ret = IntEbmType { 0 };
            goto exit_with_log;
         }
      } else if(PREDICTABLE(countBinCuts <= IntEbmType { 511 })) {
         constexpr size_t cPower = 512;
         if(cPower * 4 <= cSamples) {
            static_assert(cPower - 1 <= sizeof(binCutsLowerBoundInclusiveCopy) /
               sizeof(binCutsLowerBoundInclusiveCopy[0]), "binCutsLowerBoundInclusiveCopy buffer not large enough");

            const size_t cBinCuts = static_cast<size_t>(countBinCuts);
            const size_t cSkip = cPower - 1 - cBinCuts;

            for(size_t i = 0; i < cSkip; ++i) {
               binCutsLowerBoundInclusiveCopy[i] = -std::numeric_limits<FloatEbmType>::infinity();
            }

            char * const pBaseValid = reinterpret_cast<char *>(binCutsLowerBoundInclusiveCopy + cSkip);
            memcpy(pBaseValid, binCutsLowerBoundInclusive, sizeof(*binCutsLowerBoundInclusive) * cBinCuts);

            const size_t missingVal = cBinCuts + size_t { 1 };
            const FloatEbmType firstComparison = binCutsLowerBoundInclusiveCopy[cPower / 2 - 1];
            do {
               const FloatEbmType val = *pValue;
               char * pResult = reinterpret_cast<char *>(binCutsLowerBoundInclusiveCopy);

               pResult += UNPREDICTABLE(firstComparison <= val) ? size_t { cPower / 2 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 127 } * sizeof(FloatEbmType)) <= val) ? size_t { 128 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 63 } * sizeof(FloatEbmType)) <= val) ? size_t { 64 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 31 } * sizeof(FloatEbmType)) <= val) ? size_t { 32 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 15 } * sizeof(FloatEbmType)) <= val) ? size_t { 16 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 7 } * sizeof(FloatEbmType)) <= val) ? size_t { 8 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 3 } * sizeof(FloatEbmType)) <= val) ? size_t { 4 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 1 } * sizeof(FloatEbmType)) <= val) ? size_t { 2 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult) <= val) ? size_t { 1 } * sizeof(FloatEbmType) : size_t { 0 };

               size_t result = (pResult - pBaseValid) / sizeof(FloatEbmType);
               result = UNPREDICTABLE(std::isnan(val)) ? missingVal : result;

               *pDiscretized = static_cast<IntEbmType>(result);
               ++pDiscretized;
               ++pValue;
            } while(LIKELY(pValueEnd != pValue));
            ret = IntEbmType { 0 };
            goto exit_with_log;
         }
      } else if(PREDICTABLE(countBinCuts <= IntEbmType { 1023 })) {
         constexpr size_t cPower = 1024;
         if(cPower * 4 <= cSamples) {
            static_assert(cPower - 1 == sizeof(binCutsLowerBoundInclusiveCopy) /
               sizeof(binCutsLowerBoundInclusiveCopy[0]), "binCutsLowerBoundInclusiveCopy buffer not large enough");

            const size_t cBinCuts = static_cast<size_t>(countBinCuts);
            const size_t cSkip = cPower - 1 - cBinCuts;

            for(size_t i = 0; i < cSkip; ++i) {
               binCutsLowerBoundInclusiveCopy[i] = -std::numeric_limits<FloatEbmType>::infinity();
            }

            char * const pBaseValid = reinterpret_cast<char *>(binCutsLowerBoundInclusiveCopy + cSkip);
            memcpy(pBaseValid, binCutsLowerBoundInclusive, sizeof(*binCutsLowerBoundInclusive) * cBinCuts);

            const size_t missingVal = cBinCuts + size_t { 1 };
            const FloatEbmType firstComparison = binCutsLowerBoundInclusiveCopy[cPower / 2 - 1];
            do {
               const FloatEbmType val = *pValue;
               char * pResult = reinterpret_cast<char *>(binCutsLowerBoundInclusiveCopy);

               pResult += UNPREDICTABLE(firstComparison <= val) ? size_t { cPower / 2 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 255 } * sizeof(FloatEbmType)) <= val) ? size_t { 256 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 127 } * sizeof(FloatEbmType)) <= val) ? size_t { 128 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 63 } * sizeof(FloatEbmType)) <= val) ? size_t { 64 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 31 } * sizeof(FloatEbmType)) <= val) ? size_t { 32 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 15 } * sizeof(FloatEbmType)) <= val) ? size_t { 16 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 7 } * sizeof(FloatEbmType)) <= val) ? size_t { 8 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 3 } * sizeof(FloatEbmType)) <= val) ? size_t { 4 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 1 } * sizeof(FloatEbmType)) <= val) ? size_t { 2 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult) <= val) ? size_t { 1 } * sizeof(FloatEbmType) : size_t { 0 };

               size_t result = (pResult - pBaseValid) / sizeof(FloatEbmType);
               result = UNPREDICTABLE(std::isnan(val)) ? missingVal : result;

               *pDiscretized = static_cast<IntEbmType>(result);
               ++pDiscretized;
               ++pValue;
            } while(LIKELY(pValueEnd != pValue));
            ret = IntEbmType { 0 };
            goto exit_with_log;
         }
      }

      if(UNLIKELY(std::numeric_limits<IntEbmType>::max() == countBinCuts)) {
         // we convert back to IntEbmType when we return, and if countBinCuts is at the limit, then we don't
         // have any value to indicate missing
         LOG_0(TraceLevelError,
            "ERROR Discretize countBinCuts was too large to allow for a missing value placeholder");
         ret = IntEbmType { 1 };
         goto exit_with_log;
      }

      if(UNLIKELY(!IsNumberConvertable<size_t>(countBinCuts))) {
         // this needs to point to real memory, otherwise it's invalid
         LOG_0(TraceLevelError, "ERROR Discretize countBinCuts was too large to fit into memory");
         ret = IntEbmType { 1 };
         goto exit_with_log;
      }

      const size_t cBinCuts = static_cast<size_t>(countBinCuts);

      if(IsMultiplyError(sizeof(*binCutsLowerBoundInclusive), cBinCuts)) {
         LOG_0(TraceLevelError,
            "ERROR Discretize countBinCuts was too large to fit into binCutsLowerBoundInclusive");
         ret = IntEbmType { 1 };
         goto exit_with_log;
      }

      if(UNLIKELY(std::numeric_limits<size_t>::max() == cBinCuts)) {
         // we add 1 to cBinCuts as our missing value, so this addition must succeed
         LOG_0(TraceLevelError,
            "ERROR Discretize countBinCuts was too large to allow for a missing value placeholder");
         ret = IntEbmType { 1 };
         goto exit_with_log;
      }

      if(UNLIKELY(size_t { std::numeric_limits<ptrdiff_t>::max() } < cBinCuts)) {
         // the low value can increase until it's equal to cBinCuts, so cBinCuts must be expressable as a ptrdiff_t
         LOG_0(TraceLevelError,
            "ERROR Discretize countBinCuts was too large to allow for the binary search comparison");
         ret = IntEbmType { 1 };
         goto exit_with_log;
      }

      if(UNLIKELY(std::numeric_limits<size_t>::max() / size_t { 2 } + size_t { 1 } < cBinCuts)) {
         // our first operation towards getting the mid-point is to add the size_t low and size_t high, and that can't 
         // overflow, so check that the maximum high added to the maximum low (which is the high) don't exceed that value
         LOG_0(TraceLevelError,
            "ERROR Discretize countBinCuts was too large to allow for the binary search add");
         ret = IntEbmType { 1 };
         goto exit_with_log;
      }

      EBM_ASSERT(cBinCuts < std::numeric_limits<size_t>::max());
      const size_t missingVal = cBinCuts + size_t { 1 };
      EBM_ASSERT(size_t { 1 } <= cBinCuts);
      EBM_ASSERT(cBinCuts - size_t { 1 } <= size_t { std::numeric_limits<ptrdiff_t>::max() });
      const ptrdiff_t highStart = static_cast<ptrdiff_t>(cBinCuts - size_t { 1 });

      // if we're going to runroll our first loop, then we need to ensure that there's a next loop after the first
      // unrolled loop, otherwise we would need to check if we were done before the first real loop iteration.
      // To ensure we have 2 original loop iterations, we need 1 cut in the center, 1 cut above, and 1 cut below, so 3
      EBM_ASSERT(size_t { 3 } <= cBinCuts);
      const size_t firstMiddle = static_cast<size_t>(highStart) >> 1;
      EBM_ASSERT(firstMiddle < cBinCuts);
      const FloatEbmType firstMidVal = binCutsLowerBoundInclusive[firstMiddle];
      const ptrdiff_t firstMidLow = static_cast<ptrdiff_t>(firstMiddle) + ptrdiff_t { 1 };
      const ptrdiff_t firstMidHigh = static_cast<ptrdiff_t>(firstMiddle) - ptrdiff_t { 1 };

      do {
         const FloatEbmType val = *pValue;
         size_t middle = missingVal;
         if(PREDICTABLE(!std::isnan(val))) {
            ptrdiff_t high = UNPREDICTABLE(firstMidVal <= val) ? highStart : firstMidHigh;
            ptrdiff_t low = UNPREDICTABLE(firstMidVal <= val) ? firstMidLow : ptrdiff_t { 0 };
            FloatEbmType midVal;
            do {
               EBM_ASSERT(ptrdiff_t { 0 } <= low && static_cast<size_t>(low) < cBinCuts);
               EBM_ASSERT(ptrdiff_t { 0 } <= high && static_cast<size_t>(high) < cBinCuts);
               EBM_ASSERT(low <= high);
               // low is equal or lower than high, so summing them can't exceed 2 * high, and after division it
               // can't be higher than high, so middle can't overflow ptrdiff_t after the division since high
               // is already a ptrdiff_t.  Generally the maximum positive value of a ptrdiff_t can be doubled 
               // when converted to a size_t, although that isn't guaranteed.  A more correct statement is that
               // the following must be false (which we check above):
               // "std::numeric_limits<size_t>::max() / 2 < cBinCuts - 1"
               EBM_ASSERT(!IsAddError(static_cast<size_t>(low), static_cast<size_t>(high)));
               middle = (static_cast<size_t>(low) + static_cast<size_t>(high)) >> 1;
               EBM_ASSERT(middle <= static_cast<size_t>(high));
               EBM_ASSERT(middle < cBinCuts);
               midVal = binCutsLowerBoundInclusive[middle];
               EBM_ASSERT(middle < size_t { std::numeric_limits<ptrdiff_t>::max() });
               low = UNPREDICTABLE(midVal <= val) ? static_cast<ptrdiff_t>(middle) + ptrdiff_t { 1 } : low;
               EBM_ASSERT(ptrdiff_t { 0 } <= low && static_cast<size_t>(low) <= cBinCuts);
               high = UNPREDICTABLE(midVal <= val) ? high : static_cast<ptrdiff_t>(middle) - ptrdiff_t { 1 };
               EBM_ASSERT(ptrdiff_t { -1 } <= high && high <= highStart);

               // high can become -1 in some cases, so it needs to be ptrdiff_t.  It's tempting to try and change
               // this code and use the Hermann Bottenbruch version that checks for low != high in the loop comparison
               // since then we wouldn't have negative values and we could use size_t, but unfortunately that version
               // has a check at the end where we'd need to fetch binCutsLowerBoundInclusive[low] after exiting the 
               // loop, so this version we have here is faster given that we only need to compare to a value that
               // we've already fetched from memory.  Also, this version makes slightly faster progress since
               // it does middle + 1 AND middle - 1 instead of just middle - 1, so it often eliminates one loop
               // iteration.  In practice this version will always work since no floating point type is less than 4
               // bytes, so we shouldn't have difficulty expressing any indexes with ptrdiff_t, and our indexes
               // for accessing memory are always size_t, so those should always work.
            } while(LIKELY(low <= high));
            EBM_ASSERT(size_t { 0 } <= middle && middle < cBinCuts);
            middle = UNPREDICTABLE(midVal <= val) ? middle + size_t { 1 } : middle;
            EBM_ASSERT(size_t { 0 } <= middle && middle <= cBinCuts);
         }
         EBM_ASSERT(IsNumberConvertable<IntEbmType>(middle));
         *pDiscretized = static_cast<IntEbmType>(middle);
         ++pDiscretized;
         ++pValue;
      } while(LIKELY(pValueEnd != pValue));
      ret = IntEbmType { 0 };
   }

exit_with_log:;

   LOG_COUNTED_N(
      &g_cLogExitDiscretizeParametersMessages, 
      TraceLevelInfo, 
      TraceLevelVerbose, 
      "Exited Discretize: "
      "return=%" IntEbmTypePrintf
      ,
      ret
   );

   return ret;
}

