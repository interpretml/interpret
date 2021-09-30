// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

// TODO: use noexcept throughout our codebase (exception extern "C" functions) !  The compiler can optimize functions better if it knows there are no exceptions
// TODO: review all the C++ library calls, including things like std::abs and verify that none of them throw exceptions, otherwise use the C versions that provide this guarantee

#include <stddef.h> // size_t, ptrdiff_t
#include <limits> // std::numeric_limits

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "ebm_internal.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

EBM_NATIVE_IMPORT_EXPORT_BODY ErrorEbmType EBM_NATIVE_CALLING_CONVENTION Softmax(
   IntEbmType countTargetClasses,
   IntEbmType countSamples,
   const FloatEbmType * logits,
   FloatEbmType * probabilitiesOut
) {
   UNUSED(countTargetClasses);
   UNUSED(countSamples);
   UNUSED(logits);
   UNUSED(probabilitiesOut);

   return Error_UnexpectedInternal;

   //if(2 != countTargetClasses) {
   //   // TODO: handle multiclass
   //   exit(1);
   //};

   //UNUSED(countTargetClasses); // TODO: use this
   //for(size_t i = 0; i < static_cast<size_t>(countSamples); ++i) {
   //   // NOTE: we use the non-approximate std::exp because we want our predictions to match what other softmax functions
   //   // will generate instead of the approximation, and ordering is more sensitive to noise than boosting
   //   const FloatEbmType odds = std::exp(logits[i]);
   //   probabilitiesOut[i] = odds / (FloatEbmType { 1 } + odds);
   //}
   //return Error_None;
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

EBM_NATIVE_IMPORT_EXPORT_BODY IntEbmType EBM_NATIVE_CALLING_CONVENTION DiscretizeOne(
   const FloatEbmType featureValue,
   IntEbmType countCuts,
   const FloatEbmType * cutsLowerBoundInclusive
) {
   // unlike all of our other public interfaces we don't check our inputs for validity.  The caller is expected
   // to get these right otherwise there will be a segfault.  We do this because we want lightening fast prediction
   // speed.  We can still use EBM_ASSERT though to catch input errors on debug builds, so do that.

   EBM_ASSERT(nullptr != cutsLowerBoundInclusive);

   EBM_ASSERT(IntEbmType { 0 } <= countCuts);
   // IntEbmType::max should be the maximum number of bins that we allow.  This prevents our caller from accidentally
   // overflowing when using either the count (eg: 255) or the index (eg: 0-254).  The 0th bin is the special missing
   // bin, so we have max-1 normal bins.  We have 1 less cuts than we have bins (1 cut means 2 bins), so we can
   // have a maximum of max - 2 cuts.
   EBM_ASSERT(countCuts <= std::numeric_limits<IntEbmType>::max() - IntEbmType { 2 });
   // cutsLowerBoundInclusive needs to hold all the cuts, so we should be able to at least convert countCuts to size_t
   EBM_ASSERT(!IsConvertError<size_t>(countCuts));
   // cutsLowerBoundInclusive needs to hold all the cuts, so all the bytes need to be addressable
   EBM_ASSERT(!IsMultiplyError(sizeof(*cutsLowerBoundInclusive), static_cast<size_t>(countCuts)));

   // extra restrictions we take on from our binary search code

   // we use ptrdiff_t as our indexes in the binary search code since it's slightly faster
   EBM_ASSERT(!IsConvertError<ptrdiff_t>(countCuts));
   // we add 1 to static_cast<size_t>(countCuts) as our missing value, so this addition must succeed
   EBM_ASSERT(static_cast<size_t>(countCuts) < std::numeric_limits<size_t>::max());
   // the low value can increase until it's equal to cCuts, so cCuts must be expressable as a ptrdiff_t
   // we need to keep low as a ptrdiff_t since we compare it right after with high, which can be -1
   EBM_ASSERT(static_cast<size_t>(countCuts) <= size_t { std::numeric_limits<ptrdiff_t>::max() });
   // our first operation towards getting the mid-point is to add the size_t low and size_t high, and that can't 
   // overflow, so check that the maximum high added to the maximum low (which is the high) don't exceed that value
   EBM_ASSERT(static_cast<size_t>(countCuts) <= std::numeric_limits<size_t>::max() / size_t { 2 } + size_t { 1 });

   if(PREDICTABLE(std::isnan(featureValue))) {
      return IntEbmType { 0 };
   }
   if(UNLIKELY(countCuts <= IntEbmType { 0 })) {
      return IntEbmType { 1 };
   }

   size_t middle;
   const ptrdiff_t highStart = static_cast<ptrdiff_t>(countCuts) - ptrdiff_t { 1 };
   ptrdiff_t high = highStart;
   ptrdiff_t low = ptrdiff_t { 0 };
   FloatEbmType midVal;
   do {
      EBM_ASSERT(ptrdiff_t { 0 } <= low && static_cast<size_t>(low) < static_cast<size_t>(countCuts));
      EBM_ASSERT(ptrdiff_t { 0 } <= high && static_cast<size_t>(high) < static_cast<size_t>(countCuts));
      EBM_ASSERT(low <= high);
      // low is equal or lower than high, so summing them can't exceed 2 * high, and after division it
      // can't be higher than high, so middle can't overflow ptrdiff_t after the division since high
      // is already a ptrdiff_t.  Generally the maximum positive value of a ptrdiff_t can be doubled 
      // when converted to a size_t, although that isn't guaranteed.  A more correct statement is that
      // the following must be false (which we check above):
      // "std::numeric_limits<size_t>::max() / 2 < static_cast<size_t>(countCuts) - 1"
      EBM_ASSERT(!IsAddError(static_cast<size_t>(low), static_cast<size_t>(high)));
      middle = (static_cast<size_t>(low) + static_cast<size_t>(high)) >> 1;
      EBM_ASSERT(middle <= static_cast<size_t>(high));
      EBM_ASSERT(middle < static_cast<size_t>(countCuts));
      midVal = cutsLowerBoundInclusive[middle];
      EBM_ASSERT(middle < size_t { std::numeric_limits<ptrdiff_t>::max() });
      low = UNPREDICTABLE(midVal <= featureValue) ? static_cast<ptrdiff_t>(middle) + ptrdiff_t { 1 } : low;
      EBM_ASSERT(ptrdiff_t { 0 } <= low && static_cast<size_t>(low) <= static_cast<size_t>(countCuts));
      high = UNPREDICTABLE(midVal <= featureValue) ? high : static_cast<ptrdiff_t>(middle) - ptrdiff_t { 1 };
      EBM_ASSERT(ptrdiff_t { -1 } <= high && high <= highStart);

      // high can become -1 in some cases, so it needs to be ptrdiff_t.  It's tempting to try and change
      // this code and use the Hermann Bottenbruch version that checks for low != high in the loop comparison
      // since then we wouldn't have negative values and we could use size_t, but unfortunately that version
      // has a check at the end where we'd need to fetch cutsLowerBoundInclusive[low] after exiting the 
      // loop, so this version we have here is faster given that we only need to compare to a value that
      // we've already fetched from memory.  Also, this version makes slightly faster progress since
      // it does middle + 1 AND middle - 1 instead of just middle - 1, so it often eliminates one loop
      // iteration.  In practice this version will always work since no floating point type is less than 4
      // bytes, so we shouldn't have difficulty expressing any indexes with ptrdiff_t, and our indexes
      // for accessing memory are always size_t, so those should always work.
   } while(LIKELY(low <= high));
   EBM_ASSERT(size_t { 0 } <= middle && middle < static_cast<size_t>(countCuts));
   middle = UNPREDICTABLE(midVal <= featureValue) ? middle + size_t { 2 } : middle + size_t { 1 };
   EBM_ASSERT(size_t { 1 } <= middle && middle <= size_t { 1 } + static_cast<size_t>(countCuts));
   EBM_ASSERT(!IsConvertError<IntEbmType>(middle));
   return static_cast<IntEbmType>(middle);
}

// don't bother using a lock here.  We don't care if an extra log message is written out due to thread parallism
static int g_cLogEnterDiscretizeParametersMessages = 25;
static int g_cLogExitDiscretizeParametersMessages = 25;

EBM_NATIVE_IMPORT_EXPORT_BODY ErrorEbmType EBM_NATIVE_CALLING_CONVENTION Discretize(
   IntEbmType countSamples,
   const FloatEbmType * featureValues,
   IntEbmType countCuts,
   const FloatEbmType * cutsLowerBoundInclusive,
   IntEbmType * discretizedOut
) {
   // make the 0th bin always the missing value.  This makes cutting mains easier, since we always know where the 
   // missing bin will be, and also the first non-missing bin.  We can also increment the pointer to the histogram
   // to the first non-missing bin and reduce our bin index numbers by one, which will allow us to compress
   // binary features into 1 bit still.  It will make handling tensors with missing or no missing easier since
   // we'll always know how to skip the missing slice if desired.  None of these things are as easy if the missing
   // bin is in the Nth item because we then need to know what N is and use multiplication and badly ordered memory
   // accesses to reach it if we want to use the missing bin during cutting.  Lastly, in higher level languages, it's
   // easier to detect missing values in the discretized data, since it's always just in index zero.
   // Finally, the way we compute our tensor sums allows us to calculate a slice of the tensor at the 0th index
   // with just a single check, so having the missing value at the 0th index allows fast lookup of the missing values
   // so we can do things like try putting the missing values into the left and right statistics of any cuts easily
   //
   // this function has exactly the same behavior as numpy.digitize, including the lower bound inclusive semantics
   
   // TODO: we want this function super-fast to handle the case where someone calls Discretize repeatedly
   //       once per-sample.  We can make this faster by eliminating a lot of the initial checks by:
   //       - eliminate the checks for nullptr on our input points.  We just crash if we get these illegal inputs
   //         we can't handle bad input points in general, so this is just accepting a reality for a tradeoff in speed
   //       - create a function that converts any signed value into the proper unsigned value 
   //         we can do this using std::make_unsigned to find the type and then use the twos compliment
   //         trick in RandomStream to convert the signed value into the corresponding unsigned value
   //         (also replace the code in RandomStream with this new function).  This function will be 
   //         a no-op in assembly.
   //       - we check for both negative AND positive countCuts values that are too high.  We can
   //         combine these checks by first converting negatives into big positive unsigned values
   //         which are guaranteed to be larger than our largest valid index.  We then find via
   //         constexpr the biggest of the maxes that we care about, and then make one comparison
   //         for all these conditions.  We might have code that differentiates the results if we trigger
   //         an error return
   //       - we have comparisons to both size_t and IntEbmType max values, which we can collapse into
   //         just a single max value comparison.  If the max unsigned size_t is bigger than the max unsigned
   //         UIntEbmType then do the comparison in the bigger domain.
   //       - we can write a constexpr function that takes the value we have, and the max value in
   //         the same type and the max value in annother type and does all the right things to compare
   //         them properly.  Use SINAFE to have 2 functions (one where the initial value type is lower
   //         and another where it is higher).  We could probably use this function in a number of places
   //       - special case a function that handles when countSamples is 1 and make the check at the top
   //         since getting just 1 sample is going to be the norm a lot.  In that case it doesn't
   //         make sense to check all the other possible values or copy the cuts to new memory or use SIMD
   //         we can even eliminate the loop stuff we construct to handle multiple samples and just write
   //         directly to the pointers
   //       - we'll probably need to check countCuts for zero, which we can do by first converting it
   //         to unsigned, then decrementing it (which is legal in C++) to a huge number if it was zero
   //         then doing our upper bound comparison all in one check.  We can then filter our 0 ==countCuts
   //         after that as a special case

   LOG_COUNTED_N(
      &g_cLogEnterDiscretizeParametersMessages,
      TraceLevelInfo,
      TraceLevelVerbose,
      "Entered Discretize: "
      "countSamples=%" IntEbmTypePrintf ", "
      "featureValues=%p, "
      "countCuts=%" IntEbmTypePrintf ", "
      "cutsLowerBoundInclusive=%p, "
      "discretizedOut=%p"
      ,
      countSamples,
      static_cast<const void *>(featureValues),
      countCuts,
      static_cast<const void *>(cutsLowerBoundInclusive),
      static_cast<void *>(discretizedOut)
   );

   ErrorEbmType ret;
   if(UNLIKELY(countSamples <= IntEbmType { 0 })) {
      if(UNLIKELY(countSamples < IntEbmType { 0 })) {
         LOG_0(TraceLevelError, "ERROR Discretize countSamples cannot be negative");
         ret = Error_IllegalParamValue;
         goto exit_with_log;
      } else {
         EBM_ASSERT(IntEbmType { 0 } == countSamples);
         ret = Error_None;
         goto exit_with_log;
      }
   } else {
      if(UNLIKELY(IsConvertError<size_t>(countSamples))) {
         // this needs to point to real memory, otherwise it's invalid
         LOG_0(TraceLevelError, "ERROR Discretize countSamples was too large to fit into memory");
         ret = Error_IllegalParamValue;
         goto exit_with_log;
      }

      const size_t cSamples = static_cast<size_t>(countSamples);

      if(IsMultiplyError(sizeof(*featureValues), cSamples)) {
         LOG_0(TraceLevelError, "ERROR Discretize countSamples was too large to fit into featureValues");
         ret = Error_IllegalParamValue;
         goto exit_with_log;
      }

      if(IsMultiplyError(sizeof(*discretizedOut), cSamples)) {
         LOG_0(TraceLevelError, "ERROR Discretize countSamples was too large to fit into discretizedOut");
         ret = Error_IllegalParamValue;
         goto exit_with_log;
      }

      if(UNLIKELY(nullptr == featureValues)) {
         LOG_0(TraceLevelError, "ERROR Discretize featureValues cannot be null");
         ret = Error_IllegalParamValue;
         goto exit_with_log;
      }

      if(UNLIKELY(nullptr == discretizedOut)) {
         LOG_0(TraceLevelError, "ERROR Discretize discretizedOut cannot be null");
         ret = Error_IllegalParamValue;
         goto exit_with_log;
      }

      const FloatEbmType * pValue = featureValues;
      const FloatEbmType * const pValueEnd = featureValues + cSamples;
      IntEbmType * pDiscretized = discretizedOut;

      if(UNLIKELY(countCuts <= IntEbmType { 0 })) {
         if(UNLIKELY(countCuts < IntEbmType { 0 })) {
            LOG_0(TraceLevelError, "ERROR Discretize countCuts cannot be negative");
            ret = Error_IllegalParamValue;
            goto exit_with_log;
         }
         EBM_ASSERT(IntEbmType { 0 } == countCuts);

         do {
            const FloatEbmType val = *pValue;
            IntEbmType result;
            result = UNPREDICTABLE(std::isnan(val)) ? IntEbmType { 0 } : IntEbmType { 1 };
            EBM_ASSERT(result == DiscretizeOne(val, countCuts, cutsLowerBoundInclusive));
            *pDiscretized = result;
            ++pDiscretized;
            ++pValue;
         } while(LIKELY(pValueEnd != pValue));
         ret = Error_None;
         goto exit_with_log;
      }

      if(UNLIKELY(nullptr == cutsLowerBoundInclusive)) {
         LOG_0(TraceLevelError, "ERROR Discretize cutsLowerBoundInclusive cannot be null");
         ret = Error_IllegalParamValue;
         goto exit_with_log;
      }

#ifndef NDEBUG
      if(!IsConvertError<size_t>(countCuts)) {
         const size_t cCuts = static_cast<size_t>(countCuts);
         size_t iDebug = 0;
         while(true) {
            EBM_ASSERT(!std::isnan(cutsLowerBoundInclusive[iDebug]));
            EBM_ASSERT(!std::isinf(cutsLowerBoundInclusive[iDebug]));

            size_t iDebugInc = iDebug + 1;
            if(cCuts <= iDebugInc) {
               break;
            }
            // if the values aren't increasing, we won't crash, but we'll return non-sensical bins.  That's a tollerable
            // failure though given that this check might be expensive if cCuts was large compared to cSamples
            EBM_ASSERT(cutsLowerBoundInclusive[iDebug] < cutsLowerBoundInclusive[iDebugInc]);
            iDebug = iDebugInc;
         }
      }
# endif // NDEBUG

      if(PREDICTABLE(IntEbmType { 1 } == countCuts)) {
         const FloatEbmType cut0 = cutsLowerBoundInclusive[0];
         do {
            const FloatEbmType val = *pValue;
            IntEbmType result;

            result = UNPREDICTABLE(cut0 <= val) ? IntEbmType { 2 } : IntEbmType { 1 };
            result = UNPREDICTABLE(std::isnan(val)) ? IntEbmType { 0 } : result;

            EBM_ASSERT(result == DiscretizeOne(val, countCuts, cutsLowerBoundInclusive));

            *pDiscretized = result;
            ++pDiscretized;
            ++pValue;
         } while(LIKELY(pValueEnd != pValue));
         ret = Error_None;
         goto exit_with_log;
      }

      if(PREDICTABLE(IntEbmType { 2 } == countCuts)) {
         const FloatEbmType cut0 = cutsLowerBoundInclusive[0];
         const FloatEbmType cut1 = cutsLowerBoundInclusive[1];
         do {
            const FloatEbmType val = *pValue;
            IntEbmType result;

            result = UNPREDICTABLE(cut0 <= val) ? IntEbmType { 2 } : IntEbmType { 1 };
            result = UNPREDICTABLE(cut1 <= val) ? IntEbmType { 3 } : result;
            result = UNPREDICTABLE(std::isnan(val)) ? IntEbmType { 0 } : result;

            EBM_ASSERT(result == DiscretizeOne(val, countCuts, cutsLowerBoundInclusive));

            *pDiscretized = result;
            ++pDiscretized;
            ++pValue;
         } while(LIKELY(pValueEnd != pValue));
         ret = Error_None;
         goto exit_with_log;
      }

      if(PREDICTABLE(IntEbmType { 3 } == countCuts)) {
         const FloatEbmType cut0 = cutsLowerBoundInclusive[0];
         const FloatEbmType cut1 = cutsLowerBoundInclusive[1];
         const FloatEbmType cut2 = cutsLowerBoundInclusive[2];
         do {
            const FloatEbmType val = *pValue;
            IntEbmType result;

            result = UNPREDICTABLE(cut0 <= val) ? IntEbmType { 2 } : IntEbmType { 1 };
            result = UNPREDICTABLE(cut1 <= val) ? IntEbmType { 3 } : result;
            result = UNPREDICTABLE(cut2 <= val) ? IntEbmType { 4 } : result;
            result = UNPREDICTABLE(std::isnan(val)) ? IntEbmType { 0 } : result;

            EBM_ASSERT(result == DiscretizeOne(val, countCuts, cutsLowerBoundInclusive));

            *pDiscretized = result;
            ++pDiscretized;
            ++pValue;
         } while(LIKELY(pValueEnd != pValue));
         ret = Error_None;
         goto exit_with_log;
      }

      if(PREDICTABLE(IntEbmType { 4 } == countCuts)) {
         const FloatEbmType cut0 = cutsLowerBoundInclusive[0];
         const FloatEbmType cut1 = cutsLowerBoundInclusive[1];
         const FloatEbmType cut2 = cutsLowerBoundInclusive[2];
         const FloatEbmType cut3 = cutsLowerBoundInclusive[3];
         do {
            const FloatEbmType val = *pValue;
            IntEbmType result;

            result = UNPREDICTABLE(cut0 <= val) ? IntEbmType { 2 } : IntEbmType { 1 };
            result = UNPREDICTABLE(cut1 <= val) ? IntEbmType { 3 } : result;
            result = UNPREDICTABLE(cut2 <= val) ? IntEbmType { 4 } : result;
            result = UNPREDICTABLE(cut3 <= val) ? IntEbmType { 5 } : result;
            result = UNPREDICTABLE(std::isnan(val)) ? IntEbmType { 0 } : result;

            EBM_ASSERT(result == DiscretizeOne(val, countCuts, cutsLowerBoundInclusive));

            *pDiscretized = result;
            ++pDiscretized;
            ++pValue;
         } while(LIKELY(pValueEnd != pValue));
         ret = Error_None;
         goto exit_with_log;
      }

      if(PREDICTABLE(IntEbmType { 5 } == countCuts)) {
         const FloatEbmType cut0 = cutsLowerBoundInclusive[0];
         const FloatEbmType cut1 = cutsLowerBoundInclusive[1];
         const FloatEbmType cut2 = cutsLowerBoundInclusive[2];
         const FloatEbmType cut3 = cutsLowerBoundInclusive[3];
         const FloatEbmType cut4 = cutsLowerBoundInclusive[4];
         do {
            const FloatEbmType val = *pValue;
            IntEbmType result;

            result = UNPREDICTABLE(cut0 <= val) ? IntEbmType { 2 } : IntEbmType { 1 };
            result = UNPREDICTABLE(cut1 <= val) ? IntEbmType { 3 } : result;
            result = UNPREDICTABLE(cut2 <= val) ? IntEbmType { 4 } : result;
            result = UNPREDICTABLE(cut3 <= val) ? IntEbmType { 5 } : result;
            result = UNPREDICTABLE(cut4 <= val) ? IntEbmType { 6 } : result;
            result = UNPREDICTABLE(std::isnan(val)) ? IntEbmType { 0 } : result;

            EBM_ASSERT(result == DiscretizeOne(val, countCuts, cutsLowerBoundInclusive));

            *pDiscretized = result;
            ++pDiscretized;
            ++pValue;
         } while(LIKELY(pValueEnd != pValue));
         ret = Error_None;
         goto exit_with_log;
      }

      if(PREDICTABLE(IntEbmType { 6 } == countCuts)) {
         const FloatEbmType cut0 = cutsLowerBoundInclusive[0];
         const FloatEbmType cut1 = cutsLowerBoundInclusive[1];
         const FloatEbmType cut2 = cutsLowerBoundInclusive[2];
         const FloatEbmType cut3 = cutsLowerBoundInclusive[3];
         const FloatEbmType cut4 = cutsLowerBoundInclusive[4];
         const FloatEbmType cut5 = cutsLowerBoundInclusive[5];
         do {
            const FloatEbmType val = *pValue;
            IntEbmType result;

            result = UNPREDICTABLE(cut0 <= val) ? IntEbmType { 2 } : IntEbmType { 1 };
            result = UNPREDICTABLE(cut1 <= val) ? IntEbmType { 3 } : result;
            result = UNPREDICTABLE(cut2 <= val) ? IntEbmType { 4 } : result;
            result = UNPREDICTABLE(cut3 <= val) ? IntEbmType { 5 } : result;
            result = UNPREDICTABLE(cut4 <= val) ? IntEbmType { 6 } : result;
            result = UNPREDICTABLE(cut5 <= val) ? IntEbmType { 7 } : result;
            result = UNPREDICTABLE(std::isnan(val)) ? IntEbmType { 0 } : result;

            EBM_ASSERT(result == DiscretizeOne(val, countCuts, cutsLowerBoundInclusive));

            *pDiscretized = result;
            ++pDiscretized;
            ++pValue;
         } while(LIKELY(pValueEnd != pValue));
         ret = Error_None;
         goto exit_with_log;
      }

      FloatEbmType cutsLowerBoundInclusiveCopy[1023];
      // the only value that should be less than this one is NaN, which always returns false for comparisons
      // that are not NaN.  If we have a NaN value we expect this to convert us to the 0th bin for missing
      cutsLowerBoundInclusiveCopy[0] = -std::numeric_limits<FloatEbmType>::infinity();

      // it's always legal in C++ to convert a signed value to unsigned.  We check below for out of bounds if needed
      const size_t cCuts = static_cast<size_t>(countCuts);

      if(PREDICTABLE(countCuts <= IntEbmType { 14 })) {
         constexpr size_t cPower = 16;
         if(cPower * 4 <= cSamples) {
            static_assert(cPower - 1 <= sizeof(cutsLowerBoundInclusiveCopy) /
               sizeof(cutsLowerBoundInclusiveCopy[0]), "cutsLowerBoundInclusiveCopy buffer not large enough");

            memcpy(
               size_t { 1 } + cutsLowerBoundInclusiveCopy,
               cutsLowerBoundInclusive,
               sizeof(*cutsLowerBoundInclusive) * cCuts
            );

            if(LIKELY(cCuts != cPower - size_t { 2 })) {
               FloatEbmType * pFill = &cutsLowerBoundInclusiveCopy[cCuts + size_t { 1 }];
               const FloatEbmType * const pEndFill = &cutsLowerBoundInclusiveCopy[cPower - size_t { 1 }];
               do {
                  // NaN will always move us downwards into the region of valid cuts.  The first cut is always
                  // guaranteed to be non-NaN, so if we have a missing (NaN) value, then the binary search will
                  // go low first and never hit these upper NaN values.
                  *pFill = std::numeric_limits<FloatEbmType>::quiet_NaN();
                  ++pFill;
               } while(LIKELY(pEndFill != pFill));
            }

            const FloatEbmType firstComparison = cutsLowerBoundInclusiveCopy[cPower / 2 - 1];
            do {
               const FloatEbmType val = *pValue;
               char * pResult = reinterpret_cast<char *>(cutsLowerBoundInclusiveCopy);

               pResult += UNPREDICTABLE(firstComparison <= val) ? size_t { cPower / 2 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 3 } * sizeof(FloatEbmType)) <= val) ? size_t { 4 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 1 } * sizeof(FloatEbmType)) <= val) ? size_t { 2 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult) <= val) ? size_t { 1 } * sizeof(FloatEbmType) : size_t { 0 };

               const size_t result = 
                  (pResult - reinterpret_cast<char *>(cutsLowerBoundInclusiveCopy)) / sizeof(FloatEbmType);

               EBM_ASSERT(static_cast<IntEbmType>(result) == DiscretizeOne(val, countCuts, cutsLowerBoundInclusive));

               *pDiscretized = static_cast<IntEbmType>(result);
               ++pDiscretized;
               ++pValue;
            } while(LIKELY(pValueEnd != pValue));
            ret = Error_None;
            goto exit_with_log;
         }
      } else if(PREDICTABLE(countCuts <= IntEbmType { 30 })) {
         constexpr size_t cPower = 32;
         if(cPower * 4 <= cSamples) {
            static_assert(cPower - 1 <= sizeof(cutsLowerBoundInclusiveCopy) /
               sizeof(cutsLowerBoundInclusiveCopy[0]), "cutsLowerBoundInclusiveCopy buffer not large enough");

            memcpy(
               size_t { 1 } + cutsLowerBoundInclusiveCopy,
               cutsLowerBoundInclusive,
               sizeof(*cutsLowerBoundInclusive) * cCuts
            );

            if(LIKELY(cCuts != cPower - size_t { 2 })) {
               FloatEbmType * pFill = &cutsLowerBoundInclusiveCopy[cCuts + size_t { 1 }];
               const FloatEbmType * const pEndFill = &cutsLowerBoundInclusiveCopy[cPower - size_t { 1 }];
               do {
                  // NaN will always move us downwards into the region of valid cuts.  The first cut is always
                  // guaranteed to be non-NaN, so if we have a missing (NaN) value, then the binary search will
                  // go low first and never hit these upper NaN values.
                  *pFill = std::numeric_limits<FloatEbmType>::quiet_NaN();
                  ++pFill;
               } while(LIKELY(pEndFill != pFill));
            }

            const FloatEbmType firstComparison = cutsLowerBoundInclusiveCopy[cPower / 2 - 1];
            do {
               const FloatEbmType val = *pValue;
               char * pResult = reinterpret_cast<char *>(cutsLowerBoundInclusiveCopy);

               pResult += UNPREDICTABLE(firstComparison <= val) ? size_t { cPower / 2 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 7 } * sizeof(FloatEbmType)) <= val) ? size_t { 8 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 3 } * sizeof(FloatEbmType)) <= val) ? size_t { 4 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 1 } * sizeof(FloatEbmType)) <= val) ? size_t { 2 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult) <= val) ? size_t { 1 } * sizeof(FloatEbmType) : size_t { 0 };

               const size_t result = 
                  (pResult - reinterpret_cast<char *>(cutsLowerBoundInclusiveCopy)) / sizeof(FloatEbmType);

               EBM_ASSERT(static_cast<IntEbmType>(result) == DiscretizeOne(val, countCuts, cutsLowerBoundInclusive));

               *pDiscretized = static_cast<IntEbmType>(result);
               ++pDiscretized;
               ++pValue;
            } while(LIKELY(pValueEnd != pValue));
            ret = Error_None;
            goto exit_with_log;
         }
      } else if(PREDICTABLE(countCuts <= IntEbmType { 62 })) {
         constexpr size_t cPower = 64;
         if(cPower * 4 <= cSamples) {
            static_assert(cPower - 1 <= sizeof(cutsLowerBoundInclusiveCopy) /
               sizeof(cutsLowerBoundInclusiveCopy[0]), "cutsLowerBoundInclusiveCopy buffer not large enough");

            memcpy(
               size_t { 1 } + cutsLowerBoundInclusiveCopy,
               cutsLowerBoundInclusive,
               sizeof(*cutsLowerBoundInclusive) * cCuts
            );

            if(LIKELY(cCuts != cPower - size_t { 2 })) {
               FloatEbmType * pFill = &cutsLowerBoundInclusiveCopy[cCuts + size_t { 1 }];
               const FloatEbmType * const pEndFill = &cutsLowerBoundInclusiveCopy[cPower - size_t { 1 }];
               do {
                  // NaN will always move us downwards into the region of valid cuts.  The first cut is always
                  // guaranteed to be non-NaN, so if we have a missing (NaN) value, then the binary search will
                  // go low first and never hit these upper NaN values.
                  *pFill = std::numeric_limits<FloatEbmType>::quiet_NaN();
                  ++pFill;
               } while(LIKELY(pEndFill != pFill));
            }

            const FloatEbmType firstComparison = cutsLowerBoundInclusiveCopy[cPower / 2 - 1];
            do {
               const FloatEbmType val = *pValue;
               char * pResult = reinterpret_cast<char *>(cutsLowerBoundInclusiveCopy);

               pResult += UNPREDICTABLE(firstComparison <= val) ? size_t { cPower / 2 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 15 } * sizeof(FloatEbmType)) <= val) ? size_t { 16 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 7 } * sizeof(FloatEbmType)) <= val) ? size_t { 8 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 3 } * sizeof(FloatEbmType)) <= val) ? size_t { 4 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 1 } * sizeof(FloatEbmType)) <= val) ? size_t { 2 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult) <= val) ? size_t { 1 } * sizeof(FloatEbmType) : size_t { 0 };

               const size_t result = 
                  (pResult - reinterpret_cast<char *>(cutsLowerBoundInclusiveCopy)) / sizeof(FloatEbmType);

               EBM_ASSERT(static_cast<IntEbmType>(result) == DiscretizeOne(val, countCuts, cutsLowerBoundInclusive));

               *pDiscretized = static_cast<IntEbmType>(result);
               ++pDiscretized;
               ++pValue;
            } while(LIKELY(pValueEnd != pValue));
            ret = Error_None;
            goto exit_with_log;
         }
      } else if(PREDICTABLE(countCuts <= IntEbmType { 126 })) {
         constexpr size_t cPower = 128;
         if(cPower * 4 <= cSamples) {
            static_assert(cPower - 1 <= sizeof(cutsLowerBoundInclusiveCopy) /
               sizeof(cutsLowerBoundInclusiveCopy[0]), "cutsLowerBoundInclusiveCopy buffer not large enough");

            memcpy(
               size_t { 1 } + cutsLowerBoundInclusiveCopy,
               cutsLowerBoundInclusive,
               sizeof(*cutsLowerBoundInclusive) * cCuts
            );

            if(LIKELY(cCuts != cPower - size_t { 2 })) {
               FloatEbmType * pFill = &cutsLowerBoundInclusiveCopy[cCuts + size_t { 1 }];
               const FloatEbmType * const pEndFill = &cutsLowerBoundInclusiveCopy[cPower - size_t { 1 }];
               do {
                  // NaN will always move us downwards into the region of valid cuts.  The first cut is always
                  // guaranteed to be non-NaN, so if we have a missing (NaN) value, then the binary search will
                  // go low first and never hit these upper NaN values.
                  *pFill = std::numeric_limits<FloatEbmType>::quiet_NaN();
                  ++pFill;
               } while(LIKELY(pEndFill != pFill));
            }

            const FloatEbmType firstComparison = cutsLowerBoundInclusiveCopy[cPower / 2 - 1];
            do {
               const FloatEbmType val = *pValue;
               char * pResult = reinterpret_cast<char *>(cutsLowerBoundInclusiveCopy);

               pResult += UNPREDICTABLE(firstComparison <= val) ? size_t { cPower / 2 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 31 } * sizeof(FloatEbmType)) <= val) ? size_t { 32 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 15 } * sizeof(FloatEbmType)) <= val) ? size_t { 16 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 7 } * sizeof(FloatEbmType)) <= val) ? size_t { 8 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 3 } * sizeof(FloatEbmType)) <= val) ? size_t { 4 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 1 } * sizeof(FloatEbmType)) <= val) ? size_t { 2 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult) <= val) ? size_t { 1 } * sizeof(FloatEbmType) : size_t { 0 };

               const size_t result = 
                  (pResult - reinterpret_cast<char *>(cutsLowerBoundInclusiveCopy)) / sizeof(FloatEbmType);

               EBM_ASSERT(static_cast<IntEbmType>(result) == DiscretizeOne(val, countCuts, cutsLowerBoundInclusive));

               *pDiscretized = static_cast<IntEbmType>(result);
               ++pDiscretized;
               ++pValue;
            } while(LIKELY(pValueEnd != pValue));
            ret = Error_None;
            goto exit_with_log;
         }
      } else if(PREDICTABLE(countCuts <= IntEbmType { 254 })) {
         constexpr size_t cPower = 256;
         if(cPower * 4 <= cSamples) {
            static_assert(cPower - 1 <= sizeof(cutsLowerBoundInclusiveCopy) /
               sizeof(cutsLowerBoundInclusiveCopy[0]), "cutsLowerBoundInclusiveCopy buffer not large enough");

            memcpy(
               size_t { 1 } + cutsLowerBoundInclusiveCopy,
               cutsLowerBoundInclusive,
               sizeof(*cutsLowerBoundInclusive) * cCuts
            );

            if(LIKELY(cCuts != cPower - size_t { 2 })) {
               FloatEbmType * pFill = &cutsLowerBoundInclusiveCopy[cCuts + size_t { 1 }];
               const FloatEbmType * const pEndFill = &cutsLowerBoundInclusiveCopy[cPower - size_t { 1 }];
               do {
                  // NaN will always move us downwards into the region of valid cuts.  The first cut is always
                  // guaranteed to be non-NaN, so if we have a missing (NaN) value, then the binary search will
                  // go low first and never hit these upper NaN values.
                  *pFill = std::numeric_limits<FloatEbmType>::quiet_NaN();
                  ++pFill;
               } while(LIKELY(pEndFill != pFill));
            }

            const FloatEbmType firstComparison = cutsLowerBoundInclusiveCopy[cPower / 2 - 1];
            do {
               const FloatEbmType val = *pValue;
               char * pResult = reinterpret_cast<char *>(cutsLowerBoundInclusiveCopy);

               pResult += UNPREDICTABLE(firstComparison <= val) ? size_t { cPower / 2 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 63 } * sizeof(FloatEbmType)) <= val) ? size_t { 64 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 31 } * sizeof(FloatEbmType)) <= val) ? size_t { 32 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 15 } * sizeof(FloatEbmType)) <= val) ? size_t { 16 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 7 } * sizeof(FloatEbmType)) <= val) ? size_t { 8 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 3 } * sizeof(FloatEbmType)) <= val) ? size_t { 4 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 1 } * sizeof(FloatEbmType)) <= val) ? size_t { 2 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult) <= val) ? size_t { 1 } * sizeof(FloatEbmType) : size_t { 0 };

               const size_t result = 
                  (pResult - reinterpret_cast<char *>(cutsLowerBoundInclusiveCopy)) / sizeof(FloatEbmType);

               EBM_ASSERT(static_cast<IntEbmType>(result) == DiscretizeOne(val, countCuts, cutsLowerBoundInclusive));

               *pDiscretized = static_cast<IntEbmType>(result);
               ++pDiscretized;
               ++pValue;
            } while(LIKELY(pValueEnd != pValue));
            ret = Error_None;
            goto exit_with_log;
         }
      } else if(PREDICTABLE(countCuts <= IntEbmType { 510 })) {
         constexpr size_t cPower = 512;
         if(cPower * 4 <= cSamples) {
            static_assert(cPower - 1 <= sizeof(cutsLowerBoundInclusiveCopy) /
               sizeof(cutsLowerBoundInclusiveCopy[0]), "cutsLowerBoundInclusiveCopy buffer not large enough");

            memcpy(
               size_t { 1 } + cutsLowerBoundInclusiveCopy,
               cutsLowerBoundInclusive,
               sizeof(*cutsLowerBoundInclusive) * cCuts
            );

            if(LIKELY(cCuts != cPower - size_t { 2 })) {
               FloatEbmType * pFill = &cutsLowerBoundInclusiveCopy[cCuts + size_t { 1 }];
               const FloatEbmType * const pEndFill = &cutsLowerBoundInclusiveCopy[cPower - size_t { 1 }];
               do {
                  // NaN will always move us downwards into the region of valid cuts.  The first cut is always
                  // guaranteed to be non-NaN, so if we have a missing (NaN) value, then the binary search will
                  // go low first and never hit these upper NaN values.
                  *pFill = std::numeric_limits<FloatEbmType>::quiet_NaN();
                  ++pFill;
               } while(LIKELY(pEndFill != pFill));
            }

            const FloatEbmType firstComparison = cutsLowerBoundInclusiveCopy[cPower / 2 - 1];
            do {
               const FloatEbmType val = *pValue;
               char * pResult = reinterpret_cast<char *>(cutsLowerBoundInclusiveCopy);

               pResult += UNPREDICTABLE(firstComparison <= val) ? size_t { cPower / 2 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 127 } * sizeof(FloatEbmType)) <= val) ? size_t { 128 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 63 } * sizeof(FloatEbmType)) <= val) ? size_t { 64 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 31 } * sizeof(FloatEbmType)) <= val) ? size_t { 32 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 15 } * sizeof(FloatEbmType)) <= val) ? size_t { 16 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 7 } * sizeof(FloatEbmType)) <= val) ? size_t { 8 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 3 } * sizeof(FloatEbmType)) <= val) ? size_t { 4 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 1 } * sizeof(FloatEbmType)) <= val) ? size_t { 2 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult) <= val) ? size_t { 1 } * sizeof(FloatEbmType) : size_t { 0 };

               const size_t result = 
                  (pResult - reinterpret_cast<char *>(cutsLowerBoundInclusiveCopy)) / sizeof(FloatEbmType);

               EBM_ASSERT(static_cast<IntEbmType>(result) == DiscretizeOne(val, countCuts, cutsLowerBoundInclusive));

               *pDiscretized = static_cast<IntEbmType>(result);
               ++pDiscretized;
               ++pValue;
            } while(LIKELY(pValueEnd != pValue));
            ret = Error_None;
            goto exit_with_log;
         }
      } else if(PREDICTABLE(countCuts <= IntEbmType { 1022 })) {
         constexpr size_t cPower = 1024;
         if(cPower * 4 <= cSamples) {
            static_assert(cPower - 1 == sizeof(cutsLowerBoundInclusiveCopy) /
               sizeof(cutsLowerBoundInclusiveCopy[0]), "cutsLowerBoundInclusiveCopy buffer not large enough");

            memcpy(
               size_t { 1 } + cutsLowerBoundInclusiveCopy,
               cutsLowerBoundInclusive,
               sizeof(*cutsLowerBoundInclusive) * cCuts
            );

            if(LIKELY(cCuts != cPower - size_t { 2 })) {
               FloatEbmType * pFill = &cutsLowerBoundInclusiveCopy[cCuts + size_t { 1 }];
               const FloatEbmType * const pEndFill = &cutsLowerBoundInclusiveCopy[cPower - size_t { 1 }];
               do {
                  // NaN will always move us downwards into the region of valid cuts.  The first cut is always
                  // guaranteed to be non-NaN, so if we have a missing (NaN) value, then the binary search will
                  // go low first and never hit these upper NaN values.
                  *pFill = std::numeric_limits<FloatEbmType>::quiet_NaN();
                  ++pFill;
               } while(LIKELY(pEndFill != pFill));
            }

            const FloatEbmType firstComparison = cutsLowerBoundInclusiveCopy[cPower / 2 - 1];
            do {
               const FloatEbmType val = *pValue;
               char * pResult = reinterpret_cast<char *>(cutsLowerBoundInclusiveCopy);

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

               const size_t result = 
                  (pResult - reinterpret_cast<char *>(cutsLowerBoundInclusiveCopy)) / sizeof(FloatEbmType);

               EBM_ASSERT(static_cast<IntEbmType>(result) == DiscretizeOne(val, countCuts, cutsLowerBoundInclusive));

               *pDiscretized = static_cast<IntEbmType>(result);
               ++pDiscretized;
               ++pValue;
            } while(LIKELY(pValueEnd != pValue));
            ret = Error_None;
            goto exit_with_log;
         }
      }

      if(UNLIKELY(IsConvertError<size_t>(countCuts))) {
         // this needs to point to real memory, otherwise it's invalid
         LOG_0(TraceLevelError, "ERROR Discretize countCuts was too large to fit into memory");
         ret = Error_IllegalParamValue; // the cutsLowerBoundInclusive wouldn't be possible
         goto exit_with_log;
      }

      if(IsMultiplyError(sizeof(*cutsLowerBoundInclusive), cCuts)) {
         LOG_0(TraceLevelError,
            "ERROR Discretize countCuts was too large to fit into cutsLowerBoundInclusive");
         ret = Error_IllegalParamValue; // the cutsLowerBoundInclusive array wouldn't be possible
         goto exit_with_log;
      }

      if(UNLIKELY(std::numeric_limits<IntEbmType>::max() - IntEbmType { 2 } < countCuts)) {
         // IntEbmType::max should be the maximum number of bins that we allow.  This prevents our caller from accidentally
         // overflowing when using either the count (eg: 255) or the index (eg: 0-254).  The 0th bin is the special missing
         // bin, so we have max - 1 normal bins.  We have 1 less cuts than we have bins (1 cut means 2 bins), so we can
         // have a maximum of max - 2 cuts.

         LOG_0(TraceLevelError,
            "ERROR Discretize countCuts was too large to allow for a missing value placeholder and -+inf bins");
         // this is a non-overflow somewhat arbitrary number for the upper level software to understand
         // so instead of returning illegal parameter, we should return out of memory and pretend that we
         // tried to allocate it since it doesn't seem worth creating a new error class for it
         ret = Error_OutOfMemory;
         goto exit_with_log;
      }

      if(UNLIKELY(std::numeric_limits<size_t>::max() == cCuts)) {
         // we add 1 to cCuts as our missing value, so this addition must succeed
         LOG_0(TraceLevelError,
            "ERROR Discretize countCuts was too large to allow for a missing value placeholder");

         // this is a non-overflow somewhat arbitrary number for the upper level software to understand
         // so instead of returning illegal parameter, we should return out of memory and pretend that we
         // tried to allocate it since it doesn't seem worth creating a new error class for it
         ret = Error_OutOfMemory;
         goto exit_with_log;
      }

      if(UNLIKELY(size_t { std::numeric_limits<ptrdiff_t>::max() } < cCuts)) {
         // the low value can increase until it's equal to cCuts, so cCuts must be expressable as a ptrdiff_t
         // we need to keep low as a ptrdiff_t since we compare it right after with high, which can be -1
         LOG_0(TraceLevelError,
            "ERROR Discretize countCuts was too large to allow for the binary search comparison");

         // this is a non-overflow somewhat arbitrary number for the upper level software to understand
         // so instead of returning illegal parameter, we should return out of memory and pretend that we
         // tried to allocate it since it doesn't seem worth creating a new error class for it
         ret = Error_OutOfMemory;
         goto exit_with_log;
      }

      if(UNLIKELY(std::numeric_limits<size_t>::max() / size_t { 2 } + size_t { 1 } < cCuts)) {
         // our first operation towards getting the mid-point is to add the size_t low and size_t high, and that can't 
         // overflow, so check that the maximum high added to the maximum low (which is the high) don't exceed that value
         LOG_0(TraceLevelError,
            "ERROR Discretize countCuts was too large to allow for the binary search add");

         // this is a non-overflow somewhat arbitrary number for the upper level software to understand
         // so instead of returning illegal parameter, we should return out of memory and pretend that we
         // tried to allocate it since it doesn't seem worth creating a new error class for it
         ret = Error_OutOfMemory;
         goto exit_with_log;
      }

      EBM_ASSERT(cCuts < std::numeric_limits<size_t>::max());
      EBM_ASSERT(size_t { 1 } <= cCuts);
      EBM_ASSERT(cCuts <= size_t { std::numeric_limits<ptrdiff_t>::max() });
      const ptrdiff_t highStart = static_cast<ptrdiff_t>(cCuts) - ptrdiff_t { 1 };

      // if we're going to runroll our first loop, then we need to ensure that there's a next loop after the first
      // unrolled loop, otherwise we would need to check if we were done before the first real loop iteration.
      // To ensure we have 2 original loop iterations, we need 1 cut in the center, 1 cut above, and 1 cut below, so 3
      EBM_ASSERT(size_t { 3 } <= cCuts);
      const size_t firstMiddle = static_cast<size_t>(highStart) >> 1;
      EBM_ASSERT(firstMiddle < cCuts);
      const FloatEbmType firstMidVal = cutsLowerBoundInclusive[firstMiddle];
      const ptrdiff_t firstMidLow = static_cast<ptrdiff_t>(firstMiddle) + ptrdiff_t { 1 };
      const ptrdiff_t firstMidHigh = static_cast<ptrdiff_t>(firstMiddle) - ptrdiff_t { 1 };

      do {
         const FloatEbmType val = *pValue;
         size_t middle = size_t { 0 };
         if(PREDICTABLE(!std::isnan(val))) {
            ptrdiff_t high = UNPREDICTABLE(firstMidVal <= val) ? highStart : firstMidHigh;
            ptrdiff_t low = UNPREDICTABLE(firstMidVal <= val) ? firstMidLow : ptrdiff_t { 0 };
            FloatEbmType midVal;
            do {
               EBM_ASSERT(ptrdiff_t { 0 } <= low && static_cast<size_t>(low) < cCuts);
               EBM_ASSERT(ptrdiff_t { 0 } <= high && static_cast<size_t>(high) < cCuts);
               EBM_ASSERT(low <= high);
               // low is equal or lower than high, so summing them can't exceed 2 * high, and after division it
               // can't be higher than high, so middle can't overflow ptrdiff_t after the division since high
               // is already a ptrdiff_t.  Generally the maximum positive value of a ptrdiff_t can be doubled 
               // when converted to a size_t, although that isn't guaranteed.  A more correct statement is that
               // the following must be false (which we check above):
               // "std::numeric_limits<size_t>::max() / 2 < cCuts - 1"
               EBM_ASSERT(!IsAddError(static_cast<size_t>(low), static_cast<size_t>(high)));
               middle = (static_cast<size_t>(low) + static_cast<size_t>(high)) >> 1;
               EBM_ASSERT(middle <= static_cast<size_t>(high));
               EBM_ASSERT(middle < cCuts);
               midVal = cutsLowerBoundInclusive[middle];
               EBM_ASSERT(middle < size_t { std::numeric_limits<ptrdiff_t>::max() });
               low = UNPREDICTABLE(midVal <= val) ? static_cast<ptrdiff_t>(middle) + ptrdiff_t { 1 } : low;
               EBM_ASSERT(ptrdiff_t { 0 } <= low && static_cast<size_t>(low) <= cCuts);
               high = UNPREDICTABLE(midVal <= val) ? high : static_cast<ptrdiff_t>(middle) - ptrdiff_t { 1 };
               EBM_ASSERT(ptrdiff_t { -1 } <= high && high <= highStart);

               // high can become -1 in some cases, so it needs to be ptrdiff_t.  It's tempting to try and change
               // this code and use the Hermann Bottenbruch version that checks for low != high in the loop comparison
               // since then we wouldn't have negative values and we could use size_t, but unfortunately that version
               // has a check at the end where we'd need to fetch cutsLowerBoundInclusive[low] after exiting the 
               // loop, so this version we have here is faster given that we only need to compare to a value that
               // we've already fetched from memory.  Also, this version makes slightly faster progress since
               // it does middle + 1 AND middle - 1 instead of just middle - 1, so it often eliminates one loop
               // iteration.  In practice this version will always work since no floating point type is less than 4
               // bytes, so we shouldn't have difficulty expressing any indexes with ptrdiff_t, and our indexes
               // for accessing memory are always size_t, so those should always work.
            } while(LIKELY(low <= high));
            EBM_ASSERT(size_t { 0 } <= middle && middle < cCuts);
            middle = UNPREDICTABLE(midVal <= val) ? middle + size_t { 2 } : middle + size_t { 1 };
            EBM_ASSERT(size_t { 1 } <= middle && middle <= size_t { 1 } + cCuts);
         }
         EBM_ASSERT(!IsConvertError<IntEbmType>(middle));
         const IntEbmType result = static_cast<IntEbmType>(middle);

         EBM_ASSERT(result == DiscretizeOne(val, countCuts, cutsLowerBoundInclusive));

         *pDiscretized = result;
         ++pDiscretized;
         ++pValue;
      } while(LIKELY(pValueEnd != pValue));
      ret = Error_None;
   }

exit_with_log:;

   LOG_COUNTED_N(
      &g_cLogExitDiscretizeParametersMessages,
      TraceLevelInfo,
      TraceLevelVerbose,
      "Exited Discretize: "
      "return=%" ErrorEbmTypePrintf
      ,
      ret
   );

   return ret;
}

} // DEFINED_ZONE_NAME
