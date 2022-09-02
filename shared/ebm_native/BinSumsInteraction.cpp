// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "ebm_internal.hpp"

#include "ebm_stats.hpp"

#include "Feature.hpp"
#include "Term.hpp"
#include "DataSetInteraction.hpp"

#include "InteractionCore.hpp"
#include "InteractionShell.hpp"

#include "GradientPair.hpp"
#include "Bin.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

template<ptrdiff_t cCompilerClasses, size_t cCompilerDimensions>
class BinSumsInteractionInternal final {
public:

   BinSumsInteractionInternal() = delete; // this is a static class.  Do not construct

   //  TODO: Notes on SIMD-ifying
   //
   //- We have to handle odd numbers of samples. Our fast SIMD code won't be able to handle odd numbers of samples, so
   //  we need to write something to handle this. We can either try to create dummy samples that we process that are no-ops
   //  or we can work to undo the operations afterwards.  We can't really generally create no-ops when we have arbitrary
   //  loss functions, and undoing might be hard again for arbitrary loss functions.  Also, if we need to write code that
   //  undoes work, it's probably easier to just write the special code that does odd numbers of samples and we handle that specially
   //- right now to handle odd numbers of samples, we put the bit back of the odd samples at the end.  For the code that
   //  handles these odd samples, it's probably better to put the dregs into the FIRST bitpack.  This is because when
   //  we initialize the loop we can initialize the first iteration to process 1-N items, and then reset at the bottom of the loop
   //  to do N items from then on.  Today we have a goto that re-inserts us into the loop, but that's kind of ugly.  We'll need to create
   //  crafted datasets anyways for SIMD, so we can put the dreg items right before a memory page boundary so that the first
   //  non-dreg loop iteration operates on a page/cache boundary.  Right now the goto labels that jump back into the loops are "one_last_loop"
   //- we can't use "End" pointers if we want the compiler to unwind the loops, so we need to use cItems and iItem instead.  That
   //  changes how our loops look.  I'm not sure if the compiler will also unwind these loops when we combine the loops
   //  with SIMD primitives, so we need to look at this.  If the compiler does not unwind, then we might need to put the
   //  body of the loop into a MACRO and then have multiple versions of the functions that call the MACRO N times.  :(
   //- Let's say we have 8/16 SIMD-streams.  We'll be adding gradients into tensors with these, but
   //  if two of the SIMD streams collide in their index, then we'd have a problem with adding. We therefore need to add into
   //  SEPARATE tensors (one per SIMD-stream) and then add the final tensors together at the end.
   //- Let's say we have a binary feature, so we're packing 64 bits into a single bit pack.  When we access the gradients array
   //  we don't want to advance by 64 * 4 * 16 = 4096 bytes per access.  What we can do is carefully locate the bit packs such
   //  that we access the non-bit packed data sequentially.  To do this we would locate the first item into the lowest bits of the
   //  first bitpack, and the second item into the lowest bits of the second bitpack.  If we have 16 SIMD streams then we'll
   //  load 16 bitpacks at once, and we'll load the next sequential 16 gradient floats at the same time.  Then we'll use the
   //  lowest bits of the first bitpack to index the first tensor, and the low bits of the second bitpack and the second float
   //  to update the second tensor.  This way we sequentially load both the bitpacks and the floats.
   //- FAST is harder since we have N dimensions.  We can have the compiler unwind the dimension loop, which is very very good since
   //  unpredictable branches there are costly.  The problem though is that each feature will be bitpacked with different cluster sizes,
   //  and those cluster sizes won't match up in terms of number of bits in the packs.
   //  We therefore need to have an if check that determines if we need to fetch bit-packs from all 16 parallel SIMD stream bitpacks. We can
   //  have a single check to determine when all the bitpacks need to be realoaded and we'll pay a branch misprediction when that
   //  happens.  The check will be inside the dimension loop, and since all SIMD streams will load at the same time and they share an if statement
   //  it won't ruin our SIMD-ability the way most unpredictable branches would
   //- We can do the same arragement for FAST as boosting where the gradients get loaded 16 at a time in a contiguous manner and the lowest
   //  bits of the first bit pack are the first element in the gradient array and the lowest bits of the second bit pack are the second element in the
   //  gradient array.
   //- Similar to Boosting, we'll need two different kinds of loops.  We'll have a SIMD-optimized one and CPU one that specializes in the "dregs"
   //  The "dregs" one will will be for situations where we won't want to load 8/16 in a SIMD-cluster.  We'll do them 1 at a time in a CPU loop
   //  Since we're going to be araning the bits in the bitpack by the size of the SIMD-cluster, we'll need to have a different layout for
   //  these last dregs
   //- An interesting thing about the FAST interaction function is that since the various dimensions have different pack sizes, we'll need to
   //  start them out at different fill points, or alternatively end them at different fill points.  This shouldn't be a problem though since
   //  we'll be re-loading each dimension separetly based on information stored on the stack in an array of variables that track this
   //  I'm guessing for FAST it's perhaps it's easier to have the first bitpacks be full, and the lsat ones to be partly empty.  Since we're
   //  re-loading each feature independently we'll just never re-load the last partly complete bit-pack
   //- for boosting, unlike FAST, we probably want to have our SIMD-optimized function handle a complete SIMD-pack of the bit-packs.  So, at worst
   //  if we had AVX-512 with 16 SIMD-streams, each of those would load a 64-bit packed item, and a complete SIMD-loop would process:
   //  16 * 64 = 1,024 items.  If there were 1,023 items, we'd want that to be processed by the CPU loop and probably do it non-SIMD.  In theory we
   //  could have SIMDed code that processed 63 items in the bit pack in a SIMD-fashion, but then we wouldn't be able to share that code
   //  with the same code that does loop unwinding into 64 unwound processing instructions.  It's probably more code efficient to just have either
   //  full SIMD with full bitpacks or CPU one at a time inside a loop
   //- So, for FAST, unlike boosting, we want our SIMD-able code to be aligned on 8/16 sample boundaries, so we'll process much less 
   //  "dregs" on the CPU compared to boosting where we could do up to 1,023 samples in the CPU code
   //- since we'll need to pack the bits differently depending on the type of SIMD.  We can proceed as follows:
   //  - move entirely towards using 32-bit floats AND build infrastructure to allow for adding together tensors AND being able to process
   //    separate datasets.  We'll need this to combine the separate SIMD tensors and to combine the CPU processed data from the SIMD processed data
   //  - build a separate SIMD-specialized part of the dataset, or a new dataset that packs bits in the way that we want for our particular SIMD-cluster size
   //  - keeping our existing code as-is, copy our exising code into a SIMD-only specialized set of loops in the compute part of the code and start
   //    passing clean sets of data that is in our new SIMD-specific datasets.  We'll use the existing code to handle CPU
   //  - allow the system to process all the data via CPU (which means it can be inside a single dataset) and compare this result to the result
   //    of using the SIMD code pipeline.  Maybe we can simulate all the same access 
   //- probably I will want to first copy my existing boosting and FAST code into a SIMD-space and get that working, then after that try to unroll
   //  the interior bit-packing loop since getting the compiler to unroll that might be SIMD-dependent.  After that, I can see if it's possible
   //  to re-merge the SIMD and nonSIMD code into a single function that maybe uses some templating to combine them (a bool bCpu compiler flag?)
   //  We shouldn't really care how fast the CPU code is, so if we start from the SIMD code and try to manipulate it to get non-SIMD code that's
   //  probably the right ordering.  We'll probably be using iBitBack and cBitsInPack instead of using "End" pointers for SIMD unrolled code
   //  so if that can work for CPU then great.

   INLINE_RELEASE_UNTEMPLATED static void Func(
      InteractionShell * const pInteractionShell, 
      const size_t cRuntimeRealDimensions,
      const size_t * const aiFeatures,
      const size_t * const acBins
   ) {
      constexpr bool bClassification = IsClassification(cCompilerClasses);

      LOG_0(Trace_Verbose, "Entered BinSumsInteractionInternal");

      BinBase * const aBinsBase = pInteractionShell->GetBinBaseFast();
      auto * const aBins = aBinsBase->Specialize<FloatFast, bClassification>();

      InteractionCore * const pInteractionCore = pInteractionShell->GetInteractionCore();
      const ptrdiff_t cRuntimeClasses = pInteractionCore->GetCountClasses();

      const ptrdiff_t cClasses = GET_COUNT_CLASSES(cCompilerClasses, cRuntimeClasses);
      const size_t cScores = GetCountScores(cClasses);
      EBM_ASSERT(!IsOverflowBinSize<FloatFast>(bClassification, cScores)); // we're accessing allocated memory
      const size_t cBytesPerBin = GetBinSize<FloatFast>(bClassification, cScores);

      const DataSetInteraction * const pDataSet = pInteractionCore->GetDataSetInteraction();
      const FloatFast * pGradientAndHessian = pDataSet->GetGradientsAndHessiansPointer();
      const FloatFast * const pGradientsAndHessiansEnd = pGradientAndHessian + (bClassification ? 2 : 1) * cScores * pDataSet->GetCountSamples();

      const FloatFast * pWeight = pDataSet->GetWeights();

      const size_t cRealDimensions = GET_DIMENSIONS(cCompilerDimensions, cRuntimeRealDimensions);
      EBM_ASSERT(1 <= cRealDimensions); // for interactions, we just return 0 for interactions with zero features

#ifndef NDEBUG
      FloatFast weightTotalDebug = 0;
#endif // NDEBUG

      struct DimensionalData {
         const StorageDataType * pData;
         size_t cBins;
      };

      // this is on the stack and the compiler should be able to optimize these as if they were variables or registers
      DimensionalData aDimensionalData[k_dynamicDimensions == cCompilerDimensions ? k_cDimensionsMax : cCompilerDimensions];
      size_t iDimensionInit = 0;
      do {
         DimensionalData * const pDimensionalData = &aDimensionalData[iDimensionInit];
         pDimensionalData->pData = pDataSet->GetInputDataPointer(aiFeatures[iDimensionInit]);
         pDimensionalData->cBins = acBins[iDimensionInit];
         ++iDimensionInit;
      } while(cRealDimensions != iDimensionInit);

      for(size_t iSample = 0; pGradientsAndHessiansEnd != pGradientAndHessian; ++iSample) {
         // this loop gets about twice as slow if you add a single unpredictable branching if statement based on count, even if you still access all the memory
         // in complete sequential order, so we'll probably want to use non-branching instructions for any solution like conditional selection or multiplication
         // this loop gets about 3 times slower if you use a bad pseudo random number generator like rand(), although it might be better if you inlined rand().
         // this loop gets about 10 times slower if you use a proper pseudo random number generator like std::default_random_engine
         // taking all the above together, it seems unlikley we'll use a method of separating sets via single pass randomized set splitting.  Even if count is 
         // stored in memory if shouldn't increase the time spent fetching it by 2 times, unless our bottleneck when threading is overwhelmingly memory pressure 
         // related, and even then we could store the count for a single bit aleviating the memory pressure greatly, if we use the right sampling method 

         // TODO : try using a sampling method with non-repeating samples, and put the count into a bit.  Then unwind that loop either at the byte level 
         //   (8 times) or the uint64_t level.  This can be done without branching and doesn't require random number generators

         // TODO : we can elminate the inner vector loop for regression at least, and also if we add a templated bool for binary class.  Propegate this change 
         //   to all places that we loop on the vector

         size_t cTensorBytes = cBytesPerBin;
         unsigned char * pRawBin = reinterpret_cast<unsigned char *>(aBins);
         size_t iDimension = 0;
         do {
            DimensionalData * const pDimensionalData = &aDimensionalData[iDimension];

            const StorageDataType * const pInputData = pDimensionalData->pData;
            const StorageDataType iBinOriginal = *pInputData;
            pDimensionalData->pData = pInputData + 1;
            EBM_ASSERT(!IsConvertError<size_t>(iBinOriginal));
            const size_t iBin = static_cast<size_t>(iBinOriginal);
            
            const size_t cBins = pDimensionalData->cBins;
            // interactions return interaction score of zero earlier on any useless dimensions
            // we strip dimensions from the tensors with 1 bin, so if 1 bin was accepted here, we'd need to strip
            // the bin too
            EBM_ASSERT(size_t { 2 } <= cBins);

            EBM_ASSERT(iBin < cBins);

            pRawBin += cTensorBytes * iBin;
            cTensorBytes *= cBins;
          
            ++iDimension;
         } while(cRealDimensions != iDimension);

         auto * const pBin = reinterpret_cast<Bin<FloatFast, bClassification> *>(pRawBin);
         ASSERT_BIN_OK(cBytesPerBin, pBin, pInteractionShell->GetBinsFastEndDebug());
         pBin->SetCountSamples(pBin->GetCountSamples() + 1);
         FloatFast weight = 1;
         if(nullptr != pWeight) {
            weight = *pWeight;
            ++pWeight;
#ifndef NDEBUG
            weightTotalDebug += weight;
#endif // NDEBUG
         }
         pBin->SetWeight(pBin->GetWeight() + weight);

         auto * const aGradientPair = pBin->GetGradientPairs();

         size_t iScore = 0;
         do {
            auto * const pGradientPair = &aGradientPair[iScore];
            const FloatFast gradient = *pGradientAndHessian;
            // gradient could be NaN
            // for classification, gradient can be anything from -1 to +1 (it cannot be infinity!)
            // for regression, gradient can be anything from +infinity or -infinity
            pGradientPair->m_sumGradients += gradient * weight;
            // m_sumGradients could be NaN, or anything from +infinity or -infinity in the case of regression
            if(bClassification) {
               EBM_ASSERT(
                  std::isnan(gradient) ||
                  !std::isinf(gradient) && 
                  -1 - k_epsilonGradient <= gradient && gradient <= 1
                  );

               const FloatFast hessian = *(pGradientAndHessian + 1);
               EBM_ASSERT(
                  std::isnan(hessian) ||
                  !std::isinf(hessian) && -k_epsilonGradient <= hessian && hessian <= FloatFast { 0.25 }
               ); // since any one hessian is limited to 0 <= hessian <= 0.25, the sum must be representable by a 64 bit number, 

               const FloatFast oldHessian = pGradientPair->GetSumHessians();
               // since any one hessian is limited to 0 <= gradient <= 0.25, the sum must be representable by a 64 bit number, 
               EBM_ASSERT(std::isnan(oldHessian) || !std::isinf(oldHessian) && -k_epsilonGradient <= oldHessian);
               const FloatFast newHessian = oldHessian + hessian * weight;
               // since any one hessian is limited to 0 <= hessian <= 0.25, the sum must be representable by a 64 bit number, 
               EBM_ASSERT(std::isnan(newHessian) || !std::isinf(newHessian) && -k_epsilonGradient <= newHessian);
               // which will always be representable by a float or double, so we can't overflow to inifinity or -infinity
               pGradientPair->SetSumHessians(newHessian);
            }
            pGradientAndHessian += bClassification ? 2 : 1;
            ++iScore;
         } while(cScores != iScore);
      }
      EBM_ASSERT(0 < pDataSet->GetWeightTotal());
      EBM_ASSERT(nullptr == pWeight || static_cast<FloatBig>(weightTotalDebug * 0.999) <= pDataSet->GetWeightTotal() && 
         pDataSet->GetWeightTotal() <= static_cast<FloatBig>(1.001 * weightTotalDebug));
      EBM_ASSERT(nullptr != pWeight || 
         static_cast<FloatBig>(pDataSet->GetCountSamples()) == pDataSet->GetWeightTotal());

      LOG_0(Trace_Verbose, "Exited BinSumsInteractionInternal");
   }
};

template<ptrdiff_t cCompilerClasses, size_t cCompilerDimensionsPossible>
class BinSumsInteractionDimensions final {
public:

   BinSumsInteractionDimensions() = delete; // this is a static class.  Do not construct

   INLINE_RELEASE_UNTEMPLATED static void Func(
      InteractionShell * const pInteractionShell, 
      const size_t cRealDimensions,
      const size_t * const aiFeatures,
      const size_t * const acBins
   ) {
      static_assert(1 <= cCompilerDimensionsPossible, "can't have less than 1 dimension for interactions");
      static_assert(cCompilerDimensionsPossible <= k_cDimensionsMax, "can't have more than the max dimensions");

      EBM_ASSERT(1 <= cRealDimensions);
      EBM_ASSERT(cRealDimensions <= k_cDimensionsMax);
      if(cCompilerDimensionsPossible == cRealDimensions) {
         BinSumsInteractionInternal<cCompilerClasses, cCompilerDimensionsPossible>::Func(pInteractionShell, cRealDimensions, aiFeatures, acBins);
      } else {
         BinSumsInteractionDimensions<cCompilerClasses, cCompilerDimensionsPossible + 1>::Func(pInteractionShell, cRealDimensions, aiFeatures, acBins);
      }
   }
};

template<ptrdiff_t cCompilerClasses>
class BinSumsInteractionDimensions<cCompilerClasses, k_cCompilerOptimizedCountDimensionsMax + 1> final {
public:

   BinSumsInteractionDimensions() = delete; // this is a static class.  Do not construct

   INLINE_RELEASE_UNTEMPLATED static void Func(
      InteractionShell * const pInteractionShell, 
      const size_t cRealDimensions,
      const size_t * const aiFeatures,
      const size_t * const acBins
   ) {
      EBM_ASSERT(1 <= cRealDimensions);
      EBM_ASSERT(cRealDimensions <= k_cDimensionsMax);
      BinSumsInteractionInternal<cCompilerClasses, k_dynamicDimensions>::Func(pInteractionShell, cRealDimensions, aiFeatures, acBins);
   }
};

template<ptrdiff_t cPossibleClasses>
class BinSumsInteractionTarget final {
public:

   BinSumsInteractionTarget() = delete; // this is a static class.  Do not construct

   INLINE_RELEASE_UNTEMPLATED static void Func(
      InteractionShell * const pInteractionShell, 
      const size_t cRealDimensions,
      const size_t * const aiFeatures,
      const size_t * const acBins
   ) {
      static_assert(IsClassification(cPossibleClasses), "cPossibleClasses needs to be a classification");
      static_assert(cPossibleClasses <= k_cCompilerClassesMax, "We can't have this many items in a data pack.");

      InteractionCore * const pInteractionCore = pInteractionShell->GetInteractionCore();
      const ptrdiff_t cRuntimeClasses = pInteractionCore->GetCountClasses();
      EBM_ASSERT(IsClassification(cRuntimeClasses));
      EBM_ASSERT(cRuntimeClasses <= k_cCompilerClassesMax);

      if(cPossibleClasses == cRuntimeClasses) {
         BinSumsInteractionDimensions<cPossibleClasses, 2>::Func(pInteractionShell, cRealDimensions, aiFeatures, acBins);
      } else {
         BinSumsInteractionTarget<cPossibleClasses + 1>::Func(pInteractionShell, cRealDimensions, aiFeatures, acBins);
      }
   }
};

template<>
class BinSumsInteractionTarget<k_cCompilerClassesMax + 1> final {
public:

   BinSumsInteractionTarget() = delete; // this is a static class.  Do not construct

   INLINE_RELEASE_UNTEMPLATED static void Func(
      InteractionShell * const pInteractionShell, 
      const size_t cRealDimensions,
      const size_t * const aiFeatures,
      const size_t * const acBins
   ) {
      static_assert(IsClassification(k_cCompilerClassesMax), "k_cCompilerClassesMax needs to be a classification");

      EBM_ASSERT(IsClassification(pInteractionShell->GetInteractionCore()->GetCountClasses()));
      EBM_ASSERT(k_cCompilerClassesMax < pInteractionShell->GetInteractionCore()->GetCountClasses());

      BinSumsInteractionDimensions<k_dynamicClassification, 2>::Func(pInteractionShell, cRealDimensions, aiFeatures, acBins);
   }
};

extern void BinSumsInteraction(
   InteractionShell * const pInteractionShell,
   const size_t cRealDimensions,
   const size_t * const aiFeatures,
   const size_t * const acBins
) {
   InteractionCore * const pInteractionCore = pInteractionShell->GetInteractionCore();
   const ptrdiff_t cRuntimeClasses = pInteractionCore->GetCountClasses();

   if(IsClassification(cRuntimeClasses)) {
      BinSumsInteractionTarget<2>::Func(pInteractionShell, cRealDimensions, aiFeatures, acBins);
   } else {
      EBM_ASSERT(IsRegression(cRuntimeClasses));
      BinSumsInteractionDimensions<k_regression, 2>::Func(pInteractionShell, cRealDimensions, aiFeatures, acBins);
   }
}

} // DEFINED_ZONE_NAME
