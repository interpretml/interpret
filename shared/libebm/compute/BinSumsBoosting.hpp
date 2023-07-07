// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef BIN_SUMS_BOOSTING_HPP
#define BIN_SUMS_BOOSTING_HPP

#include <stddef.h> // size_t, ptrdiff_t

#include "logging.h" // EBM_ASSERT
#include "zones.h"

#include "common_cpp.hpp" // Multiply
#include "bridge_cpp.hpp" // BinSumsBoostingBridge
#include "GradientPair.hpp"
#include "Bin.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

template<typename TFloat, bool bHessian, size_t cCompilerScores, bool bWeight, bool bReplication, ptrdiff_t cCompilerPack>
static void BinSumsBoostingInternal(BinSumsBoostingBridge * const pParams) {
   static_assert(bWeight || !bReplication, "bReplication cannot be true if bWeight is false");

   static constexpr bool bCompilerZeroDimensional = k_cItemsPerBitPackNone == cCompilerPack;
   static constexpr size_t cArrayScores = GetArrayScores(cCompilerScores);

#ifndef GPU_COMPILE
   EBM_ASSERT(nullptr != pParams);
   EBM_ASSERT(1 <= pParams->m_cSamples);
   EBM_ASSERT(0 == pParams->m_cSamples % TFloat::k_cSIMDPack);
   EBM_ASSERT(nullptr != pParams->m_aGradientsAndHessians);
   EBM_ASSERT(nullptr != pParams->m_aFastBins);
   EBM_ASSERT(k_dynamicScores == cCompilerScores || cCompilerScores == pParams->m_cScores);
#endif // GPU_COMPILE

   const size_t cScores = GET_COUNT_SCORES(cCompilerScores, pParams->m_cScores);

   auto * const aBins = reinterpret_cast<BinBase *>(pParams->m_aFastBins)->Specialize<typename TFloat::T, typename TFloat::TInt::T, bHessian, cArrayScores>();

   const size_t cSamples = pParams->m_cSamples;

   const typename TFloat::T * pGradientAndHessian = reinterpret_cast<const typename TFloat::T *>(pParams->m_aGradientsAndHessians);
   const typename TFloat::T * const pGradientsAndHessiansEnd = pGradientAndHessian + (bHessian ? 2 : 1) * cScores * cSamples;

   typename TFloat::TInt::T cBytesPerBin;
   int cBitsPerItemMax;
   int cShift;
   int cShiftReset;
   typename TFloat::TInt maskBits;
   const typename TFloat::TInt::T * pInputData;

   if(!bCompilerZeroDimensional) {
      cBytesPerBin = static_cast<typename TFloat::TInt::T>(GetBinSize<typename TFloat::T, typename TFloat::TInt::T>(bHessian, cScores));

      const ptrdiff_t cPack = GET_ITEMS_PER_BIT_PACK(cCompilerPack, pParams->m_cPack);
#ifndef GPU_COMPILE
      EBM_ASSERT(k_cItemsPerBitPackNone != cPack); // we require this condition to be templated
      EBM_ASSERT(1 <= cPack);
#endif // GPU_COMPILE

      const int cItemsPerBitPack = static_cast<int>(cPack);
#ifndef GPU_COMPILE
      EBM_ASSERT(1 <= cItemsPerBitPack);
      EBM_ASSERT(static_cast<size_t>(cItemsPerBitPack) <= CountBitsRequiredPositiveMax<typename TFloat::TInt::T>());
#endif // GPU_COMPILE

      cBitsPerItemMax = static_cast<int>(GetCountBits<typename TFloat::TInt::T>(static_cast<size_t>(cItemsPerBitPack)));
#ifndef GPU_COMPILE
      EBM_ASSERT(1 <= cBitsPerItemMax);
      EBM_ASSERT(static_cast<size_t>(cBitsPerItemMax) <= CountBitsRequiredPositiveMax<typename TFloat::TInt::T>());
#endif // GPU_COMPILE

      cShift = static_cast<int>((cSamples / TFloat::k_cSIMDPack - size_t { 1 }) % static_cast<size_t>(cItemsPerBitPack)) * cBitsPerItemMax;
      cShiftReset = (cItemsPerBitPack - 1) * cBitsPerItemMax;

      maskBits = MakeLowMask<typename TFloat::TInt::T>(cBitsPerItemMax);

      pInputData = reinterpret_cast<const typename TFloat::TInt::T *>(pParams->m_aPacked);
#ifndef GPU_COMPILE
      EBM_ASSERT(nullptr != pInputData);
#endif // GPU_COMPILE
   }

   const uint8_t * pCountOccurrences;
   if(bReplication) {
      pCountOccurrences = pParams->m_pCountOccurrences;
   }

   const typename TFloat::T * pWeight;
   if(bWeight) {
      pWeight = reinterpret_cast<const typename TFloat::T *>(pParams->m_aWeights);
#ifndef GPU_COMPILE
      EBM_ASSERT(nullptr != pWeight);
#endif // GPU_COMPILE
   }

   do {
      // this loop gets about twice as slow if you add a single unpredictable branching if statement based on count, even if you still access all the memory
      // in complete sequential order, so we'll probably want to use non-branching instructions for any solution like conditional selection or multiplication
      // this loop gets about 3 times slower if you use a bad pseudo random number generator like rand(), although it might be better if you inlined rand().
      // this loop gets about 10 times slower if you use a proper pseudo random number generator like std::default_random_engine
      // taking all the above together, it seems unlikley we'll use a method of separating sets via single pass randomized set splitting.  Even if count is 
      // stored in memory if shouldn't increase the time spent fetching it by 2 times, unless our bottleneck when threading is overwhelmingly memory pressure
      // related, and even then we could store the count for a single bit aleviating the memory pressure greatly, if we use the right sampling method 

      // TODO : try using a sampling method with non-repeating samples, and put the count into a bit.  Then unwind that loop either at the byte level 
      //   (8 times) or the uint64_t level.  This can be done without branching and doesn't require random number generators

      // we store the already multiplied dimensional value in *pInputData
      typename TFloat::TInt iTensorBinCombined;
      if(!bCompilerZeroDimensional) {
         iTensorBinCombined = TFloat::TInt::Load(pInputData);
         pInputData += TFloat::TInt::k_cSIMDPack;
      }
      while(true) {
         Bin<typename TFloat::T, typename TFloat::TInt::T, bHessian, cArrayScores> * apBins[TFloat::k_cSIMDPack];
         if(!bCompilerZeroDimensional) {
            typename TFloat::TInt iTensorBin = (iTensorBinCombined >> cShift) & maskBits;
            iTensorBin = Multiply<typename TFloat::TInt, typename TFloat::TInt::T, k_dynamicScores != cCompilerScores, static_cast<typename TFloat::TInt::T>(GetBinSize<typename TFloat::T, typename TFloat::TInt::T>(bHessian, cCompilerScores))>(iTensorBin, cBytesPerBin);
            ExecuteFunc(iTensorBin, [aBins, &apBins](int i, typename TFloat::TInt::T x) {
               apBins[i] = IndexBin(aBins, static_cast<size_t>(x));
            });
#ifndef NDEBUG
#ifndef GPU_COMPILE
            TFloat::EmptyExecuteFunc([cBytesPerBin, apBins, pParams](int i) {
               ASSERT_BIN_OK(cBytesPerBin, apBins[i], pParams->m_pDebugFastBinsEnd);
            });
#endif // GPU_COMPILE
#endif // NDEBUG
         }

         // TODO: the ultimate version of this algorithm would:
         //   1) Write to k_cSIMDPack histograms simutaneously to avoid collisions of indexes
         //   2) Sum up the final histograms using SIMD operations in parallel.  If we hvae k_cSIMDPack
         //      histograms, then we're prefectly suited to sum them, and integers and float32 values shouldn't
         //      have issues since we stay well away from 2^32 integers, and the float values don't have addition
         //      issues anymore (where you can't add a 1 to more than 16 million floats)
         //   But to do this, we need:
         //   1) scattered reads
         //   2) scattered writes
         //   3) possibly parallel integer multiplication (?), which is from a later version of SIMD
         //   4) the ability to index everything with uint32 indexes (for all histograms)
         //   5) the scattered reads and writes to not be too slow (they at least must fit into L3 cache?)
         //   We will need to rip apart the Bin class since we'll operate on multiple bins at a time. Maybe
         //   use offsetof to index float32/uint32 indexes inside the Bin classes in parallel.  (messy!)

         if(bReplication) {
            const typename TFloat::TInt cOccurences = TFloat::TInt::LoadBytes(pCountOccurrences);
            pCountOccurrences += TFloat::k_cSIMDPack;

            if(!bCompilerZeroDimensional) {
               ExecuteFunc(cOccurences, [apBins](int i, typename TFloat::TInt::T x) {
                  auto * pBin = apBins[i];
                  // TODO: In the future we'd like to eliminate this but we need the ability to change the Bin class
                  //       such that we can remove that field optionally
                  pBin->SetCountSamples(pBin->GetCountSamples() + x);
               });
            } else {
               ExecuteUnindexedFunc(cOccurences, [aBins](typename TFloat::TInt::T x) {
                  auto * pBin = aBins;
                  // TODO: In the future we'd like to eliminate this but we need the ability to change the Bin class
                  //       such that we can remove that field optionally
                  pBin->SetCountSamples(pBin->GetCountSamples() + x);
               });
            }
         } else {
            if(!bCompilerZeroDimensional) {
               TFloat::EmptyExecuteFunc([apBins](int i) {
                  auto * pBin = apBins[i];
                  // TODO: In the future we'd like to eliminate this but we need the ability to change the Bin class
                  //       such that we can remove that field optionally
                  pBin->SetCountSamples(pBin->GetCountSamples() + typename TFloat::TInt::T { 1 });
               });
            } else {
               TFloat::EmptyUnindexedExecuteFunc([aBins]() {
                  auto * pBin = aBins;
                  // TODO: In the future we'd like to eliminate this but we need the ability to change the Bin class
                  //       such that we can remove that field optionally
                  pBin->SetCountSamples(pBin->GetCountSamples() + typename TFloat::TInt::T { 1 });
               });
            }
         }

         TFloat weight;
         if(bWeight) {
            weight = TFloat::Load(pWeight);
            pWeight += TFloat::k_cSIMDPack;

            if(!bCompilerZeroDimensional) {
               ExecuteFunc(weight, [apBins](int i, typename TFloat::T x) {
                  auto * pBin = apBins[i];
                  // TODO: In the future we'd like to eliminate this but we need the ability to change the Bin class
                  //       such that we can remove that field optionally
                  pBin->SetWeight(pBin->GetWeight() + x);
               });
            } else {
               ExecuteUnindexedFunc(weight, [aBins](typename TFloat::T x) {
                  auto * pBin = aBins;
                  // TODO: In the future we'd like to eliminate this but we need the ability to change the Bin class
                  //       such that we can remove that field optionally
                  pBin->SetWeight(pBin->GetWeight() + x);
               });
            }
         } else {
            if(!bCompilerZeroDimensional) {
               TFloat::EmptyExecuteFunc([apBins](int i) {
                  auto * pBin = apBins[i];
                  // TODO: In the future we'd like to eliminate this but we need the ability to change the Bin class
                  //       such that we can remove that field optionally
                  pBin->SetWeight(pBin->GetWeight() + typename TFloat::T { 1.0 });
               });
            } else {
               TFloat::EmptyUnindexedExecuteFunc([aBins]() {
                  auto * pBin = aBins;
                  // TODO: In the future we'd like to eliminate this but we need the ability to change the Bin class
                  //       such that we can remove that field optionally
                  pBin->SetWeight(pBin->GetWeight() + typename TFloat::T { 1.0 });
               });
            }
         }

         // TODO: we probably want a templated version of this function for Bins with only 1 cScore so that
         //       we don't have a loop here, which will mean that the cCompilerPack will be the only loop which
         //       will allow the compiler to unroll that loop (since it only unrolls one level of loops)

         size_t iScore = 0;
         do {
            TFloat gradient = TFloat::Load(bHessian ? &pGradientAndHessian[iScore << (TFloat::k_cSIMDShift + 1)] : &pGradientAndHessian[iScore << TFloat::k_cSIMDShift]);
            if(bWeight) {
               gradient *= weight;
            }
            if(!bCompilerZeroDimensional) {
               ExecuteFunc(gradient, [apBins, iScore](int i, typename TFloat::T x) {
                  auto * pBin = apBins[i];
                  auto * const aGradientPair = pBin->GetGradientPairs();
                  auto * const pGradientPair = &aGradientPair[iScore];
                  pGradientPair->m_sumGradients += x;
               });
            } else {
               ExecuteUnindexedFunc(gradient, [aBins, iScore](typename TFloat::T x) {
                  auto * pBin = aBins;
                  auto * const aGradientPair = pBin->GetGradientPairs();
                  auto * const pGradientPair = &aGradientPair[iScore];
                  pGradientPair->m_sumGradients += x;
               });
            }
            if(bHessian) {
               TFloat hessian = TFloat::Load(&pGradientAndHessian[(iScore << (TFloat::k_cSIMDShift + 1)) + TFloat::k_cSIMDPack]);
               if(bWeight) {
                  hessian *= weight;
               }

               if(!bCompilerZeroDimensional) {
                  ExecuteFunc(hessian, [apBins, iScore](int i, typename TFloat::T x) {
                     auto * pBin = apBins[i];
                     auto * const aGradientPair = pBin->GetGradientPairs();
                     auto * const pGradientPair = &aGradientPair[iScore];
                     pGradientPair->SetHess(pGradientPair->GetHess() + x);
                  });
               } else {
                  ExecuteUnindexedFunc(hessian, [aBins, iScore](typename TFloat::T x) {
                     auto * pBin = aBins;
                     auto * const aGradientPair = pBin->GetGradientPairs();
                     auto * const pGradientPair = &aGradientPair[iScore];
                     pGradientPair->SetHess(pGradientPair->GetHess() + x);
                  });
               }
            }
            ++iScore;
         } while(cScores != iScore);

         pGradientAndHessian += bHessian ? (cScores << (TFloat::k_cSIMDShift + 1)) : (cScores << TFloat::k_cSIMDShift);

         if(bCompilerZeroDimensional) {
            if(pGradientsAndHessiansEnd == pGradientAndHessian) {
               break;
            }
         } else {
            cShift -= cBitsPerItemMax;
            if(cShift < 0) {
               break;
            }
         }
      }
      if(bCompilerZeroDimensional) {
         break;
      }
      cShift = cShiftReset;
   } while(pGradientsAndHessiansEnd != pGradientAndHessian);
}


template<typename TFloat, bool bHessian, size_t cCompilerScores, bool bWeight, bool bReplication, ptrdiff_t cCompilerPack>
INLINE_RELEASE_TEMPLATED ErrorEbm OperatorBinSumsBoosting(BinSumsBoostingBridge * const pParams) {
   // TODO: in the future call back to the the operator class to allow it to inject the code into a GPU (see Objective.hpp for an example):
   // return TFloat::template OperatorBinSumsBoosting<TFloat, bHessian, cCompilerScores, bWeight, bReplication, cCompilerPack>(pParams);
   // and also return the error code returned from that call instead of always Error_None
   BinSumsBoostingInternal<TFloat, bHessian, cCompilerScores, bWeight, bReplication, cCompilerPack>(pParams);

   return Error_None;
}


template<typename TFloat, bool bHessian, size_t cCompilerScores, bool bWeight, bool bReplication>
INLINE_RELEASE_TEMPLATED static ErrorEbm BitPackBoosting(BinSumsBoostingBridge * const pParams) {
   if(k_cItemsPerBitPackNone != pParams->m_cPack) {
      return OperatorBinSumsBoosting<TFloat, bHessian, cCompilerScores, bWeight, bReplication, k_cItemsPerBitPackDynamic>(pParams);
   } else {
      // this needs to be special cased because otherwise we would inject comparisons into the dynamic version
      return OperatorBinSumsBoosting<TFloat, bHessian, cCompilerScores, bWeight, bReplication, k_cItemsPerBitPackNone>(pParams);
   }
}


template<typename TFloat, bool bHessian, size_t cCompilerScores>
INLINE_RELEASE_TEMPLATED static ErrorEbm FinalOptionsBoosting(BinSumsBoostingBridge * const pParams) {
   if(nullptr != pParams->m_aWeights) {
      static constexpr bool bWeight = true;

      if(nullptr != pParams->m_pCountOccurrences) {
         static constexpr bool bReplication = true;
         return BitPackBoosting<TFloat, bHessian, cCompilerScores, bWeight, bReplication>(pParams);
      } else {
         static constexpr bool bReplication = false;
         return BitPackBoosting<TFloat, bHessian, cCompilerScores, bWeight, bReplication>(pParams);
      }
   } else {
      static constexpr bool bWeight = false;

      // we use the weights to hold both the weights and the inner bag counts if there are inner bags
      EBM_ASSERT(nullptr == pParams->m_pCountOccurrences);
      static constexpr bool bReplication = false;

      return BitPackBoosting<TFloat, bHessian, cCompilerScores, bWeight, bReplication>(pParams);
   }
}


template<typename TFloat, bool bHessian, size_t cPossibleScores>
struct CountClassesBoosting final {
   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(BinSumsBoostingBridge * const pParams) {
      if(cPossibleScores == pParams->m_cScores) {
         return FinalOptionsBoosting<TFloat, bHessian, cPossibleScores>(pParams);
      } else {
         return CountClassesBoosting<TFloat, bHessian, cPossibleScores + 1>::Func(pParams);
      }
   }
};
template<typename TFloat, bool bHessian>
struct CountClassesBoosting<TFloat, bHessian, k_cCompilerScoresMax + 1> final {
   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(BinSumsBoostingBridge * const pParams) {
      return FinalOptionsBoosting<TFloat, bHessian, k_dynamicScores>(pParams);
   }
};

template<typename TFloat>
INLINE_RELEASE_TEMPLATED static ErrorEbm BinSumsBoosting(BinSumsBoostingBridge * const pParams) {
   LOG_0(Trace_Verbose, "Entered BinSumsBoosting");

   // all our memory should be aligned. It is required by SIMD for correctness or performance
   EBM_ASSERT(IsAligned(pParams->m_aGradientsAndHessians));
   EBM_ASSERT(IsAligned(pParams->m_aWeights));
   EBM_ASSERT(IsAligned(pParams->m_pCountOccurrences));
   EBM_ASSERT(IsAligned(pParams->m_aPacked));
   EBM_ASSERT(IsAligned(pParams->m_aFastBins));

   ErrorEbm error;

   EBM_ASSERT(1 <= pParams->m_cScores);
   if(EBM_FALSE != pParams->m_bHessian) {
      if(size_t { 1 } != pParams->m_cScores) {
         // muticlass
         error = CountClassesBoosting<TFloat, true, k_cCompilerScoresStart>::Func(pParams);
      } else {
         error = FinalOptionsBoosting<TFloat, true, k_oneScore>(pParams);
      }
   } else {
      if(size_t { 1 } != pParams->m_cScores) {
         // Odd: gradient multiclass. Allow it, but do not optimize for it
         error = FinalOptionsBoosting<TFloat, false, k_dynamicScores>(pParams);
      } else {
         error = FinalOptionsBoosting<TFloat, false, k_oneScore>(pParams);
      }
   }

   LOG_0(Trace_Verbose, "Exited BinSumsBoosting");

   return error;
}

} // DEFINED_ZONE_NAME

#endif // BIN_SUMS_BOOSTING_HPP