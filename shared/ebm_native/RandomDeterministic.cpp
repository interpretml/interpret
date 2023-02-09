// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include "RandomDeterministic.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

// I generated these as purely random numbers from 0 to 2^64-1
static const uint_fast64_t k_oneTimePadSeed[64] {
   uint_fast64_t { 12108613678499979739U },
   uint_fast64_t { 8329435972752283296U },
   uint_fast64_t { 17571318292481228596U },
   uint_fast64_t { 8474109262332614363U },
   uint_fast64_t { 4596191631384569752U },
   uint_fast64_t { 5510915809971989025U },
   uint_fast64_t { 14720389497105379764U },
   uint_fast64_t { 14047171488222988787U },
   uint_fast64_t { 7736215019039428969U },
   uint_fast64_t { 8387470286819184488U }, // 10
   uint_fast64_t { 1272264876990534779U },
   uint_fast64_t { 6473674570481891879U },
   uint_fast64_t { 13907100008306791022U },
   uint_fast64_t { 12570840646885693808U },
   uint_fast64_t { 1295043964959849187U },
   uint_fast64_t { 6044489878752700404U },
   uint_fast64_t { 3043658277907488966U },
   uint_fast64_t { 14272464241729578605U },
   uint_fast64_t { 6450652935688055649U },
   uint_fast64_t { 1122646225207300450U }, // 20
   uint_fast64_t { 9680697536788401020U },
   uint_fast64_t { 14714112283214792619U },
   uint_fast64_t { 17000224091575576715U },
   uint_fast64_t { 14555454069625694159U },
   uint_fast64_t { 12133150780644733129U },
   uint_fast64_t { 15142044263353770020U },
   uint_fast64_t { 17374501799890097513U },
   uint_fast64_t { 587457683945871661U },
   uint_fast64_t { 9480109921896005794U },
   uint_fast64_t { 6202971064614006615U }, // 30
   uint_fast64_t { 8953539312749291378U },
   uint_fast64_t { 12924949407597356887U },
   uint_fast64_t { 2067650231428397037U },
   uint_fast64_t { 1104555401015663230U },
   uint_fast64_t { 6991116900072783160U },
   uint_fast64_t { 6876003810322139051U },
   uint_fast64_t { 14819303631007586897U },
   uint_fast64_t { 443649666753471969U },
   uint_fast64_t { 8852906418479390231U },
   uint_fast64_t { 16161542782915048273U }, // 40
   uint_fast64_t { 4167557640904791684U },
   uint_fast64_t { 13274255720658362279U },
   uint_fast64_t { 17654070117302736271U },
   uint_fast64_t { 2288656479984262408U },
   uint_fast64_t { 3955707939175675669U },
   uint_fast64_t { 966811535468564117U },
   uint_fast64_t { 10689941274756927828U },
   uint_fast64_t { 6900203119099125140U },
   uint_fast64_t { 3852394839434217481U },
   uint_fast64_t { 18083665370972184874U }, // 50
   uint_fast64_t { 17516541138771931787U },
   uint_fast64_t { 13183241652889971345U },
   uint_fast64_t { 13330691503705237225U },
   uint_fast64_t { 9615905893188178094U },
   uint_fast64_t { 1892274982045638252U },
   uint_fast64_t { 1429571804636752368U },
   uint_fast64_t { 8292521317717755949U },
   uint_fast64_t { 185343338715513721U },
   uint_fast64_t { 16175019103330891636U },
   uint_fast64_t { 8904867104718226249U }, // 60
   uint_fast64_t { 15891920948755861285U },
   uint_fast64_t { 2697603254172205724U },
   uint_fast64_t { 10333533257119705764U },
   uint_fast64_t { 8350484291935387907U } // 64
};

uint_fast64_t RandomDeterministic::GetOneTimePadConversion(uint_fast64_t seed) {
   static_assert(CountBitsRequiredPositiveMax<uint64_t>() == sizeof(k_oneTimePadSeed) / sizeof(k_oneTimePadSeed[0]),
      "the one time pad must have the same length as the number of bits");

   EBM_ASSERT(seed == static_cast<uint_fast64_t>(static_cast<uint64_t>(seed)));

   // this number generates a perfectly valid converted seed in a single pass if the user passes us a seed of zero
   uint_fast64_t result = uint_fast64_t { 0x6b79a38fd52c4e71 };
   const uint_fast64_t * pRandom = k_oneTimePadSeed;
   do {
      if(UNPREDICTABLE(0 != (uint_fast64_t { 1 } & seed))) {
         result ^= *pRandom;
      }
      ++pRandom;
      seed >>= 1;
   } while(LIKELY(0 != seed));
   return result;
}

void RandomDeterministic::Initialize(const uint64_t seed) {
   static constexpr uint64_t initializeSeed = uint64_t { 0xa75f138b4a162cfd };

   m_state1 = initializeSeed;
   m_state2 = initializeSeed;
   m_stateSeedConst = initializeSeed;

   uint_fast64_t originalRandomBits = GetOneTimePadConversion(static_cast<uint_fast64_t>(seed));
   EBM_ASSERT(originalRandomBits == static_cast<uint_fast64_t>(static_cast<uint64_t>(originalRandomBits)));

   uint_fast64_t randomBits = originalRandomBits;
   // the lowest bit of our result needs to be 1 to make our number odd (per the paper)
   uint_fast64_t sanitizedSeed = (uint_fast64_t { 0xF } & randomBits) | uint_fast64_t { 1 };
   randomBits >>= 4; // remove the bits that we used
   // disallow zeros for our hex digits by ORing 1
   const uint_fast16_t disallowMapFuture = (uint_fast16_t { 1 } << sanitizedSeed) | uint_fast16_t { 1 };

   // disallow zeros for our hex digits by initially setting to 1, which is our "hash" for the zero bit
   uint_fast16_t disallowMap = uint_fast16_t { 1 };
   uint_fast8_t bitShiftCur = uint_fast8_t { 60 };
   while(true) {
      // we ignore zeros, so use a do loop instead of while
      do {
         uint_fast64_t randomHexDigit = uint_fast64_t { 0xF } & randomBits;
         const uint_fast16_t indexBit = uint_fast16_t { 1 } << randomHexDigit;
         if(LIKELY(uint_fast16_t { 0 } == (indexBit & disallowMap))) {
            sanitizedSeed |= randomHexDigit << bitShiftCur;
            bitShiftCur -= uint_fast8_t { 4 };
            if(UNLIKELY(uint_fast8_t { 0 } == bitShiftCur)) {
               goto exit_loop;
            }
            disallowMap |= indexBit;
            if(UNLIKELY(UNLIKELY(uint_fast8_t { 28 } == bitShiftCur) ||
               UNLIKELY(uint_fast8_t { 24 } == bitShiftCur))) {
               // if bitShiftCur is 28 now then we just filled the low 4 bits for the high 32 bit number,
               // so for the upper 4 bits of the lower 32 bit number don't allow it to have the same
               // value as the lowest 4 bits of the upper 32 bits, and don't allow 0 and don't allow
               // the value at the bottom 4 bits
               //
               // if bitShiftCur is 28 then remove the disallowing of the lowest 4 bits of the upper 32 bit
               // number by only disallowing the previous number we just included (the uppre 4 bits of the lower
               // 32 bit value, and don't allow the lowest 4 bits, and don't allow 0.

               disallowMap = indexBit | disallowMapFuture;
            }
         }
         randomBits >>= 4;
      } while(LIKELY(uint_fast64_t { 0 } != randomBits));
      // ok, this is sort of a two time pad I guess, but we shouldn't ever use it more than twice in real life
      const uint_fast64_t top = static_cast<uint_fast64_t>(Rand32());
      const uint_fast64_t bottom = static_cast<uint_fast64_t>(Rand32());
      originalRandomBits = GetOneTimePadConversion(originalRandomBits ^ ((top << 32) | bottom));
      randomBits = originalRandomBits;
   }
exit_loop:;
   // is the lowest bit set as it should?
   EBM_ASSERT(uint_fast64_t { 1 } == sanitizedSeed % uint_fast64_t { 2 });

   const uint64_t finalSeed = static_cast<uint64_t>(sanitizedSeed);
   m_state1 = finalSeed;
   m_state2 = finalSeed;
   m_stateSeedConst = finalSeed;
}

} // DEFINED_ZONE_NAME
