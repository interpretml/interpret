// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef SAMPLING_SET_H
#define SAMPLING_SET_H

#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "EbmInternal.h"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

class RandomStream;
class DataFrameBoosting;

class SamplingSet final {
   // Sampling with replacement is the more theoretically correct method of sampling, but it has the drawback that 
   // we need to keep a count of the number of times each sample is selected in the dataset.  
   // Sampling without replacement would require 1 bit per case, so it can be faster.

   const DataFrameBoosting * m_pOriginDataFrame;

   // TODO : make this a struct of FractionalType and size_t counts and use MACROS to have either size_t or 
   // FractionalType or both, and perf how this changes things.  We don't get a benefit anywhere by storing 
   // the raw data in both formats since it is never converted anyways, but this count is!
   size_t * m_aCountOccurrences;
   FloatEbmType * m_aWeights;
   FloatEbmType m_weightTotal;

   // we take owernship of the aCounts array.  We do not take ownership of the pOriginDataFrame since many 
   // SamplingSet objects will refer to the original one
   static SamplingSet * GenerateSingleSamplingSet(
      RandomStream * const pRandomStream, 
      const DataFrameBoosting * const pOriginDataFrame,
      const FloatEbmType * const aWeights
   );
   static SamplingSet * GenerateFlatSamplingSet(
      const DataFrameBoosting * const pOriginDataFrame,
      const FloatEbmType * const aWeights
   );
   void Free();
   void InitZero();

public:

   SamplingSet() = default; // preserve our POD status
   ~SamplingSet() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   size_t GetTotalCountSampleOccurrences() const {
      // for SamplingSet (bootstrap sampling), we have the same number of samples as our original dataset
      size_t cTotalCountSampleOccurrences = m_pOriginDataFrame->GetCountSamples();
#ifndef NDEBUG
      size_t cTotalCountSampleOccurrencesDebug = 0;
      for(size_t i = 0; i < m_pOriginDataFrame->GetCountSamples(); ++i) {
         cTotalCountSampleOccurrencesDebug += m_aCountOccurrences[i];
      }
      EBM_ASSERT(cTotalCountSampleOccurrencesDebug == cTotalCountSampleOccurrences);
#endif // NDEBUG
      return cTotalCountSampleOccurrences;
   }

   const DataFrameBoosting * GetDataFrameBoosting() const {
      return m_pOriginDataFrame;
   }

   const size_t * GetCountOccurrences() const {
      return m_aCountOccurrences;
   }
   const FloatEbmType * GetWeights() const {
      return m_aWeights;
   }
   FloatEbmType GetWeightTotal() const {
      return m_weightTotal;
   }

   static SamplingSet ** GenerateSamplingSets(
      RandomStream * const pRandomStream, 
      const DataFrameBoosting * const pOriginDataFrame, 
      const FloatEbmType * const aWeights,
      const size_t cSamplingSets
   );
   static void FreeSamplingSets(const size_t cSamplingSets, SamplingSet ** const apSamplingSets);
};
static_assert(std::is_standard_layout<SamplingSet>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<SamplingSet>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<SamplingSet>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

} // DEFINED_ZONE_NAME

#endif // SAMPLING_SET_H
