// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef SAMPLING_SET_H
#define SAMPLING_SET_H

#include <stddef.h> // size_t, ptrdiff_t

#include "EbmInternal.h" // EBM_INLINE
#include "Logging.h" // EBM_ASSERT & LOG

class RandomStream;
class DataSetByFeatureCombination;

class SamplingSet final {
   // Sampling with replacement is the more theoretically correct method of sampling, but it has the drawback that 
   // we need to keep a count of the number of times each instance is selected in the dataset.  
   // Sampling without replacement would require 1 bit per case, so it can be faster.
private:
   // we take owernship of the aCounts array.  We do not take ownership of the pOriginDataSet since many 
   // SamplingSet objects will refer to the original one
   EBM_INLINE SamplingSet(
      const DataSetByFeatureCombination * const pOriginDataSet, 
      const size_t * const aCountOccurrences
   )
      : m_pOriginDataSet(pOriginDataSet)
      , m_aCountOccurrences(aCountOccurrences) {
      EBM_ASSERT(nullptr != pOriginDataSet);
      EBM_ASSERT(nullptr != aCountOccurrences);
   }
   ~SamplingSet();

   static SamplingSet * GenerateSingleSamplingSet(
      RandomStream * const pRandomStream, 
      const DataSetByFeatureCombination * const pOriginDataSet
   );
   static SamplingSet * GenerateFlatSamplingSet(const DataSetByFeatureCombination * const pOriginDataSet);

public:
   const DataSetByFeatureCombination * const m_pOriginDataSet;

   // TODO : make this a struct of FractionalType and size_t counts and use MACROS to have either size_t or 
   // FractionalType or both, and perf how this changes things.  We don't get a benefit anywhere by storing 
   // the raw data in both formats since it is never converted anyways, but this count is!
   const size_t * const m_aCountOccurrences;

   size_t GetTotalCountInstanceOccurrences() const;

   static void FreeSamplingSets(const size_t cSamplingSets, SamplingSet ** const apSamplingSets);
   static SamplingSet ** GenerateSamplingSets(
      RandomStream * const pRandomStream, 
      const DataSetByFeatureCombination * const pOriginDataSet, 
      const size_t cSamplingSets
   );
};

#endif // SAMPLING_SET_H
