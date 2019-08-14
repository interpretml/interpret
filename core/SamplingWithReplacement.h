// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef SAMPLING_WITH_REPLACEMENT_H
#define SAMPLING_WITH_REPLACEMENT_H

#include <stddef.h> // size_t, ptrdiff_t

#include "EbmInternal.h" // EBM_INLINE
#include "Logging.h" // EBM_ASSERT & LOG

class RandomStream;
class DataSetByFeatureCombination;

// TODO: if/when we decide we want to keep SamplingWithReplacement, we should create a SamplingMethod.h and SamplingMethod.cpp
class SamplingMethod {
public:
   const DataSetByFeatureCombination * const m_pOriginDataSet;

   EBM_INLINE SamplingMethod(const DataSetByFeatureCombination * const pOriginDataSet)
      : m_pOriginDataSet(pOriginDataSet) {
      EBM_ASSERT(nullptr != pOriginDataSet);
   }

   virtual ~SamplingMethod() {
   }

   virtual size_t GetTotalCountInstanceOccurrences() const = 0;
};

// SamplingWithReplacement this is the more theoretically correct method of sampling, but it has the drawback that we need to keep a count of the number of times each case is selected in the dataset.  Sampling without replacement would require 1 bit per case, so it can be faster.
class SamplingWithReplacement final : public SamplingMethod {
public:
   // TODO : make this a struct of FractionalType and size_t counts and use MACROS to have either size_t or FractionalType or both, and perf how this changes things.  We don't get a benefit anywhere by storing the raw data in both formats since it is never converted anyways, but this count is!
   const size_t * const m_aCountOccurrences;

   // we take owernship of the aCounts array.  We do not take ownership of the pOriginDataSet since many SamplingWithReplacement objects will refer to the original one
   EBM_INLINE SamplingWithReplacement(const DataSetByFeatureCombination * const pOriginDataSet, const size_t * const aCountOccurrences)
      : SamplingMethod(pOriginDataSet)
      , m_aCountOccurrences(aCountOccurrences) {
      EBM_ASSERT(nullptr != aCountOccurrences);
   }

   virtual ~SamplingWithReplacement() final override;
   virtual size_t GetTotalCountInstanceOccurrences() const final override;

   static SamplingWithReplacement * GenerateSingleSamplingSet(RandomStream * const pRandomStream, const DataSetByFeatureCombination * const pOriginDataSet);
   static SamplingWithReplacement * GenerateFlatSamplingSet(const DataSetByFeatureCombination * const pOriginDataSet);

   static void FreeSamplingSets(const size_t cSamplingSets, SamplingMethod ** apSamplingSets);
   static SamplingMethod ** GenerateSamplingSets(RandomStream * const pRandomStream, const DataSetByFeatureCombination * const pOriginDataSet, const size_t cSamplingSets);
};

#endif // SAMPLING_WITH_REPLACEMENT_H
