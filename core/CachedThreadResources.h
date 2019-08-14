// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef CACHED_THREAD_RESOURCES_H
#define CACHED_THREAD_RESOURCES_H

#include <queue>
#include <stdlib.h> // malloc, realloc, free
#include <stddef.h> // size_t, ptrdiff_t

#include "EbmInternal.h" // EBM_INLINE
#include "Logging.h" // EBM_ASSERT & LOG

template<bool bRegression>
class TreeNode;

template<bool bRegression>
class CompareTreeNodeSplittingGain final {
public:
   // TODO : check how efficient this is.  Is there a faster way to to this via a function
   EBM_INLINE bool operator() (const TreeNode<bRegression> * const & lhs, const TreeNode<bRegression> * const & rhs) const {
      return rhs->m_UNION.afterExaminationForPossibleSplitting.splitGain < lhs->m_UNION.afterExaminationForPossibleSplitting.splitGain;
   }
};

template<bool bRegression>
struct PredictionStatistics;

template<bool bRegression>
class CachedTrainingThreadResources {
   bool m_bError;

   const CompareTreeNodeSplittingGain<bRegression> m_compareTreeNodeSplitGain;

   // this allows us to share the memory between underlying data types
   void * m_aThreadByteBuffer1;
   size_t m_cThreadByteBufferCapacity1;

   void * m_aThreadByteBuffer2;
   size_t m_cThreadByteBufferCapacity2;

public:

   PredictionStatistics<bRegression> * const m_aSumPredictionStatistics;
   PredictionStatistics<bRegression> * const m_aSumPredictionStatistics1;
   PredictionStatistics<bRegression> * const m_aSumPredictionStatisticsBest;
   FractionalDataType * const m_aSumResidualErrors2;

   // THIS SHOULD ALWAYS BE THE LAST ITEM IN THIS STRUCTURE.  C++ guarantees that constructions initialize data members in the order that they are declared
   // since this class can potentially throw an exception in the constructor, we leave it last so that we are guaranteed that the rest of our object has been initialized
   std::priority_queue<TreeNode<bRegression> *, std::vector<TreeNode<bRegression> *>, CompareTreeNodeSplittingGain<bRegression>> m_bestTreeNodeToSplit;

   // in case you were wondering, this odd syntax of putting a try outside the function is called "Function try blocks" and it's the best way of handling exception in initialization
   CachedTrainingThreadResources(const size_t cVectorLength) try
      : m_bError(true)
      , m_compareTreeNodeSplitGain()
      , m_aThreadByteBuffer1(nullptr)
      , m_cThreadByteBufferCapacity1(0)
      , m_aThreadByteBuffer2(nullptr)
      , m_cThreadByteBufferCapacity2(0)
      , m_aSumPredictionStatistics(new (std::nothrow) PredictionStatistics<bRegression>[cVectorLength])
      , m_aSumPredictionStatistics1(new (std::nothrow) PredictionStatistics<bRegression>[cVectorLength])
      , m_aSumPredictionStatisticsBest(new (std::nothrow) PredictionStatistics<bRegression>[cVectorLength])
      , m_aSumResidualErrors2(new (std::nothrow) FractionalDataType[cVectorLength])
      // m_bestTreeNodeToSplit should be constructed last because we want everything above to be initialized before the constructor for m_bestTreeNodeToSplit is called since it could throw an exception and we don't want partial state in the rest of the member data.  
      // Construction initialization actually depends on order within the class, so this placement doesn't matter here.
      , m_bestTreeNodeToSplit(std::priority_queue<TreeNode<bRegression> *, std::vector<TreeNode<bRegression> *>, CompareTreeNodeSplittingGain<bRegression>>(m_compareTreeNodeSplitGain)) {

      // an unfortunate thing about function exception handling is that accessing non-static data from the catch block gives undefined behavior
      // so, we can't set m_bError to true if an error occurs, so instead we set it to true in the static initialization
      // C++ guarantees that initialization will occur in the order the variables are declared (not in the order of initialization)
      // but since we put m_bError above m_bestTreeNodeToSplit and since m_bestTreeNodeToSplit is the only thing that can throw an exception
      // if an exception occurs then our m_bError will be left as true
      m_bError = false;
   } catch(...) {
      // the only reason we should potentially find outselves here is if there was an exception thrown during construction of m_bestTreeNodeToSplit
      // C++ exceptions are suposed to be thrown by value and caught by reference, so it shouldn't be a pointer, and we shouldn't leak memory
      // according to the spec, it's undefined to access a non-static variable from a Function-try-block, so we can't access m_bError here  https://en.cppreference.com/w/cpp/language/function-try-block
      // so instead of setting it to true here, we set it to true by default and flip it to false if our caller gets to the constructor part
   }

   ~CachedTrainingThreadResources() {
      LOG(TraceLevelInfo, "Entered ~CachedTrainingThreadResources");

      free(m_aThreadByteBuffer1);
      free(m_aThreadByteBuffer2);
      delete[] m_aSumPredictionStatistics;
      delete[] m_aSumPredictionStatistics1;
      delete[] m_aSumPredictionStatisticsBest;
      delete[] m_aSumResidualErrors2;

      LOG(TraceLevelInfo, "Exited ~CachedTrainingThreadResources");
   }

   EBM_INLINE void * GetThreadByteBuffer1(const size_t cBytesRequired) {
      if(UNLIKELY(m_cThreadByteBufferCapacity1 < cBytesRequired)) {
         m_cThreadByteBufferCapacity1 = cBytesRequired << 1;
         LOG(TraceLevelInfo, "Growing CachedTrainingThreadResources::ThreadByteBuffer1 to %zu", m_cThreadByteBufferCapacity1);
         // TODO : use malloc here instead of realloc.  We don't need to copy the data, and if we free first then we can either slot the new memory in the old slot or it can be moved
         void * const aNewThreadByteBuffer = realloc(m_aThreadByteBuffer1, m_cThreadByteBufferCapacity1);
         if(UNLIKELY(nullptr == aNewThreadByteBuffer)) {
            // according to the realloc spec, if realloc fails to allocate the new memory, it returns nullptr BUT the old memory is valid.
            // we leave m_aThreadByteBuffer1 alone in this instance and will free that memory later in the destructor
            return nullptr;
         }
         m_aThreadByteBuffer1 = aNewThreadByteBuffer;
      }
      return m_aThreadByteBuffer1;
   }

   // TODO : we can probably avoid redoing any tree growing IF realloc doesn't move the memory since all the internal pointers would still be valid in that case
   EBM_INLINE bool GrowThreadByteBuffer2(const size_t cByteBoundaries) {
      // by adding cByteBoundaries and shifting our existing size, we do 2 things:
      //   1) we ensure that if we have zero size, we'll get some size that we'll get a non-zero size after the shift
      //   2) we'll always get back an odd number of items, which is good because we always have an odd number of TreeNodeChilden
      EBM_ASSERT(0 == m_cThreadByteBufferCapacity2 % cByteBoundaries);
      m_cThreadByteBufferCapacity2 = cByteBoundaries + (m_cThreadByteBufferCapacity2 << 1);
      LOG(TraceLevelInfo, "Growing CachedTrainingThreadResources::ThreadByteBuffer2 to %zu", m_cThreadByteBufferCapacity2);
      // TODO : can we use malloc here?  We only need realloc if we need to keep the existing data
      void * const aNewThreadByteBuffer = realloc(m_aThreadByteBuffer2, m_cThreadByteBufferCapacity2);
      if(UNLIKELY(nullptr == aNewThreadByteBuffer)) {
         // according to the realloc spec, if realloc fails to allocate the new memory, it returns nullptr BUT the old memory is valid.
         // we leave m_aThreadByteBuffer1 alone in this instance and will free that memory later in the destructor
         return true;
      }
      m_aThreadByteBuffer2 = aNewThreadByteBuffer;
      return false;
   }

   EBM_INLINE void * GetThreadByteBuffer2() {
      return m_aThreadByteBuffer2;
   }

   EBM_INLINE size_t GetThreadByteBuffer2Size() const {
      return m_cThreadByteBufferCapacity2;
   }

   EBM_INLINE bool IsError() const {
      return m_bError || nullptr == m_aSumPredictionStatistics || nullptr == m_aSumPredictionStatistics1 || nullptr == m_aSumPredictionStatisticsBest || nullptr == m_aSumResidualErrors2;
   }
};

class CachedInteractionThreadResources {
   // this allows us to share the memory between underlying data types
   void * m_aThreadByteBuffer1;
   size_t m_cThreadByteBufferCapacity1;

public:

   CachedInteractionThreadResources()
      : m_aThreadByteBuffer1(nullptr)
      , m_cThreadByteBufferCapacity1(0) {
   }

   ~CachedInteractionThreadResources() {
      LOG(TraceLevelInfo, "Entered ~CachedInteractionThreadResources");

      free(m_aThreadByteBuffer1);

      LOG(TraceLevelInfo, "Exited ~CachedInteractionThreadResources");
   }

   EBM_INLINE void * GetThreadByteBuffer1(const size_t cBytesRequired) {
      if(UNLIKELY(m_cThreadByteBufferCapacity1 < cBytesRequired)) {
         m_cThreadByteBufferCapacity1 = cBytesRequired << 1;
         LOG(TraceLevelInfo, "Growing CachedInteractionThreadResources::ThreadByteBuffer1 to %zu", m_cThreadByteBufferCapacity1);
         // TODO : use malloc here instead of realloc.  We don't need to copy the data, and if we free first then we can either slot the new memory in the old slot or it can be moved
         void * const aNewThreadByteBuffer = realloc(m_aThreadByteBuffer1, m_cThreadByteBufferCapacity1);
         if(UNLIKELY(nullptr == aNewThreadByteBuffer)) {
            // according to the realloc spec, if realloc fails to allocate the new memory, it returns nullptr BUT the old memory is valid.
            // we leave m_aThreadByteBuffer1 alone in this instance and will free that memory later in the destructor
            return nullptr;
         }
         m_aThreadByteBuffer1 = aNewThreadByteBuffer;
      }
      return m_aThreadByteBuffer1;
   }
};

#endif // CACHED_THREAD_RESOURCES_H
