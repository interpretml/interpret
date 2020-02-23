// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef CACHED_THREAD_RESOURCES_H
#define CACHED_THREAD_RESOURCES_H

#include <vector>
#include <queue>
#include <stdlib.h> // malloc, realloc, free
#include <stddef.h> // size_t, ptrdiff_t

#include "EbmInternal.h" // EBM_INLINE
#include "Logging.h" // EBM_ASSERT & LOG

#include "TreeNode.h"

template<bool bClassification>
class CompareTreeNodeSplittingGain final {
public:
   // TODO : check how efficient this is.  Is there a faster way to to this
   EBM_INLINE constexpr bool operator() (const TreeNode<bClassification> * const & lhs, const TreeNode<bClassification> * const & rhs) const {
      return lhs->m_UNION.m_afterExaminationForPossibleSplitting.m_splitGain <= rhs->m_UNION.m_afterExaminationForPossibleSplitting.m_splitGain;
   }
};

template<bool bClassification>
class SafeTreeNodeQueue final {
   // make it zero the error just in case someone introduces an initialization bug such that this doesn't set set.  The default will be an error then
   bool m_bSuccess;

public:
   // THIS SHOULD ALWAYS BE THE LAST ITEM IN THIS STRUCTURE.  C++ guarantees that constructions initialize data members in the order that they are declared
   // since this class can potentially throw an exception in the constructor, we leave it last so that we are guaranteed that the rest of our object 
   // has been initialized
   std::priority_queue<TreeNode<bClassification> *, std::vector<TreeNode<bClassification> *>, CompareTreeNodeSplittingGain<bClassification>> m_queue;

   // in case you were wondering, this odd syntax of putting a try outside the function is called "Function try blocks" and it's the best way of 
   // handling exception in initialization
   SafeTreeNodeQueue() try
      : m_bSuccess(false)
      , m_queue() {
      // an unfortunate thing about function exception handling is that accessing non-static data from the catch block gives undefined behavior
      // so, we can't set m_bSuccess to false if an error occurs, so instead we set it to false in the static initialization
      // C++ guarantees that initialization will occur in the order the variables are declared (not in the order of initialization)
      // but since we put m_bSuccess at the top, if an exception occurs then our m_bSuccess will be left as false since it won't call the 
      // initializer which sets it to true
      // https://en.cppreference.com/w/cpp/language/function-try-block
      m_bSuccess = true;
   } catch(...) {
      // the only reason we should potentially find outselves here is if there was an exception thrown during construction
      // C++ exceptions are suposed to be thrown by value and caught by reference, so it shouldn't be a pointer, and we shouldn't leak memory
   }

   EBM_INLINE bool IsSuccess() const {
      return m_bSuccess;
   }
};

template<bool bClassification>
struct HistogramBucketVectorEntry;

template<bool bClassification>
class CachedBoostingThreadResources {
   // TODO: can I preallocate m_aThreadByteBuffer1 and m_aThreadByteBuffer2 without resorting to grow them if I examine my inputs

   // this allows us to share the memory between underlying data types
   void * m_aThreadByteBuffer1;
   size_t m_cThreadByteBufferCapacity1;

   void * m_aThreadByteBuffer2;
   size_t m_cThreadByteBufferCapacity2;

public:

   HistogramBucketVectorEntry<bClassification> * const m_aSumHistogramBucketVectorEntry;
   HistogramBucketVectorEntry<bClassification> * const m_aSumHistogramBucketVectorEntry1;
   FloatEbmType * const m_aTempFloatVector;

   void * m_aEquivalentSplits; // we use different structures for mains and multidimension and between classification and regression

   SafeTreeNodeQueue<bClassification> m_bestTreeNodeToSplit;

   CachedBoostingThreadResources(const size_t cVectorLength)
      : m_aThreadByteBuffer1(nullptr)
      , m_cThreadByteBufferCapacity1(0)
      , m_aThreadByteBuffer2(nullptr)
      , m_cThreadByteBufferCapacity2(0)
      , m_aSumHistogramBucketVectorEntry(new (std::nothrow) HistogramBucketVectorEntry<bClassification>[cVectorLength])
      , m_aSumHistogramBucketVectorEntry1(new (std::nothrow) HistogramBucketVectorEntry<bClassification>[cVectorLength])
      , m_aTempFloatVector(new (std::nothrow) FloatEbmType[cVectorLength])
      , m_aEquivalentSplits(nullptr)
      , m_bestTreeNodeToSplit() {
      EBM_ASSERT(0 < cVectorLength);
   }

   ~CachedBoostingThreadResources() {
      LOG_0(TraceLevelInfo, "Entered ~CachedBoostingThreadResources");

      free(m_aThreadByteBuffer1);
      free(m_aThreadByteBuffer2);
      delete[] m_aSumHistogramBucketVectorEntry;
      delete[] m_aSumHistogramBucketVectorEntry1;
      delete[] m_aTempFloatVector;
      free(m_aEquivalentSplits);

      LOG_0(TraceLevelInfo, "Exited ~CachedBoostingThreadResources");
   }

   EBM_INLINE void * GetThreadByteBuffer1(const size_t cBytesRequired) {
      if(UNLIKELY(m_cThreadByteBufferCapacity1 < cBytesRequired)) {
         m_cThreadByteBufferCapacity1 = cBytesRequired << 1;
         LOG_N(TraceLevelInfo, "Growing CachedBoostingThreadResources::ThreadByteBuffer1 to %zu", m_cThreadByteBufferCapacity1);
         // TODO : use malloc here instead of realloc.  We don't need to copy the data, and if we free first then we can either slot the new memory 
         // in the old slot or it can be moved
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

   EBM_INLINE bool GrowThreadByteBuffer2(const size_t cByteBoundaries) {
      // by adding cByteBoundaries and shifting our existing size, we do 2 things:
      //   1) we ensure that if we have zero size, we'll get some size that we'll get a non-zero size after the shift
      //   2) we'll always get back an odd number of items, which is good because we always have an odd number of TreeNodeChilden
      EBM_ASSERT(0 == m_cThreadByteBufferCapacity2 % cByteBoundaries);
      m_cThreadByteBufferCapacity2 = cByteBoundaries + (m_cThreadByteBufferCapacity2 << 1);
      LOG_N(TraceLevelInfo, "Growing CachedBoostingThreadResources::ThreadByteBuffer2 to %zu", m_cThreadByteBufferCapacity2);
      // TODO : use malloc here.  our tree objects have internal pointers, so we're going to dispose of our work anyways
      // There is no way to check if the array was re-allocated or not without invoking undefined behavior, 
      // so we don't get a benefit if the array can be resized with realloc
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
      return !m_bestTreeNodeToSplit.IsSuccess() || nullptr == m_aSumHistogramBucketVectorEntry || 
         nullptr == m_aSumHistogramBucketVectorEntry1 || nullptr == m_aTempFloatVector;
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
      LOG_0(TraceLevelInfo, "Entered ~CachedInteractionThreadResources");

      free(m_aThreadByteBuffer1);

      LOG_0(TraceLevelInfo, "Exited ~CachedInteractionThreadResources");
   }

   EBM_INLINE void * GetThreadByteBuffer1(const size_t cBytesRequired) {
      if(UNLIKELY(m_cThreadByteBufferCapacity1 < cBytesRequired)) {
         m_cThreadByteBufferCapacity1 = cBytesRequired << 1;
         LOG_N(TraceLevelInfo, "Growing CachedInteractionThreadResources::ThreadByteBuffer1 to %zu", m_cThreadByteBufferCapacity1);
         // TODO : use malloc here instead of realloc.  We don't need to copy the data, and if we free first then we can either slot the new 
         // memory in the old slot or it can be moved
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
