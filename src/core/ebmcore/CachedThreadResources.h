// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef CACHED_THREAD_RESOURCES_H
#define CACHED_THREAD_RESOURCES_H

#include <queue>
#include <stdlib.h> // malloc, realloc, free
#include <stddef.h> // size_t, ptrdiff_t

#include "EbmInternal.h" // TML_INLINE

template<bool bRegression>
class TreeNode;

template<bool bRegression>
class CompareTreeNodeSplittingScore final {
public:
   // TODO : check how efficient this is.  Is there a faster way to to this via a function
   TML_INLINE bool operator() (const TreeNode<bRegression> * const & lhs, const TreeNode<bRegression> * const & rhs) const {
      return rhs->m_UNION.afterSplit.nodeSplittingScore < lhs->m_UNION.afterSplit.nodeSplittingScore;
   }
};

template<bool bRegression>
class PredictionStatistics;

template<bool bRegression>
class CachedTrainingThreadResources {
   const CompareTreeNodeSplittingScore<bRegression> m_compareTreeNode;

   // this allows us to share the memory between underlying data types
   void * m_aThreadByteBuffer1;
   size_t m_cThreadByteBufferCapacity1;

   void * m_aThreadByteBuffer2;
   size_t m_cThreadByteBufferCapacity2;

   bool m_bError;

public:

   PredictionStatistics<bRegression> * const m_aSumPredictionStatistics;
   PredictionStatistics<bRegression> * const m_aSumPredictionStatistics1;
   PredictionStatistics<bRegression> * const m_aSumPredictionStatisticsBest;
   FractionalDataType * const m_aSumResidualErrors2;

   // THIS SHOULD ALWAYS BE THE LAST ITEM IN THIS STRUCTURE.  C++ guarantees that constructions initialize data members in the order that they are declared
   // since this class can potentially throw an exception in the constructor, we leave it last so that we are guaranteed that the rest of our object has been initialized
   std::priority_queue<TreeNode<bRegression> *, std::vector<TreeNode<bRegression> *>, CompareTreeNodeSplittingScore<bRegression>> m_bestTreeNodeToSplit;

   // in case you were wondering, this odd syntax of putting a try outside the function is called "Function try blocks" and it's the best way of handling exception in initialization
   CachedTrainingThreadResources(const size_t cVectorLength) try
      : m_compareTreeNode()
      , m_aThreadByteBuffer1(nullptr)
      , m_cThreadByteBufferCapacity1(0)
      , m_aThreadByteBuffer2(nullptr)
      , m_cThreadByteBufferCapacity2(0)
      , m_bError(true)
      , m_aSumPredictionStatistics(new (std::nothrow) PredictionStatistics<bRegression>[cVectorLength])
      , m_aSumPredictionStatistics1(new (std::nothrow) PredictionStatistics<bRegression>[cVectorLength])
      , m_aSumPredictionStatisticsBest(new (std::nothrow) PredictionStatistics<bRegression>[cVectorLength])
      , m_aSumResidualErrors2(new (std::nothrow) FractionalDataType[cVectorLength])
      // m_bestTreeNodeToSplit should be constructed last because we want everything above to be initialized before the constructor for m_bestTreeNodeToSplit is called since it could throw an exception and we don't want partial state in the rest of the member data.  
      // Construction initialization actually depends on order within the class, so this placement doesn't matter here.
      , m_bestTreeNodeToSplit(std::priority_queue<TreeNode<bRegression> *, std::vector<TreeNode<bRegression> *>, CompareTreeNodeSplittingScore<bRegression>>(m_compareTreeNode)) {
      
      // an unfortunate thing about function exception handling is that accessing non-static data give undefined behavior
      // so, we can't set m_bError to true if an error occurs, so instead we set it to true in the static initialization
      // C++ guarantees that initialization will occur in the order the variables are declared (not in the order of initialization)
      // but since we put m_bError above m_bestTreeNodeToSplit and since m_bestTreeNodeToSplit is the only thing that can throw an exception
      // if an exception occurs then our m_bError will be left as true
      m_bError = false;
   } catch(...) {
      // TODO: according to the spec, it's undefined to access a non-static variable from a Function-try-block, so we can't access m_bError here  https://en.cppreference.com/w/cpp/language/function-try-block
      // so instead of setting it to true here, we set it to true by default and flip it to false if our caller gets to the constructor part
   }

   ~CachedTrainingThreadResources() {
      free(m_aThreadByteBuffer1);
      free(m_aThreadByteBuffer2);
      delete[] m_aSumPredictionStatistics;
      delete[] m_aSumPredictionStatistics1;
      delete[] m_aSumPredictionStatisticsBest;
      delete[] m_aSumResidualErrors2;
   }

   TML_INLINE void * GetThreadByteBuffer1(size_t cBytesRequired) {
      if(UNLIKELY(m_cThreadByteBufferCapacity1 < cBytesRequired)) {
         m_cThreadByteBufferCapacity1 = cBytesRequired << 1;
         void * aNewThreadByteBuffer = realloc(m_aThreadByteBuffer1, m_cThreadByteBufferCapacity1);
         if(UNLIKELY(nullptr == aNewThreadByteBuffer)) {
            return nullptr;
         }
         m_aThreadByteBuffer1 = aNewThreadByteBuffer;
      }
      return m_aThreadByteBuffer1;
   }

   TML_INLINE bool GrowThreadByteBuffer2(size_t cByteBoundaries) {
      // by adding cByteBoundaries and shifting our existing size, we do 2 things:
      //   1) we ensure that if we have zero size, we'll get some size that we'll get a non-zero size after the shift
      //   2) we'll always get back an odd number of items, which is good because we always have an odd number of TreeNodeChilden
      m_cThreadByteBufferCapacity2 = cByteBoundaries + (m_cThreadByteBufferCapacity2 << 1);
      void * aNewThreadByteBuffer = realloc(m_aThreadByteBuffer2, m_cThreadByteBufferCapacity2);
      if(UNLIKELY(nullptr == aNewThreadByteBuffer)) {
         return true;
      }
      m_aThreadByteBuffer2 = aNewThreadByteBuffer;
      return false;
   }

   TML_INLINE void * GetThreadByteBuffer2() {
      return m_aThreadByteBuffer2;
   }

   TML_INLINE size_t GetThreadByteBuffer2Size() {
      return m_cThreadByteBufferCapacity2;
   }

   TML_INLINE bool IsError() const {
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
      free(m_aThreadByteBuffer1);
   }

   TML_INLINE void * GetThreadByteBuffer1(size_t cBytesRequired) {
      if(UNLIKELY(m_cThreadByteBufferCapacity1 < cBytesRequired)) {
         m_cThreadByteBufferCapacity1 = cBytesRequired << 1;
         void * aNewThreadByteBuffer = realloc(m_aThreadByteBuffer1, m_cThreadByteBufferCapacity1);
         if(UNLIKELY(nullptr == aNewThreadByteBuffer)) {
            return nullptr;
         }
         m_aThreadByteBuffer1 = aNewThreadByteBuffer;
      }
      return m_aThreadByteBuffer1;
   }
};

#endif // CACHED_THREAD_RESOURCES_H
