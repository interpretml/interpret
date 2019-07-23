// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeaderTestCoreApi.h"

// we roll our own test framework here since it's nice having no dependencies, and we just need a few simple tests for the C API.
// If we ended up needing something more substantial, I'd consider using doctest ( https://github.com/onqtam/doctest ) because:
//   1) It's a single include file, which is the simplest we could ask for.  Googletest is more heavyweight
//   2) It's MIT licensed, so we could include the header in our project and still keep our license 100% MIT compatible without having two licenses, unlike Catch, or Catch2
//   3) It's fast to compile.
//   4) doctest is very close to having a JUnit output feature.  JUnit isn't really required, our python testing uses JUnit, so it would be nice to have the same format -> https://github.com/onqtam/doctest/blob/master/doc/markdown/roadmap.md   https://github.com/onqtam/doctest/issues/75
//   5) If JUnit is desired in the meantime, there is a converter that will output JUnit -> https://github.com/ujiro99/doctest-junit-report
//
// In case we want to use doctest in the future, use the format of the following: TEST_CASE, CHECK & FAIL_CHECK (continues testing) / REQUIRE & FAIL (stops the current test, but we could just terminate), INFO (print to log file)
// Don't implement this since it would be harder to do: SUBCASE

#include <string>
#include <stdio.h>
#include <iostream>
#include <vector>

#include "ebmcore.h"

class Test;
typedef void (* TestFunction)(Test& myTest);

class Test {
public:
   Test(TestFunction pTestFunction, std::string description) {
      m_pTestFunction = pTestFunction;
      m_description = description;
      m_bPassed = true;
   }

   TestFunction m_pTestFunction;
   std::string m_description;
   bool m_bPassed;
};

std::vector<Test> g_tests;

inline int RegisterTest(const Test& myTest) {
   g_tests.push_back(myTest);
   return 0;
}

#define COMBINE_TOKENS(t1, t2) t1##t2
#define CONCATENATE(t1, t2) COMBINE_TOKENS(t1, t2)
#define TEST_CASE(description) \
   static void CONCATENATE(MY_TEST_FUNCTION_, __LINE__)(Test& myTest); \
   static int CONCATENATE(STUPID, __LINE__) = RegisterTest(Test(CONCATENATE(MY_TEST_FUNCTION_, __LINE__), description)); \
   static void CONCATENATE(MY_TEST_FUNCTION_, __LINE__)(Test& myTest)

// for now we always terminate on error.  If it becomes a problem then make this more elegant
#define CHECK(expression) \
   do { \
      bool bFailed = !(expression); \
      if(bFailed) { \
         std::cout << " FAILED on \"" #expression << "\""; \
         myTest.m_bPassed = false; \
      } \
   } while((void)0, 0)

// for now we always terminate on error.  If it becomes a problem then make this more elegant
#define REQUIRE(expression) CHECK(expression)



TEST_CASE("test 1") {
   CHECK(true);
}

TEST_CASE("test 2") {
   CHECK(true);
   CHECK(true);
}

TEST_CASE("test 3") {
   CHECK(true);
}

void EBMCORE_CALLING_CONVENTION LogMessage(signed char traceLevel, const char * message) {
   printf("%d - %s\n", traceLevel, message);
}

int main() {
   SetLogMessageFunction(&LogMessage);
   SetTraceLevel(TraceLevelOff);

   bool bPassed = true;
   for(auto& myTest : g_tests) {
      std::cout << "Starting test: " << myTest.m_description;
      myTest.m_pTestFunction(myTest);
      if(myTest.m_bPassed) {
         std::cout << " PASSED" << std::endl;
      } else {
         bPassed = false;
         std::cout << std::endl;
      }
   }

   std::cout << "C API test " << (bPassed ? "PASSED" : "FAILED")  << std::endl;
   return bPassed ? 0 : 1;
}
