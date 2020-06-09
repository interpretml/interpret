# InterpretML C++ Coding Style

## General

- Style is always a religious topic, but we do need to make some choices for consistency. Consider these rules
  as a declaration of what we've used here rather than what's "best".
- Most languages have official styles, and for those languages we use the official styles when available.
- C++ is kind of the wild west in terms of style. There are recent attempts to make an official one, but the 
  landscape is still so fragmented, so here's one more.

## Exceptions

- We use exceptions very sparingly in this codebase. The only places we use exceptions are when
  having to interface with STL containers or random number generators for which there aren't good C replacements.
  This is for a number of reasons:
    - We tend to use the struct hack a lot in this code for supporting multiclass.  Multiclass requires an array
      of residuals, and there are big performance benefits to co-locating this kind of data in the same region
      as the rest of the per-instance or per-bin or per-TreeNode data.  The struct hack requires using POD
      structures and those are incompatible with C++ classes, so for many data structures we end up using
      raw pointers to arrays of these POD structs.
    - In the future we plan to implement MPI data transfers to outside processes.  Classes like HistogramBucket
      and SegmentedTensor will need to be is_trivially_copyable compatible, which again means that we'll have
      non-C++ compatible structs.
    - We're writing a lot of performance critical code here. Templated iterators and complicated classes often
      do get optimized down to pointers in assembly for arrays, but then I need to reason about that and verify that 
      no slowdowns are introduced at the assembly level.  It's just easier to write performant code using raw
      pointers where I understand by looking at the C how it'll be converted into assembly with less abstraction.
    - We tend to allocate large amounts of memory in this library.  It's easy to allocate more than the
      system can handle since we're allocating large blocks.  We'd like to handle memory exhaustion gracefully
      and exit.  Most likely this will happen when we go and try to allocate gigabyte sized flat arrays, so
      we're probably not hitting true exhaustion without that last big allocation, so we can probably expect
      our environment to not have decayed from memory exhaustion yet, and if we tear down gracefully we can
      generate a nice exception in our caller's language rather than hard exiting.
    - Handling allocation errors in C++ is a huge pain, so much so that most programmers just pretend that memory
      never gets exhausted.  Handling OOM in C is usually easier than C++ because there are complicated issues
      like partial object construction and function try-blocks required to handle it properly.  Often pure C is
      easier than getting this stuff correct in C++ due to the added complexity in C++.
    - The lack of a standardized "finally" exception blocks makes mixing C and C++ more difficult. Raw pointers
      allocated in some parts of the code above potential exception generating calls need to be cleaned up.
      The standard way to handle this is to make everything RAII, but many of our objects will not be RAII 
      compatible for the above reasons, so it would be more complicated to conform than if we could be mostly 
      RAII compatible.  It's probably cleaner and easier to eliminate most of the exceptions rather than 
      RAIIing everything.
    - Even in the best of circumstances I'm not the biggest fan of RAII since then often things need to be put
      into objects which are located far away from the code you are writing and therefore it raises the complexity
      of reasoning about them. Improvements in other languages like lambda functions are huge improvement for this 
      reason as they keep code that needs to be reasoned about together in one place.  shared_ptr/unique_ptr 
      do resolve a lot of these issues though.
    - Because we're trying to avoid the use of exceptions per above, we try not to use new and delete at all
      and prefer malloc/free.  We substitute them for Allocate/Free or Initialize when not allocating.  Also,
      "new (std::nothrow)" seems to allow calling exit() on memory allocation failure for some platforms.
      Since we can't get a guarantee of getting back a nullptr from "new" we should probably just not call it
      if we want to be able to handle OOM scenarios.
    - We can use tools like valgrind and clang's sanitizers and also static analysis tools like clang-tidy to 
      catch most allocation issues instead of using RAII smart pointers and such.  Yeah, it would be better to 
      have both, but the existence of these tools reduces the benefits of RAII but leaves the costs of 
      implementation and reasoning complexity.
    - In this codebase we tend to allocate and free a fairly sparse number of objects compared to most other C++
      code where smaller objects are more frequently made, so we get less benefit from RAII in comparison.
    - Other places and projects like in most OSes and Google don't use exceptions for many of the reasons above
      and more: https://google.github.io/styleguide/cppguide.html#Exceptions

## Formatting

- No tabs. Use 3 spaces for indentation (3 spaces helps identify tabs which are often 4 spaces)
- max 120 characters per line
- Use K&R style opening braces '{' placed on the same line as the command that opens the block, unless you need to 
  break a line into two lines, in which case use whatever look more comprehensible
- always include opening and closing braces, even for 1 line statements that don't require them (it helps with diffs)
- if there is a constant in a comparison, put it on the left 
  eg: "if(1 == number)" -> this helps avoid the error "if(1 = number)"
- always use less than in comparisons instead of greater than
  eg: "if(1 <= something && something < 5)" instead of "if(something >= 1 && something < 5)"
- only use "auto" for complex types that UNDENIABLY don't benefit from knowing the type, like iterators
- try to preserve any other formatting used throughout the code

## Naming

- local variables: camelCase
- Functions: PascalCase
- classes/structs/typedefs/enum: PascalCase
- labels: snake_case
- anything dangerous: SCREAMING_CASE
- file names: snake_case generally, but PascalCase if the filename is meant to match a class name, like "SamplingSet.h"

## Abreviations

- We use a couple of prefix abbreviations for brevity:
  - p -> pointer
  - a -> array
  - i -> index
  - c -> count
  - b -> bool
  - m_ -> member variable (and we use these for structs too so that they don't need to change when code changes)
  - s_ -> static member variable
  - g_ -> global (global is evil, except when it's not)
  - k_ -> konstants (it's really constants, but we used c above for count)
- We use the following suffixes: First/Last/End/Prev/Cur/Next/Min/Max
