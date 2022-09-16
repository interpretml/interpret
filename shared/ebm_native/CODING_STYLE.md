# InterpretML C++ Coding Style

## Formatting

- No tabs. 3 spaces for indentation
- max 120 characters per line
- K&R style braces '{' placed on the same line as the command that opens the block, unless a line
  breaks into two lines, in which case whatever looks more comprehensible
- include opening and closing braces, even for 1 line statements that don't require them (it helps with diffs)
- if there is a constant in a comparison, it goes on the left 
  eg: "if(1 == number)" -> this helps avoid the bug "if(1 = number)"
- always use less than in comparisons instead of greater than
  eg: "if(1 <= something && something < 5)" instead of "if(something >= 1 && something < 5)"
- only use "auto" for complex types that undeniably don't benefit from knowing the type, like iterators
- preserve any other formatting used throughout the code

## Naming

- local variables: camelCase
- Functions: PascalCase
- classes/structs/typedefs/enum: PascalCase
- labels: snake_case
- anything dangerous: SCREAMING_CASE
- file names: snake_case generally, but PascalCase if the filename is meant to match a PascalCase class name

## Abreviations

- We use a couple of prefix abbreviations for brevity:
  - p -> pointer
  - a -> array
  - s -> pointer to C style string characters
  - i -> index
  - c -> count
  - b -> bool
  - m_ -> member variable, and we use these for structs too so that they don't need to change when code changes
  - s_ -> static member variable in a class
  - g_ -> global (global is evil, except when it's not)
  - k_ -> konstants (it's really constants, but we used c above for count)
- We use the following suffixes: First/Last/End/Prev/Cur/Next/Min/Max

## Exceptions and C-like coding style

- Using exceptions and C++ oriented style in the testing code is perfectly fine.
- For the core library, we use exceptions very sparingly. The only places we use exceptions are when
  having to interface with STL classes for which there aren't good C replacements, and for code the end user might
  want to change, like custom Loss function classes
  This is for a number of reasons:
    - our C++ library doesn't have a lot of depenencies.  Out of memory errors are almost the only error that
      we generate.
    - We tend to allocate large amounts of memory in this library, and it's easy for us to allocate more than the
      system can handle.  We'd like to handle memory exhaustion gracefully without terminating the process.  
      Most likely this will happen when we try to allocate gigabyte sized flat arrays, so we're probably
      not hitting true exhaustion until trying to allocate that last big allocation, so we can probably expect 
      that our environment has not decayed yet from being close to memory exhaustion, and if we tear down 
      gracefully we can generate a nice exception in our caller's language rather than hard exiting the process.
    - Handling allocation errors in C++ is a huge pain, so much so that most programmers just pretend that memory
      never gets exhausted.  Handling OOM in C is usually easier than in C++ because there are complicated issues
      like partial object construction and function try-blocks required to handle it properly.  Often pure C is
      easier than getting this stuff done correct in C++ due to the added complexity in C++.
    - The lack of standardized "finally" exception blocks makes mixing C and C++ more difficult. Raw pointers
      allocated in parts of the code located above potential exception generating calls need to be cleaned up.
      The standard way to handle this is in C++ is to make everything RAII, but many of our objects will not be RAII
      for reasons detailed below, so it would be more complicated to conform than if we could be mostly RAII 
      compatible.  It's probably cleaner and easier to eliminate most of the exceptions rather than RAIIing everything.
    - We tend to use the struct hack a lot in this code for supporting multiclass.  Multiclass requires an array
      of gradients/hessians/logits, and there are performance benefits to co-locating this kind of data in the same 
      region as the rest of the per-sample or per-bin or per-TreeNode data.  The struct hack requires using POD
      structures and those are incompatible with C++ classes, so for many data structures we end up using
      raw pointers to arrays of these POD structs.  Adding RAII wrappers arround these would add complexity.
    - In the future we plan to implement MPI data transfers to outside processes.  Classes like Bin
      and Tensor will need to be is_trivially_copyable compatible, which again means that we'll have
      non-C++ compatible structs.
    - We're writing a lot of performance critical code here. Templated iterators and complicated classes often
      do get optimized down to pointers in assembly for arrays, but then we need to reason about that, and verify 
      that no slowdowns are introduced at the assembly level.  It's just easier to write performant code using raw
      pointers where it's easier to understand the assembly translation by looking just at the C code.
    - Even in the best of circumstances I'm not the biggest fan of RAII since pointers need to be put
      into objects which are located far away from the code you are writing and therefore it raises the complexity
      of reasoning about them. Improvements in other languages like lambda functions are big improvement for this 
      reason as they keep code that needs to be reasoned about together in one place.  shared_ptr/unique_ptr 
      do resolve a lot of these issues though, as do lambda functions in C++
    - in some places when using dynamically sized POD structs, we are required to use malloc/free and manual memory
      management.  We moved towards using malloc/free for all allocations because early on we found some bugs where
      memory was allocated with new and deleted with free, which is an error.  Keeping everyting uniformly 
      malloc/free avoids this kind of error
    - Also, because we're trying to avoid the use of exceptions per above, we try not to use new and delete at all
      and prefer malloc/free.  We substitute them for Allocate/Create/Free or Initialize when not allocating.  Also,
      "new (std::nothrow)" seems to allow calling exit() on memory allocation failure for some platforms, so it's
      more portable to catch std::bad_alloc instead.
    - We can use tools like valgrind and clang's sanitizers and also static analysis tools like clang-tidy to 
      catch most allocation issues instead of using RAII smart pointers and such.  Yes, it would be better to 
      have both, but the existence of these tools reduces the benefits of RAII but leaves the costs of 
      implementation and reasoning complexity.
    - In this codebase we tend to allocate and free a fairly sparse number of objects compared to most other C++
      code where smaller objects are more frequently made, so we get less benefit from RAII in comparison.
    - Other projects, like most OSes, and throughout Google don't use exceptions for many of the reasons above
      and others: https://google.github.io/styleguide/cppguide.html#Exceptions
    - our higher level language to C++ interface boundary uses pure C (the only portable solution), so we can't 
      throw exceptions accross that boundary, so ultimately we need error codes at that level anyways.

- use the following order throughout for function declarations: 
  "INLINE_RELEASE_UNTEMPLATED constexpr static"
  INLINE_RELEASE_UNTEMPLATED needs to be first since it's a template under DEBUG builds for functions
  constexpr next so that it doesn't sit next to const if our function returns a const type
  static last since that's what's left
  NOTE: before C++17 "inline static" and "static inline" were both ok, but the standard obsoleted "inline static"
        https://stackoverflow.com/questions/61714110/static-inline-vs-inline-static
        However, since we use templating for INLINE_RELEASE_UNTEMPLATED we require it to be first, and since we want 
        INLINE_ALWAYS to be in the same position, we use this obsolete ordering until the compilers start to complain
- use the following order throughout for variable declarations: 
  "static const"  eg: "static const int k_myInt = 5;"
  AND
  "static constexpr"  eg: "static constexpr int k_myInt = 5;"
  For variables const should be next to the type since it is more important for understanding the variable than static
  This ordering is opposite the order we use for functions, but that is because const/constexpr are in this
  case modifying the variable as opposed to modifying the function.
  This also has the advantage of conforming to the new C++17 change that requires static to preceed const
  https://stackoverflow.com/questions/61714110/static-inline-vs-inline-static
