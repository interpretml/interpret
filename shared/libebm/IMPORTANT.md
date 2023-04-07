# InterpretML Important notes

## Including compute files

We use a Directory.Build.targets file to wildcard include files in the compute directory in all builds of each compute type
https://docs.microsoft.com/en-us/cpp/build/reference/vcxproj-files-and-wildcards?view=msvc-160

## One Definition Rule (ODR) considerations

BIG CHANGE: we need to create a new "separate" library that defines all the separate stuff below... we can use
this in order to create different SIMD compilations but also we'll want to be able to run different GPU compilers
on the stuff in that directory, so imagine compiling against CUDA, OpenCL and Metal all with different compilers
etc.  Since we're putting this into it's own library we can create separate .cpp files and we can use the 
non-anonymous namespace trick to keep them all very very separate when we link them all together.  Our VS studio
solution should be set to the default non-simd version and we can optionally use others, and maybe even CUDA etc
compilations.  If we use only C interfaces between internal libraries then there's a higher chance we can use
different compilers to build modules and link them together separately afterwards since .o files are more standardized
for C than C++.

C++ is in many ways a very flawed language.  One of the most insidious aspects are inadvertant violations of the 
One Definition Rule (ODR) and related issues of the Application Binary Interface (ABI) not being standardized in C++.
The worst part of ODR violations is that the compiler/linker is not required to detect them, and they can lead to
surprising crashes that can be very difficult to debug.  Some links which provide more context:

https://en.wikipedia.org/wiki/One_Definition_Rule
https://en.cppreference.com/w/cpp/language/definition
https://akrzemi1.wordpress.com/2016/11/28/the-one-definition-rule/
https://gieseanw.wordpress.com/2018/10/30/oops-i-violated-odr-again/
https://www.drdobbs.com/c-theory-and-practice/184403437
https://devblogs.microsoft.com/cppblog/diagnosing-hidden-odr-violations-in-visual-c-and-fixing-lnk2022/

In the blog from Andy Rich above, he gives an example of two *.cpp files with a shared header that are linked together, 
but are compiled with different options like this:

cl main.cpp /Zp2 /c
cl loader.cpp /Zp1 /c
link main.obj loader.obj

Which causes ODR violations between the *.cpp files because the /Zp2 and /Zp1 options control data alignment in
class/struct definitions, so the two *.cpp files have different interpretations of how the class should be layed out
and when an object of that class is passed between the cpp files they break.  The moral of this story is be very very
careful when compiling separate translation units together that have different options.  This particular packing
issue would have been a problem between *.c files as well, but the problem is more pronounced in C++ because
C at least gives you guarantees regarding POD data structures, wheras C++ provides no guarantees that different
classes will be layed out in memory in identical ways between different compilers, or different versions of the 
same compiler, or if you compile different .cpp files with different compiler options.  The only real solution is
to only use POD data structures between between translation units when the translation units are not compiled
identically, including with identical compiler switches.

For simple programs this isn't usually an issue, but it starts to become an issue when linking libraries together
to form more complicated programs.  Most people first encounter this when they try to link together .dll or .so
files compiled by either different compilers or different versions of the same compiler.  For anything C++
this usually breaks as name mangling often isn't compatible, but this is just one symptom of a larger problem that
hides in the shadows, namely that C++ libraries pretty much guarantee ODR violations.

The specific section that this violates in C++ is under "One Definition rule" where it says "
There can be more than one definition of a class type... (list other stuff).. 
in a program provided that each definition appears in a different translation unit, and provided the definitions 
satisfy the following requirements:
  - each definition of D shall consist of the same sequence of tokens
"

The problem is that "the same sequence of tokens" in practice means it needs to be compiled with identical
compiler switches to get this guarantee that multiple classes can be shared between translation units.

## Solutions to problems of the One Definition Rule (ODR)

Unfortunately for us, we have two problems that need to be resolved.  In InterpretML most of our .cpp files
can be compiled by a single version of a single compiler with identical compiler switches, BUT if we want
to gracefully handle different versions of SIMD operations within the same library we need to use different compiler
switches on some translation units to enable different SIMD operations.  We can then use dynamic detection to pick
the right SIMD function to call.  That breaks any guarantees of being able to share class definitions.  Most
packages solve this by building separate dynamically loaded libraries (.dll or .so) and attempting to hide their
internal names (which is esoteric in linux where symbols are public by default).  We could also use this method
to solve our SIMD issues, but we also want to enable usage in GPUs which is a harder but related problem.  Anything
that we want to handle with SIMD for performance reasons is also something we probably want to push to the GPU or
distributed system if available.  In the case of GPU we'll be re-compiling our code with a GPU compiler which is
very hard because we then can't share anything really between the modules.  We'll probably even have to translate
our data between little endian and big endian, nevermind worrying about class definition issues.  To solve this
we really need to have a very strong separation between our normal InterpretML code and the code meant to be SIMDed
and/or GPUed and/or distributed.  Since the only ABI interface that is cross compiler compatible is the C interface
and any data structures need to be POD and/or basic types we need to restrict ourselves to the C formats for
a significant part of the codebase and we need to be VERY careful how we share things.

For class definitions, functions, enumerations, variables, etc that don't need to be shared accross translation units
(*.cpp files) it's usually best to use internal linkage on those things.  We can do this for functions and
variables by declaring them "static", which we generally also like to combine with INLINE and/or constexpr
when appropriate.  class definitions are more problematic as static doesn't give member variables and functions
internal linkage (static just means share this accross objects in the context of classes), so we need to use
anonymous namespaces to give class definitions internal linkage.

## ODR and the standard library headers:

The C++ standard header files are an interesting case.  In theory, since we're using separate compilation flags 
we should be worried that we might be able to generate ODR violations when we compile them into separate
translation units and then shared within a "program" (C++ standardization term), in the same way that we generate
the ODR violation by changing the packing, but in practice the C++ standard headers are created by the compiler
vendor and the compiler vendor can add additional guarantees above the C++ standard that other library developers
can't get in a cross compatible way.  This happens in practice as the C++ standard library gets linked in with
your program and since you can set compiler flags it's possible to have different compiler flags than what the
C++ standard headers were compiled with.  Hopefully the C++ compiler provider guarantees these kinds of mismatches
are ok within their ecosystem.  In practice, we don't have a realistic resolution anyways since we 
can't put the C++ standard headers into namespaces as then everything in the standard libraries
would be considered by the compiler to be different objects from the entities in the standard library, so we just
have to assume the compiler writer is doing the right thing outside of any guarantees that the C++ standard gives.
It kind of sucks that the only C++ library that can provide this guarantee is the standard library and all others need
to use extern "C" functions and POD structures and/or header only libraries (or mixes of these!).

## InterpretML Zones:

For things that need to be shared between compilers and/or memory regions, we need to use POD data structures,
extern "C" functions, and completely separate objects.  To avoid issues we separate the codebase into the following:

- zone_main -> This is everythig in the root C++ directory.  This is code that stays in the main library and can use 
  C++ as much as desired as it won't be shared at all with any other zone.  We put everything into a single common
  namespace "EbmMain", mostly to protect against inadvertent sharing.  We use anonymous namespaces to localize 
  anything not shared between translation units AND use static on such functions and variables when possible.
- zone_separate -> this is code that is only meant to be used within the high performance GPU/SIMD/MPI boundary.
  We'll be re-compiling these sections with different compilers and/or different compiler options multiple times
  and linking them all together, so we are REQUIRED to put everything in this section within separate namespaces.
  to make them entirely separate, otherwise we'd have multiple different definitions of the same C++ classes
  that are created between our multiple per-compilation_type translation units.  There are two ways we can solve this
  problem.  The simpler solution is to have only 1 translation unit per compiler option 
  (so, one .cpp file per SIMD/GPU/etc option) and we wrap all inlcude files 
  (except the C++ standard ones and the zone_safe ones.. more on that later) within this zone inside a single 
  anonymous namespace.  Unfortunatly, in order to make this work we can only have 1 cpp file per zone.  This means 
  we'll need to #include stuff that we'd normally put into a cpp file into a single controlling cpp file.  Oh well..
  This solution is nice in that we only need to make one compiler pass in both VS and g++ and clang++.  In Visual
  Studio we just compile all the cpp files in one go and use separate compiler flags on each cpp file.  In g++
  and clang++ we can again put all cpp files in the compile command line but then change compiler flags via pragmas.
  The other potential solution would be to have a define that we set in the command line or in the project file
  for VS that is defined as the compile namespace name.  So, we'd use "namespace SEPARATE_NAMESPACE {}" where the
  SEPARATE_NAMESPACE was set on the command line.  This allows us to fully separate the multiple different compilations
  yet allows us to keep multiple cpp files.  It has the disadvantage that we now need multiple compile passes.  
  In g++, clang++ this just adds complexity to our build.sh file, but within the VS environment we'd have multiple
  configurations which means it's more complex to select which one we want and we'd want to move to a command line
  only build instead.
  Because zone_separate zones are in their own namespaces, within the zone_separate we can create
  as many complicated C++ classes as we like, as long as they don't escape the namespace.
- zone_c_interface -> We need to somehow transition between zone_main and zone_separate, so at a minimum we need 
  extern "C" functions, extern "C" function pointers (for callbacks), void * for passing data, and other basic 
  datatypes like int64_t, const char *, etc.  These shared functions can't be put inside namespaces since they 
  couldn't then be shared, so this zone_c_interface is needed in order to have something that we can include
  outside of any named or anonymous namespaces. In theory we could put any kind of POD data structure in here and even
  templated POD structures, but to get the maximum possible safety and avoid inadvertent bugs, we define no
  classes or structures at all and rely on the property that POD datatypes have precise and defined structure layouts.
  This zone consists of just 1 shared include file and 1 logging include file.  We write these to "C" include specs 
  such that it could be included into a C program.  We include our logging system here since logging calls to our 
  higher level caller are essentially passes from zone_separate to zone_main via the logging global variables even
  if we're bypassing the actual layers in the call to the higher level caller.
- zone_cpp_interface -> After we've used the extern "C" functions in zone_c_interface to transition between zones
  we still need to access common data structures that we share between zone_separate and zone_main.  These 
  cross zone data structures of course need to be POD since we get no guarantees on layout for C++ non-POD classes.
  We can do this sharing of data WITHOUT sharing a compiler understood class definition since we're relying instead 
  on the property that POD structures have precise and defined structure layouts.  Ideally, we'd put these POD 
  structures into "C" only headers, but unfortunately in that regard we do have some complex data types and use 
  templates to have the compiler optimize memory layout for us.  POD templates are allowed, and from that we get a 
  guarantee that we can share templated data between zones at the cost of added complexity.  Unlike for the 
  zone_c_interface though, we can wrap zone_cpp_interface headers inside a namespace for added safety
  since the class definitions don't need to be shared.  I'd probably prefer to keep these as headers only and write 
  to "C" specs other than the templating exception above, but it will be tempting to make these templated POD 
  structures have some added C++ functionality like accessor methods and other methods.  We'll see how this turns out.
  Since these headers will be wrapped inside a namespace outside the scope of the include files, do not include any 
  other headers within these include files!  We'll need to put any headers inside the .cpp translation units before 
  these headers.
  Generally this zone should not contain any utility functionality as it's meant purely to transition between 
  zone_main and zone_separate.
  I think it's legal to have nested POD structures (having a pointer to class2 inside class1 and passing class1 as 
  void over the function boundary, but access class2 without it being void).  Since pointers don't require 
  definition of the classes and pointers are part of the C ABI and this kind of layout.  But when communicating 
  cross process we can't use internal pointers, so the best general strategy is to use flat data structures that are
  one contiguous block of memory and not nest these objects and only use byte offsets within the datastructures.
- zone_shared -> in order to keep zone_cpp_interface as clean as possible, if we have other utility functions
  or C++ classes or whatever which isn't needed for communication between modules but we just want to cut down
  on the amount of code we write, we can put them in zone_shared and access them from anywhere they are included.
  This code NEEDS to be put into a unique namespace though since otherwise we'll be violating ODR rules because
  we can't make our own C++ classes ODR compliant otherwise.  This incurs a cost in that we'll be defining this
  code multiple times, but that's the cost of having ODR compliant C++ code in separate translation units with
  different compiler flags.
- zone_safe -> include files and .c files in the safe zone can be included and/or linked with no special
  requirements of being inside separate namespaces from either zone_main and/or from zone_separate.  This means 
  that we can't use C++ classes or include C++ headers (even from the standard library) ANYWHERE in these files 
  since we can't guarantee POD status for those and C++ gives us no guarantee of getting identical layouts, so we 
  could end up with different definitions of the class structures under different compiler flags.  All the code 
  in this zone is pure "C" compliant and can therefore be shared.  We can put C utility functions and whatever 
  here and they'll be shared accross calls within the process which helps cut down the size of our library.
  We might as well use the C compiler for these files as a check that these are C only.
  We also use a shared zone_safe file for common definitions like various MACROS such as LIKELY/UNLIKELY and other 
  shared compiler stuff.

## ODR detection tools:

- Visual Studio has the /ODR linker option, but it seems to find "warnings" in the C++ standard library.  I have to
  assume these are false positives that would be violations of the ODR rule but which are protected against by the
  fact the compiler writer is giving itself additional guarantees.  We should probably periodically review
  any warnings this spits out though.  Before using /ODR though we need to turn on full PDB information "DebugFull"
- ASAN has an ODR detector which is already enabled by default (ASAN_OPTIONS=detect_odr_violation=2)
  https://github.com/google/sanitizers/wiki/AddressSanitizerFlags

