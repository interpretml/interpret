#!/bin/sh

# This script is written as Bourne shell and is POSIX compliant to have less interoperability issues between distros and MacOS.
# it's a good idea to run this script periodically through a shell script checker like https://www.shellcheck.net/

# Periodically check the valgrind results on the build server by going to:
# https://dev.azure.com/ms/interpret/_build?definitionId=293&_a=summary
# By clicking on the build of interest, then "Test ebm_native Linux", then "View raw log"
# We normally have 2 "still reachable" blocks from the Testing executable.
#
# We run the clang-tidy and Visual Studio static analysis tools on the build server.  Warnings do not stop the build, 
# so these need to be inspected to catch static analysis issues.  The results can be viewed in the build logs here:
# https://dev.azure.com/ms/interpret/_build?definitionId=293&_a=summary
# By clicking on the build of interest, then "Build ebm_native Windows", then "View raw log"
#
# Ideally we'd prefer to have static analysis running on all OSes, but we can probably just rely on our
# build system to handle this aspect since adding it in multiple places adds complexity to this build script.
# Also, I ran into issues when I first tried it, which to me suggests these might introduce periodic issues:
#   - on Mac, clang-tidy doesn't seem to come by default in the OS.  You are suposed to 
#     "brew reinstall llvm", but I got a message that llvm was part of the OS and it suggested 
#     that upgrading was a very bad idea.  You could also compile it from scratch, but this seems
#     to me like it would complicate this build script too much for the benefit
#   - on Linux, I was able to get clang-tidy to work by using "sudo apt-get -y install clang clang-tidy"
#     but this requires installing clang and clang-tidy.  I have a better solution using Visual Studio
#   - on Ubuntu, "sudo apt-get -y install cppcheck" seems to hang my build machines, so that sucks.
#   - Visual Studio now includes support for both it's own static analysis tool and clang-tidy.  This seems to
#     be the easiest ways to access these tools for us since they require no additional installation.
#   - by adding "/p:EnableClangTidyCodeAnalysis=True /p:RunCodeAnalysis=True" to MSBuild I can get the static
#     analysis tools to run on the build system, but they won't run in typical builds in Visual Studio, which
#     would slow down our builds.
#   - If you want to enable these static checks on build in Visual Studio, go to:
#     "Solution Explorer" -> right click the project "ebm_native" -> "Properties" -> "Code Analysis"
#     From there you can enable "Clang-Tidy" and "Enable Code Analysis on Build"
#   - You also for free see the Visual Studio static analysis in the "Error List" window if you have
#     "Build + IntelliSense" selected in the drop down window with that option.
#   - any static analysis warnings don't kill the build it seems.  That's good since static analysis tool warnings
#     constantly change, so we probably don't want to turn them into errors otherwise it'll constantly be breaking.
#   - https://include-what-you-use.org/ is alpha, and it looks like it changes a lot.  Doesn't seem worth the benefit.
#   - NOTE: scan-build and clang-tidy are really the same thing, but with different interfaces

# The only things that Linux makes illegal in filenames are the zero termination character '\0' and the path character '/'
# This makes it hard to write shell scripts that handle things like control characters, spaces, newlines, etc properly.
# In Linux filenames and directories often don't have spaces and scripts often don't handle them.  In Windows
# we often do have spaces in directory names though, and we want to be able to build from the Windows Bash shell,
# so we handle them here.
# We need to pass multiple directory/filenames pairs into g++/clang++ from multiple different directories 
# (making find hard), and we also have additional files we use (see --version-script), so we need to build up 
# variables that contain the filesnames to pass into the compiler. This reference says there is no portable way 
# to handle spaces (see 1.4 Template: Building up a variable), 
# so they recommend setting IFS to tab and newline, which means files with tab and newline won't work:
# https://dwheeler.com/essays/filenames-in-shell.html
# I'm not quite sure that's true.  If we use the often frowned upon eval command we can put single quotes arround
# any strings that we use, and eval will parse them correctly with spaces.  This opens us up to issues with eval, 
# so we need to sanitize our strings beforehand similar to this discussion, although printf "%q" is too new
# currently (2016) for us to take a dependency on it, so we use sed to sanitize our single quote filenames instead:
# https://stackoverflow.com/questions/17529220/why-should-eval-be-avoided-in-bash-and-what-should-i-use-instead
#
# Here are some good references on the issues regarding odd characters in filenames: 
# - https://dwheeler.com/essays/filenames-in-shell.html
# - https://dwheeler.com/essays/fixing-unix-linux-filenames.html
# These should work in our script below (some have been explicitly tested):
# - newlines in files "a.cpp\nb.cpp" which can be interpreted as separate files in some scripts
# - files that start with '-' characters.  eg: "-myfile.txt" which can be interpreted as arguments in some scripts
# - files that begin or end or contain spaces eg: "  a.cpp  b.cpp  " which get stripped or turned into multiple arguments
# - tab characters "\t"
# - string escape characters "\\"
# - quote characters "\"" or "\'" or "\`"
# - asterix in filenames "*" eg: "*.cpp" which can get glob expanded
# - ';' character -> which can be used to run new shell commands in some scripts
# - control characters (ASCII 1-31)
# - UTF-8 characters
# - try the following stress test case (works on windows):     ./-in ter;_Pr`e't/shared/ebm_native/-ha rd;_Fi`l'e.cpp
# We also cannot use the following safely:
# - find exec with the \; ending since it eats the return codes of our compiler, which we really really want!
# - raw "exec" without re-shelling the result
#   https://unix.stackexchange.com/questions/156008/is-it-possible-to-use-find-exec-sh-c-safely
# - ld -> for C++ programs we need to know implementation specific directories and object files for initialization.
#   We re-call g++/clang++ to link with object files instead.  We need to separate our compile and link steps because
#   we compile the same C++ files over several times with different compiler options so these need to generate
#   separated .o files with different names.  I'm not sure if GNU make/cmake handles this natievly if we go that route.
# - make -> well, we might use make someday if compile speed becomes an issue, but I like that this script doesn't 
#   require installing anything before calling it on either standard mac or linux machines, and GNU make requires installation 
#   on macs and isn't part of the standard clang++ build pipeline.  cmake also requires installation.  Bourne shell POSIX script 
#   is the most out of the box compatible solution.  Also, per above, I'm not sure if make/cmake handles duplicate
#   compilation of .cpp files multiple times with different compiler options
# - tee -> we write the compiler output to both a file and to stdout, but tee swallows any error codes in the compiler
#   I've been using backticks to store the output in a variable first, which is frowned upon, so consider
#   command substitution instead, which is now even part of the more recent bourne shells and is POSIX compliant now
#   PIPESTATUS is bash specific
# - no command substitution (although I might change my mind on this given POSIX now supports command substitution).  
#   We don't use backticks much here so it's low cost to use the older method.  Backticks are more compatible, 
#   but also command substitution seems to remove trailing newlines, which although esoteric introduces an error 
#   condition we'd want to at least investigate and/or check and/or handle.
#   https://superuser.com/questions/403800/how-can-i-make-the-bash-backtick-operator-keep-newlines-in-output/827879
# - don't use echo "$something".  Use printf "%s" "$something"

sanitize() {
   # use this techinque where single quotes are expanded to '\'' (end quotes insert single quote, start quote)
   # but fixed from the version in this thread: 
   # https://stackoverflow.com/questions/15783701/which-characters-need-to-be-escaped-when-using-bash
   # https://stackoverflow.com/questions/17529220/why-should-eval-be-avoided-in-bash-and-what-should-i-use-instead
   printf "%s" "$1" | sed "s/'/'\\\\''/g; 1s/^/'/; \$s/\$/'/"
}

get_file_body() {
   # https://www.oncrashreboot.com/use-sed-to-split-path-into-filename-extension-and-directory
   printf "%s" "$1" | sed 's/\(.*\)\/\(.*\)\.\(.*\)$/\2/'
}

check_install() {
   l1_tmp_path_unsanitized="$1"
   l1_package="$2"
   
   if [ ! -f "$l1_tmp_path_unsanitized/$l1_package.chk" ]; then
      printf "%s\n" "Installing $l1_package"

      if [ "$g_is_updated" -eq 0 ]; then 

         sudo apt-get -y update
         l1_ret_code=$?
         if [ $l1_ret_code -ne 0 ]; then 
            exit $l1_ret_code
         fi

         g_is_updated=1
      fi

      sudo apt-get -y install "$l1_package"
      l1_ret_code=$?
      if [ $l1_ret_code -ne 0 ]; then 
         exit $l1_ret_code
      fi
         
      # write out an empty file to signal that this has been installed
      printf "" > "$l1_tmp_path_unsanitized/$l1_package.chk"
      l1_ret_code=$?
      if [ $l1_ret_code -ne 0 ]; then 
         exit $l1_ret_code
      fi
   fi
}

make_initial_paths_simple() {
   l2_obj_path_unsanitized="$1"
   l2_bin_path_unsanitized="$2"

   [ -d "$l2_obj_path_unsanitized" ] || mkdir -p "$l2_obj_path_unsanitized"
   l2_ret_code=$?
   if [ $l2_ret_code -ne 0 ]; then 
      exit $l2_ret_code
   fi
   [ -d "$l2_bin_path_unsanitized" ] || mkdir -p "$l2_bin_path_unsanitized"
   l2_ret_code=$?
   if [ $l2_ret_code -ne 0 ]; then 
      exit $l2_ret_code
   fi
}

compile_file() {
   l3_compiler="$1"
   l3_compiler_args_sanitized="$2"
   l3_file_unsanitized="$3"
   l3_obj_path_unsanitized="$4"
   l3_asm="$5"
   l3_zone="$6"

   l3_file_sanitized=`sanitize "$l3_file_unsanitized"`
   l3_file_body_unsanitized=`get_file_body "$l3_file_unsanitized"`
   l3_object_full_file_unsanitized="$l3_obj_path_unsanitized/${l3_file_body_unsanitized}_$l3_zone.o"
   l3_object_full_file_sanitized=`sanitize "$l3_object_full_file_unsanitized"`
   g_all_object_files_sanitized="$g_all_object_files_sanitized $l3_object_full_file_sanitized"
   l3_compile_specific="$l3_compiler $l3_compiler_args_sanitized -c $l3_file_sanitized -o $l3_object_full_file_sanitized 2>&1"
   l3_compile_out=`eval "$l3_compile_specific"`
   l3_ret_code=$?
   g_compile_out_full="$g_compile_out_full$l3_compile_out"
   if [ $l3_ret_code -ne 0 ]; then 
      printf "%s\n" "$g_compile_out_full"
      printf "%s\n" "$g_compile_out_full" > "$g_log_file_unsanitized"
      exit $l3_ret_code
   fi

   if [ $l3_asm -ne 0 ]; then
      # - I'd rather do our real compile above with no special parameters because I'm not confident the compiler would
      #   produce the same output if we included extra debugger info disassembly commands.  It's better to stick with 
      #   the normal program flow for our shared library output.  This rules out: --save-temps=obj
      #   -Wa,-adhln=myoutput.s can be used in-stream, but we've ruled this out per above.  We can also use it
      #   to generate the .s file below, but I found that this didn't have much benefit over -S and -fverbose-asm
      #   We also write out objdump disassembly from the final library output itself which should allow us to 
      #   check that this annotated assembly is the same as what gets finally generated
      # - https://panthema.net/2013/0124-GCC-Output-Assembler-Code/
      # - https://stackoverflow.com/questions/137038/how-do-you-get-assembler-output-from-c-c-source-in-gcc
      # - https://linux.die.net/man/1/as

      # If this fails then ignore the error and we'll just be missing this file.
      l3_asm_full_file_unsanitized="$l3_obj_path_unsanitized/${l3_file_body_unsanitized}_$l3_zone.s"
      l3_asm_full_file_sanitized=`sanitize "$l3_asm_full_file_unsanitized"`
      l3_compile_specific_asm="$l3_compiler $l3_compiler_args_sanitized -fverbose-asm -S $l3_file_sanitized -o $l3_asm_full_file_sanitized 2>&1"
      l3_compile_out_asm=`eval "$l3_compile_specific_asm"`
   fi
}

compile_directory_c() {
   l4_compiler="$1"
   l4_compiler_args_sanitized="$2"
   l4_src_path_unsanitized="$3"
   l4_obj_path_unsanitized="$4"
   l4_asm="$5"
   l4_zone="$6"

   # zsh (default shell in macs) terminates if you try to glob expand zero results, so check first
   find "$l4_src_path_unsanitized" -maxdepth 1 -type f -name '*.c' 2>/dev/null | grep -q .
   l4_ret_code=$?
   if [ $l4_ret_code -eq 0 ]; then 
      # use globs with preceeding directory per: https://dwheeler.com/essays/filenames-in-shell.html
      for l4_file_unsanitized in "$l4_src_path_unsanitized"/*.c ; do
         # glob expansion returns *.c when there are no matches, so we need to check for the existance of the file
         if [ -f "$l4_file_unsanitized" ] ; then
            compile_file "$l4_compiler" "$l4_compiler_args_sanitized" "$l4_file_unsanitized" "$l4_obj_path_unsanitized" "$l4_asm" "$l4_zone"
         fi
      done
   fi
}

compile_directory_cpp() {
   l5_compiler="$1"
   l5_compiler_args_sanitized="$2"
   l5_src_path_unsanitized="$3"
   l5_obj_path_unsanitized="$4"
   l5_asm="$5"
   l5_zone="$6"

   # zsh (default shell in macs) terminates if you try to glob expand zero results, so check first
   find "$l5_src_path_unsanitized" -maxdepth 1 -type f -name '*.cpp' 2>/dev/null | grep -q .
   l5_ret_code=$?
   if [ $l5_ret_code -eq 0 ]; then 
      # use globs with preceeding directory per: https://dwheeler.com/essays/filenames-in-shell.html
      for l5_file_unsanitized in "$l5_src_path_unsanitized"/*.cpp ; do
         # glob expansion returns *.cpp when there are no matches, so we need to check for the existance of the file
         if [ -f "$l5_file_unsanitized" ] ; then
            compile_file "$l5_compiler" "$l5_compiler_args_sanitized" "$l5_file_unsanitized" "$l5_obj_path_unsanitized" "$l5_asm" "$l5_zone"
         fi
      done
   fi
}

link_file() {
   l7_linker="$1"
   l7_linker_args_sanitized="$2"
   l7_bin_path_unsanitized="$3"
   l7_bin_file="$4"

   l7_bin_path_sanitized=`sanitize "$l7_bin_path_unsanitized"`
   # the linker wants to have the most dependent .o/.so/.dylib files listed FIRST
   l7_compile_specific="$l7_linker $g_all_object_files_sanitized $l7_linker_args_sanitized -o $l7_bin_path_sanitized/$l7_bin_file 2>&1"
   l7_compile_out=`eval "$l7_compile_specific"`
   l7_ret_code=$?
   g_compile_out_full="$g_compile_out_full$l7_compile_out"
   if [ $l7_ret_code -ne 0 ]; then 
      printf "%s\n" "$g_compile_out_full"
      printf "%s\n" "$g_compile_out_full" > "$g_log_file_unsanitized"
      exit $l7_ret_code
   fi
}


g_is_updated=0

debug_64=1
release_64=1
debug_32=0
release_32=0
release_arm=0
debug_arm=0

existing_debug_64=0
existing_release_64=0
existing_debug_32=0
existing_release_32=0
existing_debug_arm=0
existing_release_arm=0

use_valgrind=1
use_asan=1

for arg in "$@"; do
   if [ "$arg" = "-no_debug_64" ]; then
      debug_64=0
   fi
   if [ "$arg" = "-no_release_64" ]; then
      release_64=0
   fi
   if [ "$arg" = "-debug_32" ]; then
      debug_32=1
   fi
   if [ "$arg" = "-release_32" ]; then
      release_32=1
   fi
   if [ "$arg" = "-debug_arm" ]; then
      debug_arm=1
   fi
   if [ "$arg" = "-release_arm" ]; then
      release_arm=1
   fi

   if [ "$arg" = "-existing_debug_64" ]; then
      existing_debug_64=1
   fi
   if [ "$arg" = "-existing_release_64" ]; then
      existing_release_64=1
   fi
   if [ "$arg" = "-existing_debug_32" ]; then
      existing_debug_32=1
   fi
   if [ "$arg" = "-existing_release_32" ]; then
      existing_release_32=1
   fi
   if [ "$arg" = "-existing_debug_arm" ]; then
      existing_debug_arm=1
   fi
   if [ "$arg" = "-existing_release_arm" ]; then
      existing_release_arm=1
   fi

   if [ "$arg" = "-no_valgrind" ]; then
      use_valgrind=0
   fi
   if [ "$arg" = "-no_asan" ]; then
      use_asan=0
   fi
done

# this isn't needed in the test script, but we include them to make this script more similar to build.sh
is_asm=0

# TODO: this could be improved upon.  There is no perfect solution AFAIK for getting the script directory, and I'm not too sure how the CDPATH thing works
# Look at BASH_SOURCE[0] as well and possibly select either it or $0
# The output here needs to not be the empty string for glob substitution below:
script_path_initial=`dirname -- "$0"`
# the space after the '= ' character is required
script_path_unsanitized=`CDPATH= cd -- "$script_path_initial" && pwd -P`
if [ ! -f "$script_path_unsanitized/ebm_native_test.sh" ] ; then
   # there are all kinds of reasons why we might not have gotten the script path in $0.  It's more of a convention
   # than a requirement to have either the full path or even the script itself.  There are far more complicated
   # scripts out there that attempt to use various shell specific workarounds, like BASH_SOURCE[0] to best solve
   # the problem, but it's possible in theory to be running over an SSL connection without a script on the local
   # system at all, so getting the directory is a fundamentally unsolved problem.  We can terminate though if
   # we find ourselves in such a weird condition.  This also happens when the "source" command is used.
   printf "Could not find script file root directory for building InterpretML.  Exiting."
   exit 1
fi

root_path_unsanitized="$script_path_unsanitized/../../.."
tmp_path_unsanitized="$root_path_unsanitized/tmp"
staging_path_unsanitized="$root_path_unsanitized/staging"
staging_path_sanitized=`sanitize "$staging_path_unsanitized"`
src_path_unsanitized="$script_path_unsanitized"
src_path_sanitized=`sanitize "$src_path_unsanitized"`

bin_file="ebm_native_test"

# re-enable these warnings when they are better supported by g++ or clang: -Wduplicated-cond -Wduplicated-branches -Wrestrict
both_args=""
both_args="$both_args -Wall -Wextra"
both_args="$both_args -Wunused-result"
both_args="$both_args -Wno-parentheses"
both_args="$both_args -Wdouble-promotion"
both_args="$both_args -Wshadow"
both_args="$both_args -Wformat=2"
both_args="$both_args -fvisibility=hidden"
both_args="$both_args -fno-math-errno -fno-trapping-math"
both_args="$both_args -I$src_path_sanitized/../inc"
both_args="$both_args -I$src_path_sanitized"

c_args="-std=c99"

cpp_args="-std=c++11"
cpp_args="$cpp_args -Wold-style-cast"
cpp_args="$cpp_args -fvisibility-inlines-hidden"

link_args=""

os_type=`uname`

if [ "$os_type" = "Linux" ]; then
   # "readelf -d <lib_filename.so>" should show library rpath:    $ORIGIN/    OR    ${ORIGIN}/    for Linux so that the console app will find the ebm_native library in the same directory as the app: https://stackoverflow.com/questions/6288206/lookup-failure-when-linking-using-rpath-and-origin
   # the -l<library> parameter for some reason adds a lib at the start and .so at the end

   c_compiler=gcc
   cpp_compiler=g++

   # try moving some of these g++ specific warnings into both_args if clang eventually supports them
   both_args="$both_args -Wlogical-op"
   both_args="$both_args -march=core2"

   # the linker wants to have the most dependent .o/.so/.dylib files listed FIRST
   link_args="$link_args -L$staging_path_sanitized"
   link_args="$link_args -Wl,-rpath-link,$staging_path_sanitized"
   link_args="$link_args -Wl,-rpath,'\$ORIGIN/'"
   link_args="$link_args -Wl,--exclude-libs,ALL"
   link_args="$link_args -Wl,-z,relro,-z,now"
   link_args="$link_args -static-libgcc"
   link_args="$link_args -static-libstdc++"

   if [ $debug_64 -eq 1 ]; then
      ########################## Linux debug|x64

      if [ $existing_debug_64 -eq 0 ]; then 
         /bin/sh "$root_path_unsanitized/build.sh" -no_release_64 -analysis
         ret_code=$?
         if [ $ret_code -ne 0 ]; then 
            # build.sh should write out any messages
            exit $ret_code
         fi
      fi

      printf "%s\n" "Compiling ebm_native_test with $c_compiler/$cpp_compiler for Linux debug|x64"
      obj_path_unsanitized="$tmp_path_unsanitized/gcc/obj/debug/linux/x64/ebm_native_test"
      bin_path_unsanitized="$tmp_path_unsanitized/gcc/bin/debug/linux/x64/ebm_native_test"
      lib_file_body="_ebm_native_linux_x64_debug"
      g_log_file_unsanitized="$obj_path_unsanitized/ebm_native_test_debug_linux_x64_build_log.txt"
      both_args_extra="-m64 -O1"
      c_args_specific="$c_args $both_args $both_args_extra"
      cpp_args_specific="$cpp_args $both_args $both_args_extra"
      # the linker wants to have the most dependent .o/.so/.dylib files listed FIRST
      link_args_specific="-l$lib_file_body $link_args $cpp_args_specific"
   
      g_all_object_files_sanitized=""
      g_compile_out_full=""

      make_initial_paths_simple "$obj_path_unsanitized" "$bin_path_unsanitized"
      if [ $use_valgrind -ne 0 ]; then 
         check_install "$tmp_path_unsanitized" "valgrind"
      fi
      compile_directory_c "$c_compiler" "$c_args_specific" "$src_path_unsanitized" "$obj_path_unsanitized" "$is_asm" "test"
      compile_directory_cpp "$cpp_compiler" "$cpp_args_specific" "$src_path_unsanitized" "$obj_path_unsanitized" "$is_asm" "test"
      link_file "$cpp_compiler" "$link_args_specific" "$bin_path_unsanitized" "$bin_file"
      printf "%s\n" "$g_compile_out_full"
      printf "%s\n" "$g_compile_out_full" > "$g_log_file_unsanitized"

      cp "$staging_path_unsanitized/lib$lib_file_body.so" "$bin_path_unsanitized/"
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
      if [ $use_valgrind -eq 0 ]; then 
         "$bin_path_unsanitized/$bin_file"
      else
         valgrind --error-exitcode=99 --leak-check=yes "$bin_path_unsanitized/$bin_file"
      fi
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
   fi

   if [ $release_64 -eq 1 ]; then
      ########################## Linux release|x64

      if [ $existing_release_64 -eq 0 ]; then 
         /bin/sh "$root_path_unsanitized/build.sh" -no_debug_64 -analysis
         ret_code=$?
         if [ $ret_code -ne 0 ]; then 
            # build.sh should write out any messages
            exit $ret_code
         fi
      fi

      printf "%s\n" "Compiling ebm_native_test with $c_compiler/$cpp_compiler for Linux release|x64"
      obj_path_unsanitized="$tmp_path_unsanitized/gcc/obj/release/linux/x64/ebm_native_test"
      bin_path_unsanitized="$tmp_path_unsanitized/gcc/bin/release/linux/x64/ebm_native_test"
      lib_file_body="_ebm_native_linux_x64"
      g_log_file_unsanitized="$obj_path_unsanitized/ebm_native_test_release_linux_x64_build_log.txt"
      both_args_extra="-m64 -DNDEBUG -O1"
      c_args_specific="$c_args $both_args $both_args_extra"
      cpp_args_specific="$cpp_args $both_args $both_args_extra"
      # the linker wants to have the most dependent .o/.so/.dylib files listed FIRST
      link_args_specific="-l$lib_file_body $link_args $cpp_args_specific"
   
      g_all_object_files_sanitized=""
      g_compile_out_full=""

      make_initial_paths_simple "$obj_path_unsanitized" "$bin_path_unsanitized"
      if [ $use_valgrind -ne 0 ]; then 
         check_install "$tmp_path_unsanitized" "valgrind"
      fi
      compile_directory_c "$c_compiler" "$c_args_specific" "$src_path_unsanitized" "$obj_path_unsanitized" "$is_asm" "test"
      compile_directory_cpp "$cpp_compiler" "$cpp_args_specific" "$src_path_unsanitized" "$obj_path_unsanitized" "$is_asm" "test"
      link_file "$cpp_compiler" "$link_args_specific" "$bin_path_unsanitized" "$bin_file"
      printf "%s\n" "$g_compile_out_full"
      printf "%s\n" "$g_compile_out_full" > "$g_log_file_unsanitized"

      cp "$staging_path_unsanitized/lib$lib_file_body.so" "$bin_path_unsanitized/"
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
      if [ $use_valgrind -eq 0 ]; then 
         "$bin_path_unsanitized/$bin_file"
      else
         valgrind --error-exitcode=99 --leak-check=yes "$bin_path_unsanitized/$bin_file"
      fi
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
   fi

   if [ $debug_32 -eq 1 ]; then
      ########################## Linux debug|x86

      if [ $existing_debug_32 -eq 0 ]; then 
         /bin/sh "$root_path_unsanitized/build.sh" -no_release_64 -no_debug_64 -debug_32 -analysis
         ret_code=$?
         if [ $ret_code -ne 0 ]; then 
            # build.sh should write out any messages
            exit $ret_code
         fi
      fi

      printf "%s\n" "Compiling ebm_native_test with $c_compiler/$cpp_compiler for Linux debug|x86"
      obj_path_unsanitized="$tmp_path_unsanitized/gcc/obj/debug/linux/x86/ebm_native_test"
      bin_path_unsanitized="$tmp_path_unsanitized/gcc/bin/debug/linux/x86/ebm_native_test"
      lib_file_body="_ebm_native_linux_x86_debug"
      g_log_file_unsanitized="$obj_path_unsanitized/ebm_native_test_debug_linux_x86_build_log.txt"
      both_args_extra="-msse2 -mfpmath=sse -m32 -O1"
      c_args_specific="$c_args $both_args $both_args_extra"
      cpp_args_specific="$cpp_args $both_args $both_args_extra"
      # the linker wants to have the most dependent .o/.so/.dylib files listed FIRST
      link_args_specific="-l$lib_file_body $link_args $cpp_args_specific"
   
      g_all_object_files_sanitized=""
      g_compile_out_full=""

      make_initial_paths_simple "$obj_path_unsanitized" "$bin_path_unsanitized"
      compile_directory_c "$c_compiler" "$c_args_specific" "$src_path_unsanitized" "$obj_path_unsanitized" "$is_asm" "test"
      compile_directory_cpp "$cpp_compiler" "$cpp_args_specific" "$src_path_unsanitized" "$obj_path_unsanitized" "$is_asm" "test"
      link_file "$cpp_compiler" "$link_args_specific" "$bin_path_unsanitized" "$bin_file"
      printf "%s\n" "$g_compile_out_full"
      printf "%s\n" "$g_compile_out_full" > "$g_log_file_unsanitized"

      cp "$staging_path_unsanitized/lib$lib_file_body.so" "$bin_path_unsanitized/"
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
      "$bin_path_unsanitized/$bin_file"
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
   fi

   if [ $release_32 -eq 1 ]; then
      ########################## Linux release|x86

      if [ $existing_release_32 -eq 0 ]; then 
         /bin/sh "$root_path_unsanitized/build.sh" -no_release_64 -no_debug_64 -release_32 -analysis
         ret_code=$?
         if [ $ret_code -ne 0 ]; then 
            # build.sh should write out any messages
            exit $ret_code
         fi
      fi

      printf "%s\n" "Compiling ebm_native_test with $c_compiler/$cpp_compiler for Linux release|x86"
      obj_path_unsanitized="$tmp_path_unsanitized/gcc/obj/release/linux/x86/ebm_native_test"
      bin_path_unsanitized="$tmp_path_unsanitized/gcc/bin/release/linux/x86/ebm_native_test"
      lib_file_body="_ebm_native_linux_x86"
      g_log_file_unsanitized="$obj_path_unsanitized/ebm_native_test_release_linux_x86_build_log.txt"
      both_args_extra="-msse2 -mfpmath=sse -m32 -DNDEBUG -O1"
      c_args_specific="$c_args $both_args $both_args_extra"
      cpp_args_specific="$cpp_args $both_args $both_args_extra"
      # the linker wants to have the most dependent .o/.so/.dylib files listed FIRST
      link_args_specific="-l$lib_file_body $link_args $cpp_args_specific"
   
      g_all_object_files_sanitized=""
      g_compile_out_full=""

      make_initial_paths_simple "$obj_path_unsanitized" "$bin_path_unsanitized"
      compile_directory_c "$c_compiler" "$c_args_specific" "$src_path_unsanitized" "$obj_path_unsanitized" "$is_asm" "test"
      compile_directory_cpp "$cpp_compiler" "$cpp_args_specific" "$src_path_unsanitized" "$obj_path_unsanitized" "$is_asm" "test"
      link_file "$cpp_compiler" "$link_args_specific" "$bin_path_unsanitized" "$bin_file"
      printf "%s\n" "$g_compile_out_full"
      printf "%s\n" "$g_compile_out_full" > "$g_log_file_unsanitized"

      cp "$staging_path_unsanitized/lib$lib_file_body.so" "$bin_path_unsanitized/"
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
      "$bin_path_unsanitized/$bin_file"
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
   fi

elif [ "$os_type" = "Darwin" ]; then
   # reference on rpath & install_name: https://www.mikeash.com/pyblog/friday-qa-2009-11-06-linking-and-install-names.html
   # the -l<library> parameter for some reason adds a lib at the start and .dylib at the end

   c_compiler=clang
   cpp_compiler=clang++

   # try moving some of these clang specific warnings into both_args if g++ eventually supports them
   both_args="$both_args -Wnull-dereference"
   both_args="$both_args -Wgnu-zero-variadic-macro-arguments"

   # the linker wants to have the most dependent .o/.so/.dylib files listed FIRST
   link_args="$link_args -L$staging_path_sanitized"
   link_args="$link_args -Wl,-rpath,@loader_path"

   if [ $use_asan -eq 0 ]; then 
      ASAN_OPTIONS=detect_leaks=0:detect_stack_use_after_return=0:check_initialization_order=0:alloc_dealloc_mismatch=0:strict_init_order=0:strict_string_checks=0:detect_invalid_pointer_pairs=0
   else 
      ASAN_OPTIONS=detect_leaks=1:detect_stack_use_after_return=1:check_initialization_order=1:alloc_dealloc_mismatch=1:strict_init_order=1:strict_string_checks=1:detect_invalid_pointer_pairs=2
   fi

   if [ $debug_64 -eq 1 ]; then
      ########################## macOS debug|x64

      if [ $existing_debug_64 -eq 0 ]; then 
         # TODO: add options to build.sh to only build debug x64 (not arm!)
         /bin/sh "$root_path_unsanitized/build.sh" -no_release_64 -analysis
         ret_code=$?
         if [ $ret_code -ne 0 ]; then 
            # build.sh should write out any messages
            exit $ret_code
         fi
      fi

      printf "%s\n" "Compiling ebm_native_test with $c_compiler/$cpp_compiler for macOS debug|x64"
      obj_path_unsanitized="$tmp_path_unsanitized/clang/obj/debug/mac/x64/ebm_native_test"
      bin_path_unsanitized="$tmp_path_unsanitized/clang/bin/debug/mac/x64/ebm_native_test"
      lib_file_body="_ebm_native_mac_x64_debug"
      g_log_file_unsanitized="$obj_path_unsanitized/ebm_native_test_debug_mac_x64_build_log.txt"
      both_args_extra="-march=core2 -target x86_64-apple-macos10.12 -m64 -O1 -fno-optimize-sibling-calls -fno-omit-frame-pointer"
      c_args_specific="$c_args $both_args $both_args_extra"
      cpp_args_specific="$cpp_args $both_args $both_args_extra"
      # the linker wants to have the most dependent .o/.so/.dylib files listed FIRST
      link_args_specific="-l$lib_file_body $link_args $cpp_args_specific"
   
      g_all_object_files_sanitized=""
      g_compile_out_full=""

      make_initial_paths_simple "$obj_path_unsanitized" "$bin_path_unsanitized"
      compile_directory_c "$c_compiler" "$c_args_specific" "$src_path_unsanitized" "$obj_path_unsanitized" "$is_asm" "test"
      compile_directory_cpp "$cpp_compiler" "$cpp_args_specific" "$src_path_unsanitized" "$obj_path_unsanitized" "$is_asm" "test"
      link_file "$cpp_compiler" "$link_args_specific" "$bin_path_unsanitized" "$bin_file"
      printf "%s\n" "$g_compile_out_full"
      printf "%s\n" "$g_compile_out_full" > "$g_log_file_unsanitized"

      cp "$staging_path_unsanitized/lib$lib_file_body.dylib" "$bin_path_unsanitized/"
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
      "$bin_path_unsanitized/$bin_file"
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
   fi

   if [ $release_64 -eq 1 ]; then
      ########################## macOS release|x64

      if [ $existing_release_64 -eq 0 ]; then 
         # TODO: add options to build.sh to only build release x64 (not arm!)
         /bin/sh "$root_path_unsanitized/build.sh" -no_debug_64 -analysis
         ret_code=$?
         if [ $ret_code -ne 0 ]; then 
            # build.sh should write out any messages
            exit $ret_code
         fi
      fi

      printf "%s\n" "Compiling ebm_native_test with $c_compiler/$cpp_compiler for macOS release|x64"
      obj_path_unsanitized="$tmp_path_unsanitized/clang/obj/release/mac/x64/ebm_native_test"
      bin_path_unsanitized="$tmp_path_unsanitized/clang/bin/release/mac/x64/ebm_native_test"
      lib_file_body="_ebm_native_mac_x64"
      g_log_file_unsanitized="$obj_path_unsanitized/ebm_native_test_release_mac_x64_build_log.txt"
      both_args_extra="-march=core2 -target x86_64-apple-macos10.12 -m64 -DNDEBUG -O1 -fno-optimize-sibling-calls -fno-omit-frame-pointer"
      c_args_specific="$c_args $both_args $both_args_extra"
      cpp_args_specific="$cpp_args $both_args $both_args_extra"
      # the linker wants to have the most dependent .o/.so/.dylib files listed FIRST
      link_args_specific="-l$lib_file_body $link_args $cpp_args_specific"
   
      g_all_object_files_sanitized=""
      g_compile_out_full=""

      make_initial_paths_simple "$obj_path_unsanitized" "$bin_path_unsanitized"
      compile_directory_c "$c_compiler" "$c_args_specific" "$src_path_unsanitized" "$obj_path_unsanitized" "$is_asm" "test"
      compile_directory_cpp "$cpp_compiler" "$cpp_args_specific" "$src_path_unsanitized" "$obj_path_unsanitized" "$is_asm" "test"
      link_file "$cpp_compiler" "$link_args_specific" "$bin_path_unsanitized" "$bin_file"
      printf "%s\n" "$g_compile_out_full"
      printf "%s\n" "$g_compile_out_full" > "$g_log_file_unsanitized"

      cp "$staging_path_unsanitized/lib$lib_file_body.dylib" "$bin_path_unsanitized/"
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
      "$bin_path_unsanitized/$bin_file"
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
   fi

   if [ $debug_arm -eq 1 ]; then
      ########################## macOS debug|arm

      if [ $existing_debug_arm -eq 0 ]; then 
         # TODO: add options to build.sh to only build release arm
         /bin/sh "$root_path_unsanitized/build.sh" -analysis
         ret_code=$?
         if [ $ret_code -ne 0 ]; then 
            # build.sh should write out any messages
            exit $ret_code
         fi
      fi

      printf "%s\n" "Compiling ebm_native_test with $c_compiler/$cpp_compiler for macOS debug|arm"
      obj_path_unsanitized="$tmp_path_unsanitized/clang/obj/debug/mac/arm/ebm_native_test"
      bin_path_unsanitized="$tmp_path_unsanitized/clang/bin/debug/mac/arm/ebm_native_test"
      lib_file_body="_ebm_native_mac_arm_debug"
      g_log_file_unsanitized="$obj_path_unsanitized/ebm_native_test_debug_mac_arm_build_log.txt"
      both_args_extra="-target arm64-apple-macos11 -m64 -O1 -fno-optimize-sibling-calls -fno-omit-frame-pointer"
      c_args_specific="$c_args $both_args $both_args_extra"
      cpp_args_specific="$cpp_args $both_args $both_args_extra"
      # the linker wants to have the most dependent .o/.so/.dylib files listed FIRST
      link_args_specific="-l$lib_file_body $link_args $cpp_args_specific"
   
      g_all_object_files_sanitized=""
      g_compile_out_full=""

      make_initial_paths_simple "$obj_path_unsanitized" "$bin_path_unsanitized"
      compile_directory_c "$c_compiler" "$c_args_specific" "$src_path_unsanitized" "$obj_path_unsanitized" "$is_asm" "test"
      compile_directory_cpp "$cpp_compiler" "$cpp_args_specific" "$src_path_unsanitized" "$obj_path_unsanitized" "$is_asm" "test"
      link_file "$cpp_compiler" "$link_args_specific" "$bin_path_unsanitized" "$bin_file"
      printf "%s\n" "$g_compile_out_full"
      printf "%s\n" "$g_compile_out_full" > "$g_log_file_unsanitized"

      cp "$staging_path_unsanitized/lib$lib_file_body.dylib" "$bin_path_unsanitized/"
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
      "$bin_path_unsanitized/$bin_file"
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
   fi

   if [ $release_arm -eq 1 ]; then
      ########################## macOS release|arm

      if [ $existing_release_arm -eq 0 ]; then 
         # TODO: add options to build.sh to only build release arm
         /bin/sh "$root_path_unsanitized/build.sh" -analysis
         ret_code=$?
         if [ $ret_code -ne 0 ]; then 
            # build.sh should write out any messages
            exit $ret_code
         fi
      fi

      printf "%s\n" "Compiling ebm_native_test with $c_compiler/$cpp_compiler for macOS release|arm"
      obj_path_unsanitized="$tmp_path_unsanitized/clang/obj/release/mac/arm/ebm_native_test"
      bin_path_unsanitized="$tmp_path_unsanitized/clang/bin/release/mac/arm/ebm_native_test"
      lib_file_body="_ebm_native_mac_arm"
      g_log_file_unsanitized="$obj_path_unsanitized/ebm_native_test_release_mac_arm_build_log.txt"
      both_args_extra="-target arm64-apple-macos11 -m64 -DNDEBUG -O1 -fno-optimize-sibling-calls -fno-omit-frame-pointer"
      c_args_specific="$c_args $both_args $both_args_extra"
      cpp_args_specific="$cpp_args $both_args $both_args_extra"
      # the linker wants to have the most dependent .o/.so/.dylib files listed FIRST
      link_args_specific="-l$lib_file_body $link_args $cpp_args_specific"
   
      g_all_object_files_sanitized=""
      g_compile_out_full=""

      make_initial_paths_simple "$obj_path_unsanitized" "$bin_path_unsanitized"
      compile_directory_c "$c_compiler" "$c_args_specific" "$src_path_unsanitized" "$obj_path_unsanitized" "$is_asm" "test"
      compile_directory_cpp "$cpp_compiler" "$cpp_args_specific" "$src_path_unsanitized" "$obj_path_unsanitized" "$is_asm" "test"
      link_file "$cpp_compiler" "$link_args_specific" "$bin_path_unsanitized" "$bin_file"
      printf "%s\n" "$g_compile_out_full"
      printf "%s\n" "$g_compile_out_full" > "$g_log_file_unsanitized"

      cp "$staging_path_unsanitized/lib$lib_file_body.dylib" "$bin_path_unsanitized/"
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
      "$bin_path_unsanitized/$bin_file"
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
   fi

else
   printf "%s\n" "OS $os_type not recognized.  We support clang/clang++ on macOS and gcc/g++ on Linux"
   exit 1
fi

echo Passed All Tests
