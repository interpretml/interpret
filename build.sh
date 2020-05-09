#!/bin/sh

# TODO also build our html resources here, and also in the .bat file for Windows

clang_pp_bin=clang++
g_pp_bin=g++
os_type=`uname`
root_path=`dirname "$0"`
src_path="$root_path/shared/ebm_native"
python_lib="$root_path/python/interpret-core/interpret/lib"
staging_path="$root_path/staging"

build_32_bit=0
for arg in "$@"; do
   if [ "$arg" = "-32bit" ]; then
      build_32_bit=1
   fi
done

# re-enable these warnings when they are better supported by g++ or clang: -Wduplicated-cond -Wduplicated-branches -Wrestrict
compile_all=""
compile_all="$compile_all \"$src_path/DataSetByFeature.cpp\""
compile_all="$compile_all \"$src_path/DataSetByFeatureCombination.cpp\""
compile_all="$compile_all \"$src_path/InteractionDetection.cpp\""
compile_all="$compile_all \"$src_path/Logging.cpp\""
compile_all="$compile_all \"$src_path/SamplingWithReplacement.cpp\""
compile_all="$compile_all \"$src_path/Boosting.cpp\""
compile_all="$compile_all \"$src_path/Discretization.cpp\""
compile_all="$compile_all -I\"$src_path\""
compile_all="$compile_all -I\"$src_path/inc\""
compile_all="$compile_all -Wall -Wextra -Wno-parentheses -Wold-style-cast -Wdouble-promotion -Wshadow -Wformat=2 -std=c++11"
compile_all="$compile_all -fvisibility=hidden -fvisibility-inlines-hidden -O3 -ffast-math -fno-finite-math-only -march=core2 -DEBM_NATIVE_EXPORTS -fpic"

if [ "$os_type" = "Darwin" ]; then
   # reference on rpath & install_name: https://www.mikeash.com/pyblog/friday-qa-2009-11-06-linking-and-install-names.html

   # try moving some of these clang specific warnings into compile_all if g++ eventually supports them
   compile_mac="$compile_all -Wnull-dereference -Wgnu-zero-variadic-macro-arguments -dynamiclib"

   printf "%s\n" "Creating initial directories"
   [ -d "$staging_path" ] || mkdir -p "$staging_path"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   [ -d "$python_lib" ] || mkdir -p "$python_lib"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi

   printf "%s\n" "Compiling ebm_native with $clang_pp_bin for macOS release|x64"
   intermediate_path="$root_path/tmp/clang/intermediate/release/mac/x64/ebm_native"
   bin_path="$root_path/tmp/clang/bin/release/mac/x64/ebm_native"
   bin_file="lib_ebm_native_mac_x64.dylib"
   log_file="$intermediate_path/ebm_native_release_mac_x64_build_log.txt"
   compile_command="$clang_pp_bin $compile_mac -m64 -DNDEBUG -install_name @rpath/$bin_file -o \"$bin_path/$bin_file\" 2>&1"
   
   [ -d "$intermediate_path" ] || mkdir -p "$intermediate_path"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   [ -d "$bin_path" ] || mkdir -p "$bin_path"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   compile_out=`eval $compile_command`
   ret_code=$?
   printf "%s\n" "$compile_out"
   printf "%s\n" "$compile_out" > "$log_file"
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   cp "$bin_path/$bin_file" "$python_lib/"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   cp "$bin_path/$bin_file" "$staging_path/"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi

   printf "%s\n" "Compiling ebm_native with $clang_pp_bin for macOS debug|x64"
   intermediate_path="$root_path/tmp/clang/intermediate/debug/mac/x64/ebm_native"
   bin_path="$root_path/tmp/clang/bin/debug/mac/x64/ebm_native"
   bin_file="lib_ebm_native_mac_x64_debug.dylib"
   log_file="$intermediate_path/ebm_native_debug_mac_x64_build_log.txt"
   compile_command="$clang_pp_bin $compile_mac -m64 -install_name @rpath/$bin_file -o \"$bin_path/$bin_file\" 2>&1"
   
   [ -d "$intermediate_path" ] || mkdir -p "$intermediate_path"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   [ -d "$bin_path" ] || mkdir -p "$bin_path"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   compile_out=`eval $compile_command`
   ret_code=$?
   printf "%s\n" "$compile_out"
   printf "%s\n" "$compile_out" > "$log_file"
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   cp "$bin_path/$bin_file" "$python_lib/"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   cp "$bin_path/$bin_file" "$staging_path/"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
elif [ "$os_type" = "Linux" ]; then

   # try moving some of these g++ specific warnings into compile_all if clang eventually supports them
   compile_linux="$compile_all"
   compile_linux="$compile_linux -Wlogical-op -Wl,--version-script=\"$src_path/ebm_native_exports.txt\" -Wl,--exclude-libs,ALL -Wl,-z,relro,-z,now"
   compile_linux="$compile_linux -Wl,--wrap=memcpy \"$src_path/wrap_func.cpp\" -static-libgcc -static-libstdc++ -shared"

   printf "%s\n" "Creating initial directories"
   [ -d "$staging_path" ] || mkdir -p "$staging_path"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   [ -d "$python_lib" ] || mkdir -p "$python_lib"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi

   printf "%s\n" "Compiling ebm_native with $g_pp_bin for Linux release|x64"
   intermediate_path="$root_path/tmp/gcc/intermediate/release/linux/x64/ebm_native"
   bin_path="$root_path/tmp/gcc/bin/release/linux/x64/ebm_native"
   bin_file="lib_ebm_native_linux_x64.so"
   log_file="$intermediate_path/ebm_native_release_linux_x64_build_log.txt"
   compile_command="$g_pp_bin $compile_linux -m64 -DNDEBUG -o \"$bin_path/$bin_file\" 2>&1"
   
   [ -d "$intermediate_path" ] || mkdir -p "$intermediate_path"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   [ -d "$bin_path" ] || mkdir -p "$bin_path"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   compile_out=`eval $compile_command`
   ret_code=$?
   printf "%s\n" "$compile_out"
   printf "%s\n" "$compile_out" > "$log_file"
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   cp "$bin_path/$bin_file" "$python_lib/"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   cp "$bin_path/$bin_file" "$staging_path/"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi

   printf "%s\n" "Compiling ebm_native with $g_pp_bin for Linux debug|x64"
   intermediate_path="$root_path/tmp/gcc/intermediate/debug/linux/x64/ebm_native"
   bin_path="$root_path/tmp/gcc/bin/debug/linux/x64/ebm_native"
   bin_file="lib_ebm_native_linux_x64_debug.so"
   log_file="$intermediate_path/ebm_native_debug_linux_x64_build_log.txt"
   compile_command="$g_pp_bin $compile_linux -m64 -o \"$bin_path/$bin_file\" 2>&1"
   
   [ -d "$intermediate_path" ] || mkdir -p "$intermediate_path"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   [ -d "$bin_path" ] || mkdir -p "$bin_path"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   compile_out=`eval $compile_command`
   ret_code=$?
   printf "%s\n" "$compile_out"
   printf "%s\n" "$compile_out" > "$log_file"
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   cp "$bin_path/$bin_file" "$python_lib/"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   cp "$bin_path/$bin_file" "$staging_path/"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi

   if [ $build_32_bit -eq 1 ]; then
      printf "%s\n" "Compiling ebm_native with $g_pp_bin for Linux release|x86"
      intermediate_path="$root_path/tmp/gcc/intermediate/release/linux/x86/ebm_native"
      bin_path="$root_path/tmp/gcc/bin/release/linux/x86/ebm_native"
      bin_file="lib_ebm_native_linux_x86.so"
      log_file="$intermediate_path/ebm_native_release_linux_x86_build_log.txt"
      compile_command="$g_pp_bin $compile_linux -m32 -DNDEBUG -o \"$bin_path/$bin_file\" 2>&1"
      
      if [ -d "$intermediate_path" ]; then
         printf "%s\n" "Doing first time installation of x86"

         # this is the first time we're being compiled x86 on this machine, so install other required items

         # TODO consider NOT running sudo inside this script and move that requirement to the caller
         #      per https://askubuntu.com/questions/425754/how-do-i-run-a-sudo-command-inside-a-script

         sudo apt-get -y install g++-multilib
         ret_code=$?
         if [ $ret_code -ne 0 ]; then 
            exit $ret_code
         fi

         mkdir -p "$intermediate_path"
      fi
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
      [ -d "$bin_path" ] || mkdir -p "$bin_path"
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
      compile_out=`eval $compile_command`
      ret_code=$?
      printf "%s\n" "$compile_out"
      printf "%s\n" "$compile_out" > "$log_file"
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
      cp "$bin_path/$bin_file" "$python_lib/"
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
      cp "$bin_path/$bin_file" "$staging_path/"
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi

      printf "%s\n" "Compiling ebm_native with $g_pp_bin for Linux debug|x86"
      intermediate_path="$root_path/tmp/gcc/intermediate/debug/linux/x86/ebm_native"
      bin_path="$root_path/tmp/gcc/bin/debug/linux/x86/ebm_native"
      bin_file="lib_ebm_native_linux_x86_debug.so"
      log_file="$intermediate_path/ebm_native_debug_linux_x86_build_log.txt"
      compile_command="$g_pp_bin $compile_linux -m32 -o \"$bin_path/$bin_file\" 2>&1"
      
      [ -d "$intermediate_path" ] || mkdir -p "$intermediate_path"
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
      [ -d "$bin_path" ] || mkdir -p "$bin_path"
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
      compile_out=`eval $compile_command`
      ret_code=$?
      printf "%s\n" "$compile_out"
      printf "%s\n" "$compile_out" > "$log_file"
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
      cp "$bin_path/$bin_file" "$python_lib/"
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
      cp "$bin_path/$bin_file" "$staging_path/"
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
   fi
else
   printf "%s\n" "OS $os_type not recognized.  We support $clang_pp_bin on macOS and $g_pp_bin on Linux"
   exit 1
fi
