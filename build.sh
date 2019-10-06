#!/bin/sh

clang_pp_bin=clang++
g_pp_bin=g++
os_type=`uname`
root_path=`dirname "$0"`

build_32_bit=0
for arg in "$@"; do
   if [ "$arg" = "-32bit" ]; then
      build_32_bit=1
   fi
done

# re-enable these warnings when they are better supported by g++ or clang: -Wduplicated-cond -Wduplicated-branches -Wrestrict
compile_all="\"$root_path/core/DataSetByFeature.cpp\" \"$root_path/core/DataSetByFeatureCombination.cpp\" \"$root_path/core/InteractionDetection.cpp\" \"$root_path/core/Logging.cpp\" \"$root_path/core/SamplingWithReplacement.cpp\" \"$root_path/core/Training.cpp\" -I\"$root_path/core\" -I\"$root_path/core/inc\" -Wall -Wextra -Wno-parentheses -Wold-style-cast -Wdouble-promotion -Wshadow -Wformat=2 -std=c++11 -fvisibility=hidden -fvisibility-inlines-hidden -O3 -march=core2 -DEBMCORE_EXPORTS -fpic"

if [ "$os_type" = "Darwin" ]; then
   # reference on rpath & install_name: https://www.mikeash.com/pyblog/friday-qa-2009-11-06-linking-and-install-names.html

   # try moving some of these clang specific warnings into compile_all if g++ eventually supports them
   compile_mac="$compile_all -Wnull-dereference -Wgnu-zero-variadic-macro-arguments -dynamiclib"

   printf "%s\n" "Creating initial directories"
   [ -d "$root_path/staging" ] || mkdir -p "$root_path/staging"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   [ -d "$root_path/python/interpret-core/interpret/lib" ] || mkdir -p "$root_path/python/interpret-core/interpret/lib"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi

   printf "%s\n" "Compiling ebmcore with $clang_pp_bin for macOS release|x64"
   [ -d "$root_path/tmp/clang/intermediate/release/mac/x64/ebmcore" ] || mkdir -p "$root_path/tmp/clang/intermediate/release/mac/x64/ebmcore"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   [ -d "$root_path/tmp/clang/bin/release/mac/x64/ebmcore" ] || mkdir -p "$root_path/tmp/clang/bin/release/mac/x64/ebmcore"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   compile_command="$clang_pp_bin $compile_mac -m64 -DNDEBUG -install_name @rpath/lib_ebmcore_mac_x64.dylib -o \"$root_path/tmp/clang/bin/release/mac/x64/ebmcore/lib_ebmcore_mac_x64.dylib\" 2>&1"
   compile_out=`eval $compile_command`
   ret_code=$?
   printf "%s\n" "$compile_out"
   printf "%s\n" "$compile_out" > "$root_path/tmp/clang/intermediate/release/mac/x64/ebmcore/ebmcore_release_mac_x64_build_log.txt"
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   cp "$root_path/tmp/clang/bin/release/mac/x64/ebmcore/lib_ebmcore_mac_x64.dylib" "$root_path/python/interpret-core/interpret/lib/"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   cp "$root_path/tmp/clang/bin/release/mac/x64/ebmcore/lib_ebmcore_mac_x64.dylib" "$root_path/staging/"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi

   printf "%s\n" "Compiling ebmcore with $clang_pp_bin for macOS debug|x64"
   [ -d "$root_path/tmp/clang/intermediate/debug/mac/x64/ebmcore" ] || mkdir -p "$root_path/tmp/clang/intermediate/debug/mac/x64/ebmcore"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   [ -d "$root_path/tmp/clang/bin/debug/mac/x64/ebmcore" ] || mkdir -p "$root_path/tmp/clang/bin/debug/mac/x64/ebmcore"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   compile_command="$clang_pp_bin $compile_mac -m64 -install_name @rpath/lib_ebmcore_mac_x64_debug.dylib -o \"$root_path/tmp/clang/bin/debug/mac/x64/ebmcore/lib_ebmcore_mac_x64_debug.dylib\" 2>&1"
   compile_out=`eval $compile_command`
   ret_code=$?
   printf "%s\n" "$compile_out"
   printf "%s\n" "$compile_out" > "$root_path/tmp/clang/intermediate/debug/mac/x64/ebmcore/ebmcore_debug_mac_x64_build_log.txt"
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   cp "$root_path/tmp/clang/bin/debug/mac/x64/ebmcore/lib_ebmcore_mac_x64_debug.dylib" "$root_path/python/interpret-core/interpret/lib/"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   cp "$root_path/tmp/clang/bin/debug/mac/x64/ebmcore/lib_ebmcore_mac_x64_debug.dylib" "$root_path/staging/"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi

   if [ $build_32_bit -eq 1 ]; then
      printf "%s\n" "Compiling ebmcore with $clang_pp_bin for macOS release|x86"
      [ -d "$root_path/tmp/clang/intermediate/release/mac/x86/ebmcore" ] || mkdir -p "$root_path/tmp/clang/intermediate/release/mac/x86/ebmcore"
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
      [ -d "$root_path/tmp/clang/bin/release/mac/x86/ebmcore" ] || mkdir -p "$root_path/tmp/clang/bin/release/mac/x86/ebmcore"
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
      compile_command="$clang_pp_bin $compile_mac -m32 -DNDEBUG -install_name @rpath/lib_ebmcore_mac_x86.dylib -o \"$root_path/tmp/clang/bin/release/mac/x86/ebmcore/lib_ebmcore_mac_x86.dylib\" 2>&1"
      compile_out=`eval $compile_command`
      ret_code=$?
      printf "%s\n" "$compile_out"
      printf "%s\n" "$compile_out" > "$root_path/tmp/clang/intermediate/release/mac/x86/ebmcore/ebmcore_release_mac_x86_build_log.txt"
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
      cp "$root_path/tmp/clang/bin/release/mac/x86/ebmcore/lib_ebmcore_mac_x86.dylib" "$root_path/python/interpret-core/interpret/lib/"
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
      cp "$root_path/tmp/clang/bin/release/mac/x86/ebmcore/lib_ebmcore_mac_x86.dylib" "$root_path/staging/"
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi

      printf "%s\n" "Compiling ebmcore with $clang_pp_bin for macOS debug|x86"
      [ -d "$root_path/tmp/clang/intermediate/debug/mac/x86/ebmcore" ] || mkdir -p "$root_path/tmp/clang/intermediate/debug/mac/x86/ebmcore"
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
      [ -d "$root_path/tmp/clang/bin/debug/mac/x86/ebmcore" ] || mkdir -p "$root_path/tmp/clang/bin/debug/mac/x86/ebmcore"
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
      compile_command="$clang_pp_bin $compile_mac -m32 -install_name @rpath/lib_ebmcore_mac_x86_debug.dylib -o \"$root_path/tmp/clang/bin/debug/mac/x86/ebmcore/lib_ebmcore_mac_x86_debug.dylib\" 2>&1"
      compile_out=`eval $compile_command`
      ret_code=$?
      printf "%s\n" "$compile_out"
      printf "%s\n" "$compile_out" > "$root_path/tmp/clang/intermediate/debug/mac/x86/ebmcore/ebmcore_debug_mac_x86_build_log.txt"
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
      cp "$root_path/tmp/clang/bin/debug/mac/x86/ebmcore/lib_ebmcore_mac_x86_debug.dylib" "$root_path/python/interpret-core/interpret/lib/"
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
      cp "$root_path/tmp/clang/bin/debug/mac/x86/ebmcore/lib_ebmcore_mac_x86_debug.dylib" "$root_path/staging/"
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
   fi
elif [ "$os_type" = "Linux" ]; then

   # to cross compile for different architectures x86/x64, run the following command: sudo apt-get install g++-multilib

   # try moving some of these g++ specific warnings into compile_all if clang eventually supports them
   compile_linux="$compile_all -Wlogical-op -Wl,--version-script=\"$root_path/core/ebmcore_exports.txt\" -Wl,--exclude-libs,ALL -Wl,--wrap=memcpy \"$root_path/core/wrap_func.cpp\" -static-libgcc -static-libstdc++ -shared"

   printf "%s\n" "Creating initial directories"
   [ -d "$root_path/staging" ] || mkdir -p "$root_path/staging"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   [ -d "$root_path/python/interpret-core/interpret/lib" ] || mkdir -p "$root_path/python/interpret-core/interpret/lib"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi

   printf "%s\n" "Compiling ebmcore with $g_pp_bin for Linux release|x64"
   [ -d "$root_path/tmp/gcc/intermediate/release/linux/x64/ebmcore" ] || mkdir -p "$root_path/tmp/gcc/intermediate/release/linux/x64/ebmcore"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   [ -d "$root_path/tmp/gcc/bin/release/linux/x64/ebmcore" ] || mkdir -p "$root_path/tmp/gcc/bin/release/linux/x64/ebmcore"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   compile_command="$g_pp_bin $compile_linux -m64 -DNDEBUG -o \"$root_path/tmp/gcc/bin/release/linux/x64/ebmcore/lib_ebmcore_linux_x64.so\" 2>&1"
   compile_out=`eval $compile_command`
   ret_code=$?
   printf "%s\n" "$compile_out"
   printf "%s\n" "$compile_out" > "$root_path/tmp/gcc/intermediate/release/linux/x64/ebmcore/ebmcore_release_linux_x64_build_log.txt"
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   cp "$root_path/tmp/gcc/bin/release/linux/x64/ebmcore/lib_ebmcore_linux_x64.so" "$root_path/python/interpret-core/interpret/lib/"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   cp "$root_path/tmp/gcc/bin/release/linux/x64/ebmcore/lib_ebmcore_linux_x64.so" "$root_path/staging/"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi

   printf "%s\n" "Compiling ebmcore with $g_pp_bin for Linux debug|x64"
   [ -d "$root_path/tmp/gcc/intermediate/debug/linux/x64/ebmcore" ] || mkdir -p "$root_path/tmp/gcc/intermediate/debug/linux/x64/ebmcore"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   [ -d "$root_path/tmp/gcc/bin/debug/linux/x64/ebmcore" ] || mkdir -p "$root_path/tmp/gcc/bin/debug/linux/x64/ebmcore"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   compile_command="$g_pp_bin $compile_linux -m64 -o \"$root_path/tmp/gcc/bin/debug/linux/x64/ebmcore/lib_ebmcore_linux_x64_debug.so\" 2>&1"
   compile_out=`eval $compile_command`
   ret_code=$?
   printf "%s\n" "$compile_out"
   printf "%s\n" "$compile_out" > "$root_path/tmp/gcc/intermediate/debug/linux/x64/ebmcore/ebmcore_debug_linux_x64_build_log.txt"
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   cp "$root_path/tmp/gcc/bin/debug/linux/x64/ebmcore/lib_ebmcore_linux_x64_debug.so" "$root_path/python/interpret-core/interpret/lib/"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   cp "$root_path/tmp/gcc/bin/debug/linux/x64/ebmcore/lib_ebmcore_linux_x64_debug.so" "$root_path/staging/"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi

   if [ $build_32_bit -eq 1 ]; then
      printf "%s\n" "Compiling ebmcore with $g_pp_bin for Linux release|x86"
      [ -d "$root_path/tmp/gcc/intermediate/release/linux/x86/ebmcore" ] || mkdir -p "$root_path/tmp/gcc/intermediate/release/linux/x86/ebmcore"
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
      [ -d "$root_path/tmp/gcc/bin/release/linux/x86/ebmcore" ] || mkdir -p "$root_path/tmp/gcc/bin/release/linux/x86/ebmcore"
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
      compile_command="$g_pp_bin $compile_linux -m32 -DNDEBUG -o \"$root_path/tmp/gcc/bin/release/linux/x86/ebmcore/lib_ebmcore_linux_x86.so\" 2>&1"
      compile_out=`eval $compile_command`
      ret_code=$?
      printf "%s\n" "$compile_out"
      printf "%s\n" "$compile_out" > "$root_path/tmp/gcc/intermediate/release/linux/x86/ebmcore/ebmcore_release_linux_x86_build_log.txt"
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
      cp "$root_path/tmp/gcc/bin/release/linux/x86/ebmcore/lib_ebmcore_linux_x86.so" "$root_path/python/interpret-core/interpret/lib/"
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
      cp "$root_path/tmp/gcc/bin/release/linux/x86/ebmcore/lib_ebmcore_linux_x86.so" "$root_path/staging/"
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi

      printf "%s\n" "Compiling ebmcore with $g_pp_bin for Linux debug|x86"
      [ -d "$root_path/tmp/gcc/intermediate/debug/linux/x86/ebmcore" ] || mkdir -p "$root_path/tmp/gcc/intermediate/debug/linux/x86/ebmcore"
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
      [ -d "$root_path/tmp/gcc/bin/debug/linux/x86/ebmcore" ] || mkdir -p "$root_path/tmp/gcc/bin/debug/linux/x86/ebmcore"
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
      compile_command="$g_pp_bin $compile_linux -m32 -o \"$root_path/tmp/gcc/bin/debug/linux/x86/ebmcore/lib_ebmcore_linux_x86_debug.so\" 2>&1"
      compile_out=`eval $compile_command`
      ret_code=$?
      printf "%s\n" "$compile_out"
      printf "%s\n" "$compile_out" > "$root_path/tmp/gcc/intermediate/debug/linux/x86/ebmcore/ebmcore_debug_linux_x86_build_log.txt"
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
      cp "$root_path/tmp/gcc/bin/debug/linux/x86/ebmcore/lib_ebmcore_linux_x86_debug.so" "$root_path/python/interpret-core/interpret/lib/"
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
      cp "$root_path/tmp/gcc/bin/debug/linux/x86/ebmcore/lib_ebmcore_linux_x86_debug.so" "$root_path/staging/"
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
   fi
else
   printf "%s\n" "OS $os_type not recognized.  We support $clang_pp_bin on macOS and $g_pp_bin on Linux"
   exit 1
fi
