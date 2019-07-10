#!/bin/sh

clang_pp_bin=clang++
g_pp_bin=g++
os_type=`uname`
script_path=`dirname "$0"`

build_32_bit=0
for arg in "$@"; do
   if [ "$arg" = "-32bit" ]; then
      build_32_bit=1
   fi
done

all_compiles="\"$script_path/src/core/ebmcore/DataSetByAttribute.cpp\" \"$script_path/src/core/ebmcore/DataSetByAttributeCombination.cpp\" \"$script_path/src/core/ebmcore/InteractionDetection.cpp\" \"$script_path/src/core/ebmcore/Logging.cpp\" \"$script_path/src/core/ebmcore/SamplingWithReplacement.cpp\" \"$script_path/src/core/ebmcore/Training.cpp\" -I\"$script_path/src/core/ebmcore\" -I\"$script_path/src/core/inc\" -std=c++11 -fpermissive -fvisibility=hidden -fvisibility-inlines-hidden -O3 -march=core2 -DEBMCORE_EXPORTS -fpic"

if [ "$os_type" = "Darwin" ]; then
   # reference on rpath & install_name: https://www.mikeash.com/pyblog/friday-qa-2009-11-06-linking-and-install-names.html

   all_mac="$all_compiles -dynamiclib"

   echo "Creating initial directories"
   [ -d "$script_path/staging" ] || mkdir -p "$script_path/staging"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   [ -d "$script_path/src/python/interpret/lib" ] || mkdir -p "$script_path/src/python/interpret/lib"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi

   echo "Compiling with $clang_pp_bin for macOS release|x64"
   [ -d "$script_path/tmp/clang/intermediate/release/mac/x64/ebmcore" ] || mkdir -p "$script_path/tmp/clang/intermediate/release/mac/x64/ebmcore"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   [ -d "$script_path/tmp/clang/bin/release/mac/x64/ebmcore" ] || mkdir -p "$script_path/tmp/clang/bin/release/mac/x64/ebmcore"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   compile_out=`eval $clang_pp_bin $all_mac -m64 -DNDEBUG -install_name @rpath/ebmcore_mac_x64.dylib -o \"$script_path/tmp/clang/bin/release/mac/x64/ebmcore/ebmcore_mac_x64.dylib\" 2>&1`
   ret_code=$?
   echo -n "$compile_out"
   echo -n "$compile_out" > "$script_path/tmp/clang/intermediate/release/mac/x64/ebmcore/ebmcore_release_mac_x64_build_log.txt"
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   cp "$script_path/tmp/clang/bin/release/mac/x64/ebmcore/ebmcore_mac_x64.dylib" "$script_path/src/python/interpret/lib/"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   cp "$script_path/tmp/clang/bin/release/mac/x64/ebmcore/ebmcore_mac_x64.dylib" "$script_path/staging/"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi

   echo "Compiling with $clang_pp_bin for macOS debug|x64"
   [ -d "$script_path/tmp/clang/intermediate/debug/mac/x64/ebmcore" ] || mkdir -p "$script_path/tmp/clang/intermediate/debug/mac/x64/ebmcore"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   [ -d "$script_path/tmp/clang/bin/debug/mac/x64/ebmcore" ] || mkdir -p "$script_path/tmp/clang/bin/debug/mac/x64/ebmcore"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   compile_out=`eval $clang_pp_bin $all_mac -m64 -install_name @rpath/ebmcore_mac_x64_debug.dylib -o \"$script_path/tmp/clang/bin/debug/mac/x64/ebmcore/ebmcore_mac_x64_debug.dylib\" 2>&1`
   ret_code=$?
   echo -n "$compile_out"
   echo -n "$compile_out" > "$script_path/tmp/clang/intermediate/debug/mac/x64/ebmcore/ebmcore_debug_mac_x64_build_log.txt"
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   cp "$script_path/tmp/clang/bin/debug/mac/x64/ebmcore/ebmcore_mac_x64_debug.dylib" "$script_path/src/python/interpret/lib/"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   cp "$script_path/tmp/clang/bin/debug/mac/x64/ebmcore/ebmcore_mac_x64_debug.dylib" "$script_path/staging/"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi

   if [ $build_32_bit -eq 1 ]; then
      echo "Compiling with $clang_pp_bin for macOS release|x86"
      [ -d "$script_path/tmp/clang/intermediate/release/mac/x86/ebmcore" ] || mkdir -p "$script_path/tmp/clang/intermediate/release/mac/x86/ebmcore"
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
      [ -d "$script_path/tmp/clang/bin/release/mac/x86/ebmcore" ] || mkdir -p "$script_path/tmp/clang/bin/release/mac/x86/ebmcore"
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
      compile_out=`eval $clang_pp_bin $all_mac -m32 -DNDEBUG -install_name @rpath/ebmcore_mac_x86.dylib -o \"$script_path/tmp/clang/bin/release/mac/x86/ebmcore/ebmcore_mac_x86.dylib\" 2>&1`
      ret_code=$?
      echo -n "$compile_out"
      echo -n "$compile_out" > "$script_path/tmp/clang/intermediate/release/mac/x86/ebmcore/ebmcore_release_mac_x86_build_log.txt"
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
      cp "$script_path/tmp/clang/bin/release/mac/x86/ebmcore/ebmcore_mac_x86.dylib" "$script_path/src/python/interpret/lib/"
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
      cp "$script_path/tmp/clang/bin/release/mac/x86/ebmcore/ebmcore_mac_x86.dylib" "$script_path/staging/"
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi

      echo "Compiling with $clang_pp_bin for macOS debug|x86"
      [ -d "$script_path/tmp/clang/intermediate/debug/mac/x86/ebmcore" ] || mkdir -p "$script_path/tmp/clang/intermediate/debug/mac/x86/ebmcore"
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
      [ -d "$script_path/tmp/clang/bin/debug/mac/x86/ebmcore" ] || mkdir -p "$script_path/tmp/clang/bin/debug/mac/x86/ebmcore"
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
      compile_out=`eval $clang_pp_bin $all_mac -m32 -install_name @rpath/ebmcore_mac_x86_debug.dylib -o \"$script_path/tmp/clang/bin/debug/mac/x86/ebmcore/ebmcore_mac_x86_debug.dylib\" 2>&1`
      ret_code=$?
      echo -n "$compile_out"
      echo -n "$compile_out" > "$script_path/tmp/clang/intermediate/debug/mac/x86/ebmcore/ebmcore_debug_mac_x86_build_log.txt"
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
      cp "$script_path/tmp/clang/bin/debug/mac/x86/ebmcore/ebmcore_mac_x86_debug.dylib" "$script_path/src/python/interpret/lib/"
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
      cp "$script_path/tmp/clang/bin/debug/mac/x86/ebmcore/ebmcore_mac_x86_debug.dylib" "$script_path/staging/"
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
   fi
elif [ "$os_type" = "Linux" ]; then

   # to cross compile for different architectures x86/x64, run the following command: sudo apt-get install g++-multilib

   all_linux="$all_compiles -Wl,--version-script=\"$script_path/src/core/ebmcore/EbmCoreExports.txt\" -Wl,--exclude-libs,ALL -Wl,--wrap=memcpy \"$script_path/src/core/ebmcore/WrapFunc.cpp\" -static-libgcc -static-libstdc++ -shared"

   echo "Creating initial directories"
   [ -d "$script_path/staging" ] || mkdir -p "$script_path/staging"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   [ -d "$script_path/src/python/interpret/lib" ] || mkdir -p "$script_path/src/python/interpret/lib"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi

   echo "Compiling with $g_pp_bin for Linux release|x64"
   [ -d "$script_path/tmp/gcc/intermediate/release/linux/x64/ebmcore" ] || mkdir -p "$script_path/tmp/gcc/intermediate/release/linux/x64/ebmcore"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   [ -d "$script_path/tmp/gcc/bin/release/linux/x64/ebmcore" ] || mkdir -p "$script_path/tmp/gcc/bin/release/linux/x64/ebmcore"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   compile_out=`eval $g_pp_bin $all_linux -m64 -DNDEBUG -o \"$script_path/tmp/gcc/bin/release/linux/x64/ebmcore/ebmcore_linux_x64.so\" 2>&1`
   ret_code=$?
   echo -n "$compile_out"
   echo -n "$compile_out" > "$script_path/tmp/gcc/intermediate/release/linux/x64/ebmcore/ebmcore_release_linux_x64_build_log.txt"
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   cp "$script_path/tmp/gcc/bin/release/linux/x64/ebmcore/ebmcore_linux_x64.so" "$script_path/src/python/interpret/lib/"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   cp "$script_path/tmp/gcc/bin/release/linux/x64/ebmcore/ebmcore_linux_x64.so" "$script_path/staging/"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi

   echo "Compiling with $g_pp_bin for Linux debug|x64"
   [ -d "$script_path/tmp/gcc/intermediate/debug/linux/x64/ebmcore" ] || mkdir -p "$script_path/tmp/gcc/intermediate/debug/linux/x64/ebmcore"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   [ -d "$script_path/tmp/gcc/bin/debug/linux/x64/ebmcore" ] || mkdir -p "$script_path/tmp/gcc/bin/debug/linux/x64/ebmcore"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   compile_out=`eval $g_pp_bin $all_linux -m64 -o \"$script_path/tmp/gcc/bin/debug/linux/x64/ebmcore/ebmcore_linux_x64_debug.so\" 2>&1`
   ret_code=$?
   echo -n "$compile_out"
   echo -n "$compile_out" > "$script_path/tmp/gcc/intermediate/debug/linux/x64/ebmcore/ebmcore_debug_linux_x64_build_log.txt"
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   cp "$script_path/tmp/gcc/bin/debug/linux/x64/ebmcore/ebmcore_linux_x64_debug.so" "$script_path/src/python/interpret/lib/"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   cp "$script_path/tmp/gcc/bin/debug/linux/x64/ebmcore/ebmcore_linux_x64_debug.so" "$script_path/staging/"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi

   if [ $build_32_bit -eq 1 ]; then
      echo "Compiling with $g_pp_bin for Linux release|x86"
      [ -d "$script_path/tmp/gcc/intermediate/release/linux/x86/ebmcore" ] || mkdir -p "$script_path/tmp/gcc/intermediate/release/linux/x86/ebmcore"
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
      [ -d "$script_path/tmp/gcc/bin/release/linux/x86/ebmcore" ] || mkdir -p "$script_path/tmp/gcc/bin/release/linux/x86/ebmcore"
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
      compile_out=`eval $g_pp_bin $all_linux -m32 -DNDEBUG -o \"$script_path/tmp/gcc/bin/release/linux/x86/ebmcore/ebmcore_linux_x86.so\" 2>&1`
      ret_code=$?
      echo -n "$compile_out"
      echo -n "$compile_out" > "$script_path/tmp/gcc/intermediate/release/linux/x86/ebmcore/ebmcore_release_linux_x86_build_log.txt"
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
      cp "$script_path/tmp/gcc/bin/release/linux/x86/ebmcore/ebmcore_linux_x86.so" "$script_path/src/python/interpret/lib/"
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
      cp "$script_path/tmp/gcc/bin/release/linux/x86/ebmcore/ebmcore_linux_x86.so" "$script_path/staging/"
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi

      echo "Compiling with $g_pp_bin for Linux debug|x86"
      [ -d "$script_path/tmp/gcc/intermediate/debug/linux/x86/ebmcore" ] || mkdir -p "$script_path/tmp/gcc/intermediate/debug/linux/x86/ebmcore"
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
      [ -d "$script_path/tmp/gcc/bin/debug/linux/x86/ebmcore" ] || mkdir -p "$script_path/tmp/gcc/bin/debug/linux/x86/ebmcore"
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
      compile_out=`eval $g_pp_bin $all_linux -m32 -o \"$script_path/tmp/gcc/bin/debug/linux/x86/ebmcore/ebmcore_linux_x86_debug.so\" 2>&1`
      ret_code=$?
      echo -n "$compile_out"
      echo -n "$compile_out" > "$script_path/tmp/gcc/intermediate/debug/linux/x86/ebmcore/ebmcore_debug_linux_x86_build_log.txt"
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
      cp "$script_path/tmp/gcc/bin/debug/linux/x86/ebmcore/ebmcore_linux_x86_debug.so" "$script_path/src/python/interpret/lib/"
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
      cp "$script_path/tmp/gcc/bin/debug/linux/x86/ebmcore/ebmcore_linux_x86_debug.so" "$script_path/staging/"
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi
   fi
else
   echo "OS $os_type not recognized.  We support $clang_pp_bin on macOS and $g_pp_bin on Linux"
   exit 1
fi
