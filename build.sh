#!/bin/sh

clang_pp_bin=clang++
g_pp_bin=g++
os_type=`uname`
script_path="`dirname \"$0\"`"

if [ "$os_type" = "Darwin" ]; then

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

   echo "Compiling with $clang_pp_bin for $os_type release|x64"
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
   compile_out=`$clang_pp_bin "$script_path/src/core/ebmcore/AttributeSet.cpp" "$script_path/src/core/ebmcore/DataSetByAttribute.cpp" "$script_path/src/core/ebmcore/DataSetByAttributeCombination.cpp" "$script_path/src/core/ebmcore/InteractionDetection.cpp" "$script_path/src/core/ebmcore/SamplingWithReplacement.cpp" "$script_path/src/core/ebmcore/Training.cpp" -I"$script_path/src/core/ebmcore" -I"$script_path/src/core/inc" -std=c++11 -fpermissive -fvisibility=hidden -O3 -march=core2 -static-libstdc++ -DEBMCORE_EXPORTS -fPIC -o "$script_path/tmp/clang/bin/release/mac/x64/ebmcore/ebmcore_mac_x64.dylib" -dynamiclib -install_name @rpath/ebmcore_mac_x64.dylib -m64 -DNDEBUG 2>&1`
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

   echo "Compiling with $clang_pp_bin for $os_type debug|x64"
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
   compile_out=`$clang_pp_bin "$script_path/src/core/ebmcore/AttributeSet.cpp" "$script_path/src/core/ebmcore/DataSetByAttribute.cpp" "$script_path/src/core/ebmcore/DataSetByAttributeCombination.cpp" "$script_path/src/core/ebmcore/InteractionDetection.cpp" "$script_path/src/core/ebmcore/SamplingWithReplacement.cpp" "$script_path/src/core/ebmcore/Training.cpp" -I"$script_path/src/core/ebmcore" -I"$script_path/src/core/inc" -std=c++11 -fpermissive -fvisibility=hidden -O3 -march=core2 -static-libstdc++ -DEBMCORE_EXPORTS -fPIC -o "$script_path/tmp/clang/bin/debug/mac/x64/ebmcore/ebmcore_mac_x64_debug.dylib" -dynamiclib -install_name @rpath/ebmcore_mac_x64_debug.dylib -m64 2>&1`
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

   # echo "Compiling with $clang_pp_bin for $os_type release|x86"
   # [ -d "$script_path/tmp/clang/intermediate/release/mac/x86/ebmcore" ] || mkdir -p "$script_path/tmp/clang/intermediate/release/mac/x86/ebmcore"
   # ret_code=$?
   # if [ $ret_code -ne 0 ]; then 
   #    exit $ret_code
   # fi
   # [ -d "$script_path/tmp/clang/bin/release/mac/x86/ebmcore" ] || mkdir -p "$script_path/tmp/clang/bin/release/mac/x86/ebmcore"
   # ret_code=$?
   # if [ $ret_code -ne 0 ]; then 
   #    exit $ret_code
   # fi
   # compile_out=`$clang_pp_bin "$script_path/src/core/ebmcore/AttributeSet.cpp" "$script_path/src/core/ebmcore/DataSetByAttribute.cpp" "$script_path/src/core/ebmcore/DataSetByAttributeCombination.cpp" "$script_path/src/core/ebmcore/InteractionDetection.cpp" "$script_path/src/core/ebmcore/SamplingWithReplacement.cpp" "$script_path/src/core/ebmcore/Training.cpp" -I"$script_path/src/core/ebmcore" -I"$script_path/src/core/inc" -std=c++11 -fpermissive -fvisibility=hidden -O3 -march=core2 -static-libstdc++ -DEBMCORE_EXPORTS -fPIC -o "$script_path/tmp/clang/bin/release/mac/x86/ebmcore/ebmcore_mac_x86.dylib" -dynamiclib -install_name @rpath/ebmcore_mac_x86.dylib -m32 -DNDEBUG 2>&1`
   # ret_code=$?
   # echo -n "$compile_out"
   # echo -n "$compile_out" > "$script_path/tmp/clang/intermediate/release/mac/x86/ebmcore/ebmcore_release_mac_x86_build_log.txt"
   # if [ $ret_code -ne 0 ]; then 
   #    exit $ret_code
   # fi
   # cp "$script_path/tmp/clang/bin/release/mac/x86/ebmcore/ebmcore_mac_x86.dylib" "$script_path/src/python/interpret/lib/"
   # ret_code=$?
   # if [ $ret_code -ne 0 ]; then 
   #    exit $ret_code
   # fi
   # cp "$script_path/tmp/clang/bin/release/mac/x86/ebmcore/ebmcore_mac_x86.dylib" "$script_path/staging/"
   # ret_code=$?
   # if [ $ret_code -ne 0 ]; then 
   #    exit $ret_code
   # fi

   # echo "Compiling with $clang_pp_bin for $os_type debug|x86"
   # [ -d "$script_path/tmp/clang/intermediate/debug/mac/x86/ebmcore" ] || mkdir -p "$script_path/tmp/clang/intermediate/debug/mac/x86/ebmcore"
   # ret_code=$?
   # if [ $ret_code -ne 0 ]; then 
   #    exit $ret_code
   # fi
   # [ -d "$script_path/tmp/clang/bin/debug/mac/x86/ebmcore" ] || mkdir -p "$script_path/tmp/clang/bin/debug/mac/x86/ebmcore"
   # ret_code=$?
   # if [ $ret_code -ne 0 ]; then 
   #    exit $ret_code
   # fi
   # compile_out=`$clang_pp_bin "$script_path/src/core/ebmcore/AttributeSet.cpp" "$script_path/src/core/ebmcore/DataSetByAttribute.cpp" "$script_path/src/core/ebmcore/DataSetByAttributeCombination.cpp" "$script_path/src/core/ebmcore/InteractionDetection.cpp" "$script_path/src/core/ebmcore/SamplingWithReplacement.cpp" "$script_path/src/core/ebmcore/Training.cpp" -I"$script_path/src/core/ebmcore" -I"$script_path/src/core/inc" -std=c++11 -fpermissive -fvisibility=hidden -O3 -march=core2 -static-libstdc++ -DEBMCORE_EXPORTS -fPIC -o "$script_path/tmp/clang/bin/debug/mac/x86/ebmcore/ebmcore_mac_x86_debug.dylib" -dynamiclib -install_name @rpath/ebmcore_mac_x86_debug.dylib -m32 2>&1`
   # ret_code=$?
   # echo -n "$compile_out"
   # echo -n "$compile_out" > "$script_path/tmp/clang/intermediate/debug/mac/x86/ebmcore/ebmcore_debug_mac_x86_build_log.txt"
   # if [ $ret_code -ne 0 ]; then 
   #    exit $ret_code
   # fi
   # cp "$script_path/tmp/clang/bin/debug/mac/x86/ebmcore/ebmcore_mac_x86_debug.dylib" "$script_path/src/python/interpret/lib/"
   # ret_code=$?
   # if [ $ret_code -ne 0 ]; then 
   #    exit $ret_code
   # fi
   # cp "$script_path/tmp/clang/bin/debug/mac/x86/ebmcore/ebmcore_mac_x86_debug.dylib" "$script_path/staging/"
   # ret_code=$?
   # if [ $ret_code -ne 0 ]; then 
   #    exit $ret_code
   # fi

elif [ "$os_type" = "Linux" ]; then

   # to cross compile for different architectures x86/x64, run the following command: sudo apt-get install g++-multilib

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

   echo "Compiling with $g_pp_bin for $os_type release|x64"
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
   compile_out=`$g_pp_bin "$script_path/src/core/ebmcore/AttributeSet.cpp" "$script_path/src/core/ebmcore/DataSetByAttribute.cpp" "$script_path/src/core/ebmcore/DataSetByAttributeCombination.cpp" "$script_path/src/core/ebmcore/InteractionDetection.cpp" "$script_path/src/core/ebmcore/SamplingWithReplacement.cpp" "$script_path/src/core/ebmcore/Training.cpp" -I"$script_path/src/core/ebmcore" -I"$script_path/src/core/inc" -std=c++11 -fpermissive -fvisibility=hidden -O3 -march=core2 -static-libgcc -static-libstdc++ -DEBMCORE_EXPORTS -fPIC -o "$script_path/tmp/gcc/bin/release/linux/x64/ebmcore/ebmcore_linux_x64.so" -shared -m64 -DNDEBUG 2>&1`
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

   echo "Compiling with $g_pp_bin for $os_type debug|x64"
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
   compile_out=`$g_pp_bin "$script_path/src/core/ebmcore/AttributeSet.cpp" "$script_path/src/core/ebmcore/DataSetByAttribute.cpp" "$script_path/src/core/ebmcore/DataSetByAttributeCombination.cpp" "$script_path/src/core/ebmcore/InteractionDetection.cpp" "$script_path/src/core/ebmcore/SamplingWithReplacement.cpp" "$script_path/src/core/ebmcore/Training.cpp" -I"$script_path/src/core/ebmcore" -I"$script_path/src/core/inc" -std=c++11 -fpermissive -fvisibility=hidden -O3 -march=core2 -static-libgcc -static-libstdc++ -DEBMCORE_EXPORTS -fPIC -o "$script_path/tmp/gcc/bin/debug/linux/x64/ebmcore/ebmcore_linux_x64_debug.so" -shared -m64 2>&1`
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

   # echo "Compiling with $g_pp_bin for $os_type release|x86"
   # [ -d "$script_path/tmp/gcc/intermediate/release/linux/x86/ebmcore" ] || mkdir -p "$script_path/tmp/gcc/intermediate/release/linux/x86/ebmcore"
   # ret_code=$?
   # if [ $ret_code -ne 0 ]; then 
   #    exit $ret_code
   # fi
   # [ -d "$script_path/tmp/gcc/bin/release/linux/x86/ebmcore" ] || mkdir -p "$script_path/tmp/gcc/bin/release/linux/x86/ebmcore"
   # ret_code=$?
   # if [ $ret_code -ne 0 ]; then 
   #    exit $ret_code
   # fi
   # compile_out=`$g_pp_bin "$script_path/src/core/ebmcore/AttributeSet.cpp" "$script_path/src/core/ebmcore/DataSetByAttribute.cpp" "$script_path/src/core/ebmcore/DataSetByAttributeCombination.cpp" "$script_path/src/core/ebmcore/InteractionDetection.cpp" "$script_path/src/core/ebmcore/SamplingWithReplacement.cpp" "$script_path/src/core/ebmcore/Training.cpp" -I"$script_path/src/core/ebmcore" -I"$script_path/src/core/inc" -std=c++11 -fpermissive -fvisibility=hidden -O3 -march=core2 -static-libgcc -static-libstdc++ -DEBMCORE_EXPORTS -fPIC -o "$script_path/tmp/gcc/bin/release/linux/x86/ebmcore/ebmcore_linux_x86.so" -shared -m32 -DNDEBUG 2>&1`
   # ret_code=$?
   # echo -n "$compile_out"
   # echo -n "$compile_out" > "$script_path/tmp/gcc/intermediate/release/linux/x86/ebmcore/ebmcore_release_linux_x86_build_log.txt"
   # if [ $ret_code -ne 0 ]; then 
   #    exit $ret_code
   # fi
   # cp "$script_path/tmp/gcc/bin/release/linux/x86/ebmcore/ebmcore_linux_x86.so" "$script_path/src/python/interpret/lib/"
   # ret_code=$?
   # if [ $ret_code -ne 0 ]; then 
   #    exit $ret_code
   # fi
   # cp "$script_path/tmp/gcc/bin/release/linux/x86/ebmcore/ebmcore_linux_x86.so" "$script_path/staging/"
   # ret_code=$?
   # if [ $ret_code -ne 0 ]; then 
   #    exit $ret_code
   # fi

   # echo "Compiling with $g_pp_bin for $os_type debug|x86"
   # [ -d "$script_path/tmp/gcc/intermediate/debug/linux/x86/ebmcore" ] || mkdir -p "$script_path/tmp/gcc/intermediate/debug/linux/x86/ebmcore"
   # ret_code=$?
   # if [ $ret_code -ne 0 ]; then 
   #    exit $ret_code
   # fi
   # [ -d "$script_path/tmp/gcc/bin/debug/linux/x86/ebmcore" ] || mkdir -p "$script_path/tmp/gcc/bin/debug/linux/x86/ebmcore"
   # ret_code=$?
   # if [ $ret_code -ne 0 ]; then 
   #    exit $ret_code
   # fi
   # compile_out=`$g_pp_bin "$script_path/src/core/ebmcore/AttributeSet.cpp" "$script_path/src/core/ebmcore/DataSetByAttribute.cpp" "$script_path/src/core/ebmcore/DataSetByAttributeCombination.cpp" "$script_path/src/core/ebmcore/InteractionDetection.cpp" "$script_path/src/core/ebmcore/SamplingWithReplacement.cpp" "$script_path/src/core/ebmcore/Training.cpp" -I"$script_path/src/core/ebmcore" -I"$script_path/src/core/inc" -std=c++11 -fpermissive -fvisibility=hidden -O3 -march=core2 -static-libgcc -static-libstdc++ -DEBMCORE_EXPORTS -fPIC -o "$script_path/tmp/gcc/bin/debug/linux/x86/ebmcore/ebmcore_linux_x86_debug.so" -shared -m32 2>&1`
   # ret_code=$?
   # echo -n "$compile_out"
   # echo -n "$compile_out" > "$script_path/tmp/gcc/intermediate/debug/linux/x86/ebmcore/ebmcore_debug_linux_x86_build_log.txt"
   # if [ $ret_code -ne 0 ]; then 
   #    exit $ret_code
   # fi
   # cp "$script_path/tmp/gcc/bin/debug/linux/x86/ebmcore/ebmcore_linux_x86_debug.so" "$script_path/src/python/interpret/lib/"
   # ret_code=$?
   # if [ $ret_code -ne 0 ]; then 
   #    exit $ret_code
   # fi
   # cp "$script_path/tmp/gcc/bin/debug/linux/x86/ebmcore/ebmcore_linux_x86_debug.so" "$script_path/staging/"
   # ret_code=$?
   # if [ $ret_code -ne 0 ]; then 
   #    exit $ret_code
   # fi

else
   echo "OS $os_type not recognized.  We support $clang_pp_bin on Darwin and $g_pp_bin on Linux"
   exit 1
fi
