#!/bin/sh

clang_pp_bin=clang++
g_pp_bin=g++
os_type="$(uname)"
script_path="`dirname \"$0\"`"

if [ "$os_type" = "Darwin" ]; then
   echo "Creating initial directories"
   [ -d "$script_path/staging" ] || mkdir -p "$script_path/staging"
   [ -d "$script_path/src/python/interpret/lib" ] || mkdir -p "$script_path/src/python/interpret/lib"

   echo "Compiling with $clang_pp_bin for $os_type release|x64"
   [ -d "$script_path/tmp/clang/intermediate/release/mac/x64/ebmcore" ] || mkdir -p "$script_path/tmp/clang/intermediate/release/mac/x64/ebmcore"
   [ -d "$script_path/tmp/clang/bin/release/mac/x64/ebmcore" ] || mkdir -p "$script_path/tmp/clang/bin/release/mac/x64/ebmcore"
   $clang_pp_bin "$script_path/src/core/ebmcore/AttributeSet.cpp" "$script_path/src/core/ebmcore/DataSetByAttribute.cpp" "$script_path/src/core/ebmcore/DataSetByAttributeCombination.cpp" "$script_path/src/core/ebmcore/InteractionDetection.cpp" "$script_path/src/core/ebmcore/SamplingWithReplacement.cpp" "$script_path/src/core/ebmcore/Training.cpp" -I"$script_path/src/core/ebmcore" -I"$script_path/src/core/inc" -m64 -std=c++11 -fpermissive -fvisibility=hidden -O3 -march=native -DNDEBUG -DEBMCORE_EXPORTS -fPIC -dynamiclib -install_name @rpath/ebmcore_mac_x64.dylib -o "$script_path/tmp/clang/bin/release/mac/x64/ebmcore/ebmcore_mac_x64.dylib" 2> "$script_path/tmp/clang/intermediate/release/mac/x64/ebmcore/ebmcore_release_mac_x64_build_log.txt"
   cp "$script_path/tmp/clang/bin/release/mac/x64/ebmcore/ebmcore_mac_x64.dylib" "$script_path/src/python/interpret/lib/"
   cp "$script_path/tmp/clang/bin/release/mac/x64/ebmcore/ebmcore_mac_x64.dylib" "$script_path/staging/"

   echo "Compiling with $clang_pp_bin for $os_type debug|x64"
   [ -d "$script_path/tmp/clang/intermediate/debug/mac/x64/ebmcore" ] || mkdir -p "$script_path/tmp/clang/intermediate/debug/mac/x64/ebmcore"
   [ -d "$script_path/tmp/clang/bin/debug/mac/x64/ebmcore" ] || mkdir -p "$script_path/tmp/clang/bin/debug/mac/x64/ebmcore"
   $clang_pp_bin "$script_path/src/core/ebmcore/AttributeSet.cpp" "$script_path/src/core/ebmcore/DataSetByAttribute.cpp" "$script_path/src/core/ebmcore/DataSetByAttributeCombination.cpp" "$script_path/src/core/ebmcore/InteractionDetection.cpp" "$script_path/src/core/ebmcore/SamplingWithReplacement.cpp" "$script_path/src/core/ebmcore/Training.cpp" -I"$script_path/src/core/ebmcore" -I"$script_path/src/core/inc" -m64 -std=c++11 -fpermissive -fvisibility=hidden -O3 -march=native -DEBMCORE_EXPORTS -fPIC -dynamiclib -install_name @rpath/ebmcore_mac_x64_debug.dylib -o "$script_path/tmp/clang/bin/debug/mac/x64/ebmcore/ebmcore_mac_x64_debug.dylib" 2> "$script_path/tmp/clang/intermediate/debug/mac/x64/ebmcore/ebmcore_debug_mac_x64_build_log.txt"
   cp "$script_path/tmp/clang/bin/debug/mac/x64/ebmcore/ebmcore_mac_x64_debug.dylib" "$script_path/src/python/interpret/lib/"
   cp "$script_path/tmp/clang/bin/debug/mac/x64/ebmcore/ebmcore_mac_x64_debug.dylib" "$script_path/staging/"

   # echo "Compiling with $clang_pp_bin for $os_type release|x86"
   # [ -d "$script_path/tmp/clang/intermediate/release/mac/x86/ebmcore" ] || mkdir -p "$script_path/tmp/clang/intermediate/release/mac/x86/ebmcore"
   # [ -d "$script_path/tmp/clang/bin/release/mac/x86/ebmcore" ] || mkdir -p "$script_path/tmp/clang/bin/release/mac/x86/ebmcore"
   # $clang_pp_bin "$script_path/src/core/ebmcore/AttributeSet.cpp" "$script_path/src/core/ebmcore/DataSetByAttribute.cpp" "$script_path/src/core/ebmcore/DataSetByAttributeCombination.cpp" "$script_path/src/core/ebmcore/InteractionDetection.cpp" "$script_path/src/core/ebmcore/SamplingWithReplacement.cpp" "$script_path/src/core/ebmcore/Training.cpp" -I"$script_path/src/core/ebmcore" -I"$script_path/src/core/inc" -m32 -std=c++11 -fpermissive -fvisibility=hidden -O3 -march=native -DNDEBUG -DEBMCORE_EXPORTS -fPIC -dynamiclib -install_name @rpath/ebmcore_mac_x86.dylib -o "$script_path/tmp/clang/bin/release/mac/x86/ebmcore/ebmcore_mac_x86.dylib" 2> "$script_path/tmp/clang/intermediate/release/mac/x86/ebmcore/ebmcore_release_mac_x86_build_log.txt"
   # cp "$script_path/tmp/clang/bin/release/mac/x86/ebmcore/ebmcore_mac_x86.dylib" "$script_path/src/python/interpret/lib/"
   # cp "$script_path/tmp/clang/bin/release/mac/x86/ebmcore/ebmcore_mac_x86.dylib" "$script_path/staging/"

   # echo "Compiling with $clang_pp_bin for $os_type debug|x86"
   # [ -d "$script_path/tmp/clang/intermediate/debug/mac/x86/ebmcore" ] || mkdir -p "$script_path/tmp/clang/intermediate/debug/mac/x86/ebmcore"
   # [ -d "$script_path/tmp/clang/bin/debug/mac/x86/ebmcore" ] || mkdir -p "$script_path/tmp/clang/bin/debug/mac/x86/ebmcore"
   # $clang_pp_bin "$script_path/src/core/ebmcore/AttributeSet.cpp" "$script_path/src/core/ebmcore/DataSetByAttribute.cpp" "$script_path/src/core/ebmcore/DataSetByAttributeCombination.cpp" "$script_path/src/core/ebmcore/InteractionDetection.cpp" "$script_path/src/core/ebmcore/SamplingWithReplacement.cpp" "$script_path/src/core/ebmcore/Training.cpp" -I"$script_path/src/core/ebmcore" -I"$script_path/src/core/inc" -m32 -std=c++11 -fpermissive -fvisibility=hidden -O3 -march=native -DEBMCORE_EXPORTS -fPIC -dynamiclib -install_name @rpath/ebmcore_mac_x86_debug.dylib -o "$script_path/tmp/clang/bin/debug/mac/x86/ebmcore/ebmcore_mac_x86_debug.dylib" 2> "$script_path/tmp/clang/intermediate/debug/mac/x86/ebmcore/ebmcore_debug_mac_x86_build_log.txt"
   # cp "$script_path/tmp/clang/bin/debug/mac/x86/ebmcore/ebmcore_mac_x86_debug.dylib" "$script_path/src/python/interpret/lib/"
   # cp "$script_path/tmp/clang/bin/debug/mac/x86/ebmcore/ebmcore_mac_x86_debug.dylib" "$script_path/staging/"

elif [ "$os_type" = "Linux" ]; then
   # to cross compile for different architectures x86/x64, run the following command: sudo apt-get install g++-multilib
   echo "Creating initial directories"
   [ -d "$script_path/staging" ] || mkdir -p "$script_path/staging"
   [ -d "$script_path/src/python/interpret/lib" ] || mkdir -p "$script_path/src/python/interpret/lib"

   echo "Compiling with $g_pp_bin for $os_type release|x64"
   [ -d "$script_path/tmp/gcc/intermediate/release/linux/x64/ebmcore" ] || mkdir -p "$script_path/tmp/gcc/intermediate/release/linux/x64/ebmcore"
   [ -d "$script_path/tmp/gcc/bin/release/linux/x64/ebmcore" ] || mkdir -p "$script_path/tmp/gcc/bin/release/linux/x64/ebmcore"
   $g_pp_bin "$script_path/src/core/ebmcore/AttributeSet.cpp" "$script_path/src/core/ebmcore/DataSetByAttribute.cpp" "$script_path/src/core/ebmcore/DataSetByAttributeCombination.cpp" "$script_path/src/core/ebmcore/InteractionDetection.cpp" "$script_path/src/core/ebmcore/SamplingWithReplacement.cpp" "$script_path/src/core/ebmcore/Training.cpp" -I"$script_path/src/core/ebmcore" -I"$script_path/src/core/inc" -m64 -std=c++11 -fpermissive -fvisibility=hidden -O3 -march=native -DNDEBUG -DEBMCORE_EXPORTS -fPIC -shared -o "$script_path/tmp/gcc/bin/release/linux/x64/ebmcore/ebmcore_linux_x64.so" 2> "$script_path/tmp/gcc/intermediate/release/linux/x64/ebmcore/ebmcore_release_linux_x64_build_log.txt"
   cp "$script_path/tmp/gcc/bin/release/linux/x64/ebmcore/ebmcore_linux_x64.so" "$script_path/src/python/interpret/lib/"
   cp "$script_path/tmp/gcc/bin/release/linux/x64/ebmcore/ebmcore_linux_x64.so" "$script_path/staging/"

   echo "Compiling with $g_pp_bin for $os_type debug|x64"
   [ -d "$script_path/tmp/gcc/intermediate/debug/linux/x64/ebmcore" ] || mkdir -p "$script_path/tmp/gcc/intermediate/debug/linux/x64/ebmcore"
   [ -d "$script_path/tmp/gcc/bin/debug/linux/x64/ebmcore" ] || mkdir -p "$script_path/tmp/gcc/bin/debug/linux/x64/ebmcore"
   $g_pp_bin "$script_path/src/core/ebmcore/AttributeSet.cpp" "$script_path/src/core/ebmcore/DataSetByAttribute.cpp" "$script_path/src/core/ebmcore/DataSetByAttributeCombination.cpp" "$script_path/src/core/ebmcore/InteractionDetection.cpp" "$script_path/src/core/ebmcore/SamplingWithReplacement.cpp" "$script_path/src/core/ebmcore/Training.cpp" -I"$script_path/src/core/ebmcore" -I"$script_path/src/core/inc" -m64 -std=c++11 -fpermissive -fvisibility=hidden -O3 -march=native -DEBMCORE_EXPORTS -fPIC -shared -o "$script_path/tmp/gcc/bin/debug/linux/x64/ebmcore/ebmcore_linux_x64_debug.so" 2> "$script_path/tmp/gcc/intermediate/debug/linux/x64/ebmcore/ebmcore_debug_linux_x64_build_log.txt"
   cp "$script_path/tmp/gcc/bin/debug/linux/x64/ebmcore/ebmcore_linux_x64_debug.so" "$script_path/src/python/interpret/lib/"
   cp "$script_path/tmp/gcc/bin/debug/linux/x64/ebmcore/ebmcore_linux_x64_debug.so" "$script_path/staging/"

   # echo "Compiling with $g_pp_bin for $os_type release|x86"
   # [ -d "$script_path/tmp/gcc/intermediate/release/linux/x86/ebmcore" ] || mkdir -p "$script_path/tmp/gcc/intermediate/release/linux/x86/ebmcore"
   # [ -d "$script_path/tmp/gcc/bin/release/linux/x86/ebmcore" ] || mkdir -p "$script_path/tmp/gcc/bin/release/linux/x86/ebmcore"
   # $g_pp_bin "$script_path/src/core/ebmcore/AttributeSet.cpp" "$script_path/src/core/ebmcore/DataSetByAttribute.cpp" "$script_path/src/core/ebmcore/DataSetByAttributeCombination.cpp" "$script_path/src/core/ebmcore/InteractionDetection.cpp" "$script_path/src/core/ebmcore/SamplingWithReplacement.cpp" "$script_path/src/core/ebmcore/Training.cpp" -I"$script_path/src/core/ebmcore" -I"$script_path/src/core/inc" -m32 -std=c++11 -fpermissive -fvisibility=hidden -O3 -march=native -DNDEBUG -DEBMCORE_EXPORTS -fPIC -shared -o "$script_path/tmp/gcc/bin/release/linux/x86/ebmcore/ebmcore_linux_x86.so" 2> "$script_path/tmp/gcc/intermediate/release/linux/x86/ebmcore/ebmcore_release_linux_x86_build_log.txt"
   # cp "$script_path/tmp/gcc/bin/release/linux/x86/ebmcore/ebmcore_linux_x86.so" "$script_path/src/python/interpret/lib/"
   # cp "$script_path/tmp/gcc/bin/release/linux/x86/ebmcore/ebmcore_linux_x86.so" "$script_path/staging/"

   # echo "Compiling with $g_pp_bin for $os_type debug|x86"
   # [ -d "$script_path/tmp/gcc/intermediate/debug/linux/x86/ebmcore" ] || mkdir -p "$script_path/tmp/gcc/intermediate/debug/linux/x86/ebmcore"
   # [ -d "$script_path/tmp/gcc/bin/debug/linux/x86/ebmcore" ] || mkdir -p "$script_path/tmp/gcc/bin/debug/linux/x86/ebmcore"
   # $g_pp_bin "$script_path/src/core/ebmcore/AttributeSet.cpp" "$script_path/src/core/ebmcore/DataSetByAttribute.cpp" "$script_path/src/core/ebmcore/DataSetByAttributeCombination.cpp" "$script_path/src/core/ebmcore/InteractionDetection.cpp" "$script_path/src/core/ebmcore/SamplingWithReplacement.cpp" "$script_path/src/core/ebmcore/Training.cpp" -I"$script_path/src/core/ebmcore" -I"$script_path/src/core/inc" -m32 -std=c++11 -fpermissive -fvisibility=hidden -O3 -march=native -DEBMCORE_EXPORTS -fPIC -shared -o "$script_path/tmp/gcc/bin/debug/linux/x86/ebmcore/ebmcore_linux_x86_debug.so" 2> "$script_path/tmp/gcc/intermediate/debug/linux/x86/ebmcore/ebmcore_debug_linux_x86_build_log.txt"
   # cp "$script_path/tmp/gcc/bin/debug/linux/x86/ebmcore/ebmcore_linux_x86_debug.so" "$script_path/src/python/interpret/lib/"
   # cp "$script_path/tmp/gcc/bin/debug/linux/x86/ebmcore/ebmcore_linux_x86_debug.so" "$script_path/staging/"

else
   echo "OS $os_type not recognized.  We support $clang_pp_bin on Darwin and $g_pp_bin on Linux"
   exit 1
fi
