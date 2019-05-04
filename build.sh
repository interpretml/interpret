#!/bin/sh

clang_pp_bin=clang++
g_pp_bin=g++

###################

os_type="$(uname)"

if [ "$os_type" = "Darwin" ]; then

   echo "Creating directories"
   [ -d tmp/clang/intermediate/release/mac/x64/ebmcore ] || mkdir -p tmp/clang/intermediate/release/mac/x64/ebmcore
   [ -d tmp/clang/bin/release/mac/x64/ebmcore ] || mkdir -p tmp/clang/bin/release/mac/x64/ebmcore
   [ -d tmp/clang/intermediate/release/mac/x86/ebmcore ] || mkdir -p tmp/clang/intermediate/release/mac/x86/ebmcore
   [ -d tmp/clang/bin/release/mac/x86/ebmcore ] || mkdir -p tmp/clang/bin/release/mac/x86/ebmcore
   [ -d tmp/clang/intermediate/debug/mac/x64/ebmcore ] || mkdir -p tmp/clang/intermediate/debug/mac/x64/ebmcore
   [ -d tmp/clang/bin/debug/mac/x64/ebmcore ] || mkdir -p tmp/clang/bin/debug/mac/x64/ebmcore
   [ -d tmp/clang/intermediate/debug/mac/x86/ebmcore ] || mkdir -p tmp/clang/intermediate/debug/mac/x86/ebmcore
   [ -d tmp/clang/bin/debug/mac/x86/ebmcore ] || mkdir -p tmp/clang/bin/debug/mac/x86/ebmcore

   echo "Compiling with $clang_pp_bin for $os_type"
   $clang_pp_bin src/core/ebmcore/AttributeSet.cpp src/core/ebmcore/DataSetByAttribute.cpp src/core/ebmcore/DataSetByAttributeCombination.cpp src/core/ebmcore/InteractionDetection.cpp src/core/ebmcore/SamplingWithReplacement.cpp src/core/ebmcore/Training.cpp -Isrc/core/ebmcore -Isrc/core/inc -m64 -std=c++11 -fpermissive -fvisibility=hidden -O3 -march=native -DNDEBUG -DEBMCORE_EXPORTS -fPIC -dynamiclib -install_name @rpath/ebmcore_mac_x64.dylib -o tmp/clang/bin/release/mac/x64/ebmcore/ebmcore_mac_x64.dylib 2> tmp/clang/intermediate/release/mac/x64/ebmcore/ebmcore_mac_x64_build_output.txt
   $clang_pp_bin src/core/ebmcore/AttributeSet.cpp src/core/ebmcore/DataSetByAttribute.cpp src/core/ebmcore/DataSetByAttributeCombination.cpp src/core/ebmcore/InteractionDetection.cpp src/core/ebmcore/SamplingWithReplacement.cpp src/core/ebmcore/Training.cpp -Isrc/core/ebmcore -Isrc/core/inc -m32 -std=c++11 -fpermissive -fvisibility=hidden -O3 -march=native -DNDEBUG -DEBMCORE_EXPORTS -fPIC -dynamiclib -install_name @rpath/ebmcore_mac_x86.dylib -o tmp/clang/bin/release/mac/x86/ebmcore/ebmcore_mac_x86.dylib 2> tmp/clang/intermediate/release/mac/x86/ebmcore/ebmcore_mac_x86_build_output.txt
   $clang_pp_bin src/core/ebmcore/AttributeSet.cpp src/core/ebmcore/DataSetByAttribute.cpp src/core/ebmcore/DataSetByAttributeCombination.cpp src/core/ebmcore/InteractionDetection.cpp src/core/ebmcore/SamplingWithReplacement.cpp src/core/ebmcore/Training.cpp -Isrc/core/ebmcore -Isrc/core/inc -m64 -std=c++11 -fpermissive -fvisibility=hidden -O3 -march=native -DEBMCORE_EXPORTS -fPIC -dynamiclib -install_name @rpath/ebmcore_mac_x64_debug.dylib -o tmp/clang/bin/debug/mac/x64/ebmcore/ebmcore_mac_x64_debug.dylib 2> tmp/clang/intermediate/debug/mac/x64/ebmcore/ebmcore_mac_x64_debug_build_output.txt
   $clang_pp_bin src/core/ebmcore/AttributeSet.cpp src/core/ebmcore/DataSetByAttribute.cpp src/core/ebmcore/DataSetByAttributeCombination.cpp src/core/ebmcore/InteractionDetection.cpp src/core/ebmcore/SamplingWithReplacement.cpp src/core/ebmcore/Training.cpp -Isrc/core/ebmcore -Isrc/core/inc -m32 -std=c++11 -fpermissive -fvisibility=hidden -O3 -march=native -DEBMCORE_EXPORTS -fPIC -dynamiclib -install_name @rpath/ebmcore_mac_x86_debug.dylib -o tmp/clang/bin/debug/mac/x86/ebmcore/ebmcore_mac_x86_debug.dylib 2> tmp/clang/intermediate/debug/mac/x86/ebmcore/ebmcore_mac_x86_debug_build_output.txt

   echo "Copying results"
   [ -d staging ] || mkdir -p staging
   [ -d src/python/interpret/lib ] || mkdir -p src/python/interpret/lib
   cp tmp/clang/bin/release/mac/x64/ebmcore/ebmcore_mac_x64.dylib staging/
   cp tmp/clang/bin/release/mac/x64/ebmcore/ebmcore_mac_x64.dylib src/python/interpret/lib/
   cp tmp/clang/bin/release/mac/x86/ebmcore/ebmcore_mac_x86.dylib staging/
   cp tmp/clang/bin/release/mac/x86/ebmcore/ebmcore_mac_x86.dylib src/python/interpret/lib/
   cp tmp/clang/bin/debug/mac/x64/ebmcore/ebmcore_mac_x64_debug.dylib staging/
   cp tmp/clang/bin/debug/mac/x64/ebmcore/ebmcore_mac_x64_debug.dylib src/python/interpret/lib/
   cp tmp/clang/bin/debug/mac/x86/ebmcore/ebmcore_mac_x86_debug.dylib staging/
   cp tmp/clang/bin/debug/mac/x86/ebmcore/ebmcore_mac_x86_debug.dylib src/python/interpret/lib/

elif [ "$os_type" = "Linux" ]; then

   echo "Creating directories"
   [ -d tmp/gcc/intermediate/release/linux/x64/ebmcore ] || mkdir -p tmp/gcc/intermediate/release/linux/x64/ebmcore
   [ -d tmp/gcc/bin/release/linux/x64/ebmcore ] || mkdir -p tmp/gcc/bin/release/linux/x64/ebmcore
   [ -d tmp/gcc/intermediate/release/linux/x86/ebmcore ] || mkdir -p tmp/gcc/intermediate/release/linux/x86/ebmcore
   [ -d tmp/gcc/bin/release/linux/x86/ebmcore ] || mkdir -p tmp/gcc/bin/release/linux/x86/ebmcore
   [ -d tmp/gcc/intermediate/debug/linux/x64/ebmcore ] || mkdir -p tmp/gcc/intermediate/debug/linux/x64/ebmcore
   [ -d tmp/gcc/bin/debug/linux/x64/ebmcore ] || mkdir -p tmp/gcc/bin/debug/linux/x64/ebmcore
   [ -d tmp/gcc/intermediate/debug/linux/x86/ebmcore ] || mkdir -p tmp/gcc/intermediate/debug/linux/x86/ebmcore
   [ -d tmp/gcc/bin/debug/linux/x86/ebmcore ] || mkdir -p tmp/gcc/bin/debug/linux/x86/ebmcore

   echo "Compiling with $g_pp_bin for $os_type"
   # to cross compile for different architectures x86/x64, run the following command: sudo apt-get install gcc-multilib g++-multilib
   $g_pp_bin src/core/ebmcore/AttributeSet.cpp src/core/ebmcore/DataSetByAttribute.cpp src/core/ebmcore/DataSetByAttributeCombination.cpp src/core/ebmcore/InteractionDetection.cpp src/core/ebmcore/SamplingWithReplacement.cpp src/core/ebmcore/Training.cpp -Isrc/core/ebmcore -Isrc/core/inc -m64 -std=c++11 -fpermissive -fvisibility=hidden -O3 -march=native -DNDEBUG -DEBMCORE_EXPORTS -fPIC -shared -o tmp/gcc/bin/release/linux/x64/ebmcore/ebmcore_linux_x64.so 2> tmp/gcc/intermediate/release/linux/x64/ebmcore/ebmcore_linux_x64_build_output.txt
   $g_pp_bin src/core/ebmcore/AttributeSet.cpp src/core/ebmcore/DataSetByAttribute.cpp src/core/ebmcore/DataSetByAttributeCombination.cpp src/core/ebmcore/InteractionDetection.cpp src/core/ebmcore/SamplingWithReplacement.cpp src/core/ebmcore/Training.cpp -Isrc/core/ebmcore -Isrc/core/inc -m32 -std=c++11 -fpermissive -fvisibility=hidden -O3 -march=native -DNDEBUG -DEBMCORE_EXPORTS -fPIC -shared -o tmp/gcc/bin/release/linux/x86/ebmcore/ebmcore_linux_x86.so 2> tmp/gcc/intermediate/release/linux/x86/ebmcore/ebmcore_linux_x86_build_output.txt
   $g_pp_bin src/core/ebmcore/AttributeSet.cpp src/core/ebmcore/DataSetByAttribute.cpp src/core/ebmcore/DataSetByAttributeCombination.cpp src/core/ebmcore/InteractionDetection.cpp src/core/ebmcore/SamplingWithReplacement.cpp src/core/ebmcore/Training.cpp -Isrc/core/ebmcore -Isrc/core/inc -m64 -std=c++11 -fpermissive -fvisibility=hidden -O3 -march=native -DEBMCORE_EXPORTS -fPIC -shared -o tmp/gcc/bin/debug/linux/x64/ebmcore/ebmcore_linux_x64_debug.so 2> tmp/gcc/intermediate/debug/linux/x64/ebmcore/ebmcore_linux_x64_debug_build_output.txt
   $g_pp_bin src/core/ebmcore/AttributeSet.cpp src/core/ebmcore/DataSetByAttribute.cpp src/core/ebmcore/DataSetByAttributeCombination.cpp src/core/ebmcore/InteractionDetection.cpp src/core/ebmcore/SamplingWithReplacement.cpp src/core/ebmcore/Training.cpp -Isrc/core/ebmcore -Isrc/core/inc -m32 -std=c++11 -fpermissive -fvisibility=hidden -O3 -march=native -DEBMCORE_EXPORTS -fPIC -shared -o tmp/gcc/bin/debug/linux/x86/ebmcore/ebmcore_linux_x86_debug.so 2> tmp/gcc/intermediate/debug/linux/x86/ebmcore/ebmcore_linux_x86_debug_build_output.txt

   echo "Copying results"
   [ -d staging ] || mkdir -p staging
   [ -d src/python/interpret/lib ] || mkdir -p src/python/interpret/lib
   cp tmp/gcc/bin/release/linux/x64/ebmcore/ebmcore_linux_x64.so staging/
   cp tmp/gcc/bin/release/linux/x64/ebmcore/ebmcore_linux_x64.so src/python/interpret/lib/
   cp tmp/gcc/bin/release/linux/x86/ebmcore/ebmcore_linux_x86.so staging/
   cp tmp/gcc/bin/release/linux/x86/ebmcore/ebmcore_linux_x86.so src/python/interpret/lib/
   cp tmp/gcc/bin/debug/linux/x64/ebmcore/ebmcore_linux_x64_debug.so staging/
   cp tmp/gcc/bin/debug/linux/x64/ebmcore/ebmcore_linux_x64_debug.so src/python/interpret/lib/
   cp tmp/gcc/bin/debug/linux/x86/ebmcore/ebmcore_linux_x86_debug.so staging/
   cp tmp/gcc/bin/debug/linux/x86/ebmcore/ebmcore_linux_x86_debug.so src/python/interpret/lib/

else
   echo "OS $os_type not recognized.  We support $clang_pp_bin on Darwin and $g_pp_bin on Linux"
   exit 1
fi
