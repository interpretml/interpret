#!/bin/sh

clang_pp_bin=clang++
g_pp_bin=g++
os_type=`uname`
script_path=`dirname "$0"`
root_path="$script_path/../../.."
src_path="$script_path"
staging_path="$root_path/staging"
bin_file="ebm_native_test"

build_pipeline=0
for arg in "$@"; do
   if [ "$arg" = "-pipeline" ]; then
      build_pipeline=1
   fi
done

if [ $build_pipeline -eq 0 ]; then
   echo "Building ebm_native library..."
   /bin/sh "$root_path/build.sh" -32bit -analysis
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      # build.sh should write out any messages
      exit $ret_code
   fi
fi

compile_all=""
compile_all="$compile_all \"$src_path/bit_packing_extremes.cpp\""
compile_all="$compile_all \"$src_path/boosting_unusual_inputs.cpp\""
compile_all="$compile_all \"$src_path/CutQuantile.cpp\""
compile_all="$compile_all \"$src_path/CutUniform.cpp\""
compile_all="$compile_all \"$src_path/CutWinsorized.cpp\""
compile_all="$compile_all \"$src_path/Discretize.cpp\""
compile_all="$compile_all \"$src_path/ebm_native_test.cpp\""
compile_all="$compile_all \"$src_path/interaction_unusual_inputs.cpp\""
compile_all="$compile_all \"$src_path/random_numbers.cpp\""
compile_all="$compile_all \"$src_path/rehydrate_booster.cpp\""
compile_all="$compile_all \"$src_path/SuggestGraphBounds.cpp\""
compile_all="$compile_all -I\"$src_path\""
compile_all="$compile_all -I\"$root_path/shared/ebm_native/inc\""
compile_all="$compile_all -fno-math-errno -fno-trapping-math"
compile_all="$compile_all -march=core2"
compile_all="$compile_all -std=c++11"

if [ "$os_type" = "Darwin" ]; then
   # reference on rpath & install_name: https://www.mikeash.com/pyblog/friday-qa-2009-11-06-linking-and-install-names.html
   # the -l<library> parameter for some reason adds a lib at the start and .dylib at the end

   compile_mac="$compile_all -L\"$staging_path\" -Wl,-rpath,@loader_path"

   ASAN_OPTIONS=detect_leaks=1:detect_stack_use_after_return=1:check_initialization_order=1:alloc_dealloc_mismatch=1:strict_init_order=1:strict_string_checks=1:detect_invalid_pointer_pairs=2

   ########################## macOS debug|x64

   echo "Compiling $bin_file with $clang_pp_bin for macOS debug|x64"
   obj_path="$root_path/tmp/clang/obj/debug/mac/x64/ebm_native_test"
   bin_path="$root_path/tmp/clang/bin/debug/mac/x64/ebm_native_test"
   lib_file_body="_ebm_native_mac_x64_debug"
   log_file="$obj_path/ebm_native_test_debug_mac_x64_build_log.txt"
   compile_command="$clang_pp_bin $compile_mac -m64 -l$lib_file_body -o \"$bin_path/$bin_file\" 2>&1"

   [ -d "$obj_path" ] || mkdir -p "$obj_path"
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
   echo -n "$compile_out"
   echo -n "$compile_out" > "$log_file"
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   cp "$staging_path/lib$lib_file_body.dylib" "$bin_path/"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   "$bin_path/$bin_file"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi

   ########################## macOS release|x64

   echo "Compiling $bin_file with $clang_pp_bin for macOS release|x64"
   obj_path="$root_path/tmp/clang/obj/release/mac/x64/ebm_native_test"
   bin_path="$root_path/tmp/clang/bin/release/mac/x64/ebm_native_test"
   lib_file_body="_ebm_native_mac_x64"
   log_file="$obj_path/ebm_native_test_release_mac_x64_build_log.txt"
   compile_command="$clang_pp_bin $compile_mac -m64 -DNDEBUG -l$lib_file_body -o \"$bin_path/$bin_file\" 2>&1"

   [ -d "$obj_path" ] || mkdir -p "$obj_path"
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
   echo -n "$compile_out"
   echo -n "$compile_out" > "$log_file"
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   cp "$staging_path/lib$lib_file_body.dylib" "$bin_path/"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   "$bin_path/$bin_file"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi

elif [ "$os_type" = "Linux" ]; then
   # "readelf -d <lib_filename.so>" should show library rpath:    $ORIGIN/    OR    ${ORIGIN}/    for Linux so that the console app will find the ebm_native library in the same directory as the app: https://stackoverflow.com/questions/6288206/lookup-failure-when-linking-using-rpath-and-origin
   # the -l<library> parameter for some reason adds a lib at the start and .so at the end

   compile_linux="$compile_all -L\"$staging_path\" -Wl,-rpath-link,\"$staging_path\" -Wl,-rpath,'\$ORIGIN/'"


   ########################## Linux debug|x64

   echo "Compiling $bin_file with $g_pp_bin for Linux debug|x64"
   obj_path="$root_path/tmp/gcc/obj/debug/linux/x64/ebm_native_test"
   bin_path="$root_path/tmp/gcc/bin/debug/linux/x64/ebm_native_test"
   lib_file_body="_ebm_native_linux_x64_debug"
   log_file="$obj_path/ebm_native_test_debug_linux_x64_build_log.txt"
   compile_command="$g_pp_bin $compile_linux -m64 -l$lib_file_body -o \"$bin_path/$bin_file\" 2>&1"

   if [ ! -d "$obj_path" ]; then
      printf "%s\n" "Doing first time installation of Linux debug|x64"

      # this is the first time we're being compiled x64 on this machine, so install other required items

      # TODO consider NOT running sudo inside this script and move that requirement to the caller
      #      per https://askubuntu.com/questions/425754/how-do-i-run-a-sudo-command-inside-a-script

      sudo apt-get -y update
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi

      sudo apt-get -y install valgrind
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         exit $ret_code
      fi

      mkdir -p "$obj_path"
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
   echo -n "$compile_out"
   echo -n "$compile_out" > "$log_file"
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   cp "$staging_path/lib$lib_file_body.so" "$bin_path/"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   valgrind --error-exitcode=99 --leak-check=yes "$bin_path/$bin_file"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi

   ########################## Linux release|x64

   echo "Compiling $bin_file with $g_pp_bin for Linux release|x64"
   obj_path="$root_path/tmp/gcc/obj/release/linux/x64/ebm_native_test"
   bin_path="$root_path/tmp/gcc/bin/release/linux/x64/ebm_native_test"
   lib_file_body="_ebm_native_linux_x64"
   log_file="$obj_path/ebm_native_test_release_linux_x64_build_log.txt"
   compile_command="$g_pp_bin $compile_linux -m64 -DNDEBUG -l$lib_file_body -o \"$bin_path/$bin_file\" 2>&1"

   [ -d "$obj_path" ] || mkdir -p "$obj_path"
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
   echo -n "$compile_out"
   echo -n "$compile_out" > "$log_file"
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   cp "$staging_path/lib$lib_file_body.so" "$bin_path/"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   valgrind --error-exitcode=99 --leak-check=yes "$bin_path/$bin_file"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi



   ########################## Pipeline build for 32 bit and static analysis
   if [ $build_pipeline -eq 1 ]; then
      echo "Building ebm_native library for 32 bit and static analysis..."
      /bin/sh "$root_path/build.sh" -32bit -analysis
      ret_code=$?
      if [ $ret_code -ne 0 ]; then 
         # build.sh should write out any messages
         exit $ret_code
      fi
   fi


   ########################## Linux debug|x86

   echo "Compiling $bin_file with $g_pp_bin for Linux debug|x86"
   obj_path="$root_path/tmp/gcc/obj/debug/linux/x86/ebm_native_test"
   bin_path="$root_path/tmp/gcc/bin/debug/linux/x86/ebm_native_test"
   lib_file_body="_ebm_native_linux_x86_debug"
   log_file="$obj_path/ebm_native_test_debug_linux_x86_build_log.txt"
   compile_command="$g_pp_bin $compile_linux -m32 -l$lib_file_body -o \"$bin_path/$bin_file\" 2>&1"

   [ -d "$obj_path" ] || mkdir -p "$obj_path"
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
   echo -n "$compile_out"
   echo -n "$compile_out" > "$log_file"
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   cp "$staging_path/lib$lib_file_body.so" "$bin_path/"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   "$bin_path/$bin_file"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi

   ########################## Linux release|x86

   echo "Compiling $bin_file with $g_pp_bin for Linux release|x86"
   obj_path="$root_path/tmp/gcc/obj/release/linux/x86/ebm_native_test"
   bin_path="$root_path/tmp/gcc/bin/release/linux/x86/ebm_native_test"
   lib_file_body="_ebm_native_linux_x86"
   log_file="$obj_path/ebm_native_test_release_linux_x86_build_log.txt"
   compile_command="$g_pp_bin $compile_linux -m32 -DNDEBUG -l$lib_file_body -o \"$bin_path/$bin_file\" 2>&1"

   [ -d "$obj_path" ] || mkdir -p "$obj_path"
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
   echo -n "$compile_out"
   echo -n "$compile_out" > "$log_file"
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   cp "$staging_path/lib$lib_file_body.so" "$bin_path/"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   "$bin_path/$bin_file"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi


   echo Passed All Tests
else
   echo "OS $os_type not recognized.  We support $clang_pp_bin on macOS and $g_pp_bin on Linux"
   exit 1
fi
