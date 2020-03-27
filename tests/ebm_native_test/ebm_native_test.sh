#!/bin/sh

clang_pp_bin=clang++
g_pp_bin=g++
os_type=`uname`
script_path=`dirname "$0"`
root_path="$script_path/../.."
src_path="$script_path"
staging_path="$root_path/staging"
bin_file="ebm_native_test"

build_ebm_native=1
for arg in "$@"; do
   if [ "$arg" = "-nobuildebmnative" ]; then
      build_ebm_native=0
   fi
done

if [ $build_ebm_native -eq 1 ]; then
   echo "Building ebm_native library..."
   /bin/sh "$root_path/build.sh" -32bit
else
   echo "ebm_native library NOT being built"
fi

compile_all=""
compile_all="$compile_all \"$src_path/EbmNativeTest.cpp\""
compile_all="$compile_all -I\"$src_path\""
compile_all="$compile_all -I\"$root_path/shared/ebm_native/inc\""
compile_all="$compile_all -std=c++11 -fpermissive -O3 -march=core2"

if [ "$os_type" = "Darwin" ]; then
   # reference on rpath & install_name: https://www.mikeash.com/pyblog/friday-qa-2009-11-06-linking-and-install-names.html
   # the -l<library> parameter for some reason adds a lib at the start and .dylib at the end

   compile_mac="$compile_all -L\"$staging_path\" -Wl,-rpath,@loader_path"

   echo "Compiling $bin_file with $clang_pp_bin for macOS release|x64"
   intermediate_path="$root_path/tmp/clang/intermediate/release/mac/x64/ebm_native_test"
   bin_path="$root_path/tmp/clang/bin/release/mac/x64/ebm_native_test"
   lib_file_body="_ebm_native_mac_x64"
   log_file="$intermediate_path/ebm_native_test_release_mac_x64_build_log.txt"
   compile_command="$clang_pp_bin $compile_mac -m64 -DNDEBUG -l$lib_file_body -o \"$bin_path/$bin_file\" 2>&1"

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

   echo "Compiling $bin_file with $clang_pp_bin for macOS debug|x64"
   intermediate_path="$root_path/tmp/clang/intermediate/debug/mac/x64/ebm_native_test"
   bin_path="$root_path/tmp/clang/bin/debug/mac/x64/ebm_native_test"
   lib_file_body="_ebm_native_mac_x64_debug"
   log_file="$intermediate_path/ebm_native_test_debug_mac_x64_build_log.txt"
   compile_command="$clang_pp_bin $compile_mac -m64 -l$lib_file_body -o \"$bin_path/$bin_file\" 2>&1"

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

#    echo "Compiling $bin_file with $clang_pp_bin for macOS release|x86"
#    intermediate_path="$root_path/tmp/clang/intermediate/release/mac/x86/ebm_native_test"
#    bin_path="$root_path/tmp/clang/bin/release/mac/x86/ebm_native_test"
#    lib_file_body="_ebm_native_mac_x86"
#    log_file="$intermediate_path/ebm_native_test_release_mac_x86_build_log.txt"
#    compile_command="$clang_pp_bin $compile_mac -m32 -DNDEBUG -l$lib_file_body -o \"$bin_path/$bin_file\" 2>&1"

#    [ -d "$intermediate_path" ] || mkdir -p "$intermediate_path"
#    ret_code=$?
#    if [ $ret_code -ne 0 ]; then 
#       exit $ret_code
#    fi
#    [ -d "$bin_path" ] || mkdir -p "$bin_path"
#    ret_code=$?
#    if [ $ret_code -ne 0 ]; then 
#       exit $ret_code
#    fi
#    compile_out=`eval $compile_command`
#    ret_code=$?
#    echo -n "$compile_out"
#    echo -n "$compile_out" > "$log_file"
#    if [ $ret_code -ne 0 ]; then 
#       exit $ret_code
#    fi
#    cp "$staging_path/lib$lib_file_body.dylib" "$bin_path/"
#    ret_code=$?
#    if [ $ret_code -ne 0 ]; then 
#       exit $ret_code
#    fi
#    "$bin_path/$bin_file"
#    ret_code=$?
#    if [ $ret_code -ne 0 ]; then 
#       exit $ret_code
#    fi

#    echo "Compiling $bin_file with $clang_pp_bin for macOS debug|x86"
#    intermediate_path="$root_path/tmp/clang/intermediate/debug/mac/x86/ebm_native_test"
#    bin_path="$root_path/tmp/clang/bin/debug/mac/x86/ebm_native_test"
#    lib_file_body="_ebm_native_mac_x86_debug"
#    log_file="$intermediate_path/ebm_native_test_debug_mac_x86_build_log.txt"
#    compile_command="$clang_pp_bin $compile_mac -m32 -l$lib_file_body -o \"$bin_path/$bin_file\" 2>&1"

#    [ -d "$intermediate_path" ] || mkdir -p "$intermediate_path"
#    ret_code=$?
#    if [ $ret_code -ne 0 ]; then 
#       exit $ret_code
#    fi
#    [ -d "$bin_path" ] || mkdir -p "$bin_path"
#    ret_code=$?
#    if [ $ret_code -ne 0 ]; then 
#       exit $ret_code
#    fi
#    compile_out=`eval $compile_command`
#    ret_code=$?
#    echo -n "$compile_out"
#    echo -n "$compile_out" > "$log_file"
#    if [ $ret_code -ne 0 ]; then 
#       exit $ret_code
#    fi
#    cp "$staging_path/lib$lib_file_body.dylib" "$bin_path/"
#    ret_code=$?
#    if [ $ret_code -ne 0 ]; then 
#       exit $ret_code
#    fi
#    "$bin_path/$bin_file"
#    ret_code=$?
#    if [ $ret_code -ne 0 ]; then 
#       exit $ret_code
#    fi

elif [ "$os_type" = "Linux" ]; then
   # to cross compile for different architectures x86/x64, run the following command: sudo apt-get install g++-multilib
   # "readelf -d <lib_filename.so>" should show library rpath:    $ORIGIN/    OR    ${ORIGIN}/    for Linux so that the console app will find the ebm_native library in the same directory as the app: https://stackoverflow.com/questions/6288206/lookup-failure-when-linking-using-rpath-and-origin
   # the -l<library> parameter for some reason adds a lib at the start and .so at the end

   compile_linux="$compile_all -L\"$staging_path\" -Wl,-rpath-link,\"$staging_path\" -Wl,-rpath,'\$ORIGIN/'"

   echo "Compiling $bin_file with $g_pp_bin for Linux release|x64"
   intermediate_path="$root_path/tmp/gcc/intermediate/release/linux/x64/ebm_native_test"
   bin_path="$root_path/tmp/gcc/bin/release/linux/x64/ebm_native_test"
   lib_file_body="_ebm_native_linux_x64"
   log_file="$intermediate_path/ebm_native_test_release_linux_x64_build_log.txt"
   compile_command="$g_pp_bin $compile_linux -m64 -DNDEBUG -l$lib_file_body -o \"$bin_path/$bin_file\" 2>&1"

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

   echo "Compiling $bin_file with $g_pp_bin for Linux debug|x64"
   intermediate_path="$root_path/tmp/gcc/intermediate/debug/linux/x64/ebm_native_test"
   bin_path="$root_path/tmp/gcc/bin/debug/linux/x64/ebm_native_test"
   lib_file_body="_ebm_native_linux_x64_debug"
   log_file="$intermediate_path/ebm_native_test_debug_linux_x64_build_log.txt"
   compile_command="$g_pp_bin $compile_linux -m64 -l$lib_file_body -o \"$bin_path/$bin_file\" 2>&1"

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

   echo "Compiling $bin_file with $g_pp_bin for Linux release|x86"
   intermediate_path="$root_path/tmp/gcc/intermediate/release/linux/x86/ebm_native_test"
   bin_path="$root_path/tmp/gcc/bin/release/linux/x86/ebm_native_test"
   lib_file_body="_ebm_native_linux_x86"
   log_file="$intermediate_path/ebm_native_test_release_linux_x86_build_log.txt"
   compile_command="$g_pp_bin $compile_linux -m32 -DNDEBUG -l$lib_file_body -o \"$bin_path/$bin_file\" 2>&1"

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

   echo "Compiling $bin_file with $g_pp_bin for Linux debug|x86"
   intermediate_path="$root_path/tmp/gcc/intermediate/debug/linux/x86/ebm_native_test"
   bin_path="$root_path/tmp/gcc/bin/debug/linux/x86/ebm_native_test"
   lib_file_body="_ebm_native_linux_x86_debug"
   log_file="$intermediate_path/ebm_native_test_debug_linux_x86_build_log.txt"
   compile_command="$g_pp_bin $compile_linux -m32 -l$lib_file_body -o \"$bin_path/$bin_file\" 2>&1"

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
