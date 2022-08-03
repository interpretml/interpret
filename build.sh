#!/bin/sh

# TODO also build our html resources here, and also in the .bat file for Windows

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

compile_compute() {
   l6_compiler="$1"
   l6_compiler_args_sanitized="$2"
   l6_src_path_sanitized="$3"
   l6_src_path_unsanitized="$4"
   l6_obj_path_unsanitized="$5"
   l6_asm="$6"
   l6_zone="$7"

   compile_directory_cpp "$l6_compiler" "$l6_compiler_args_sanitized -DZONE_$l6_zone" "$l6_src_path_unsanitized/compute" "$l6_obj_path_unsanitized" "$l6_asm" "$l6_zone"
   compile_directory_cpp "$l6_compiler" "$l6_compiler_args_sanitized -I$l6_src_path_sanitized/compute/${l6_zone}_ebm -DZONE_$l6_zone" "$l6_src_path_unsanitized/compute/${l6_zone}_ebm" "$l6_obj_path_unsanitized" "$l6_asm" "$l6_zone"
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

copy_bin_files() {
   l8_bin_path_unsanitized="$1"
   l8_bin_file="$2"
   l8_python_lib_unsanitized="$3"
   l8_staging_path_unsanitized="$4"

   cp "$l8_bin_path_unsanitized/$l8_bin_file" "$l8_python_lib_unsanitized/"
   l8_ret_code=$?
   if [ $l8_ret_code -ne 0 ]; then 
      exit $l8_ret_code
   fi
   cp "$l8_bin_path_unsanitized/$l8_bin_file" "$l8_staging_path_unsanitized/"
   l8_ret_code=$?
   if [ $l8_ret_code -ne 0 ]; then 
      exit $l8_ret_code
   fi
}

copy_asm_files() {
   l9_obj_path_unsanitized="$1"
   l9_tmp_path_unsanitized="$2"
   l9_bin_file_unsanitized="$3" 
   l9_staging_tag="$4"
   l9_asm="$5"

   if [ $l9_asm -ne 0 ]; then 
      l9_tagged_path_unsanitized="$l9_tmp_path_unsanitized/staging_$l9_staging_tag"

      [ -d "$l9_tagged_path_unsanitized" ] || mkdir -p "$l9_tagged_path_unsanitized"
      l9_ret_code=$?
      if [ $l9_ret_code -ne 0 ]; then 
         exit $l9_ret_code
      fi

      cp "$l9_obj_path_unsanitized"/*.s "$l9_tagged_path_unsanitized/"
      l9_ret_code=$?
      if [ $l9_ret_code -ne 0 ]; then 
         exit $l9_ret_code
      fi

      #also generate a disassembly from the final output that we can compare the individual files against
      l9_bin_file_body_unsanitized=`get_file_body "$l9_bin_file_unsanitized"`
      os_type=`uname`
      if [ "$os_type" = "Linux" ]; then
         # - https://stackoverflow.com/questions/1289881/using-gcc-to-produce-readable-assembly
         # GNU objdump https://linux.die.net/man/1/objdump
         objdump --disassemble --private-headers --reloc --dynamic-reloc --section-headers --syms --line-numbers --no-show-raw-insn --source "$l9_bin_file_unsanitized" > "$l9_tagged_path_unsanitized/$l9_bin_file_body_unsanitized.s"
      elif [ "$os_type" = "Darwin" ]; then
         # objdump on mac is actually llvm-objdump
         # https://llvm.org/docs/CommandGuide/llvm-objdump.html
         # otool might be a better choice on mac, but this does what we need in combination with the individual 
         # module assembly, so keep it consistent with linux unless we need something more in the future
         objdump --disassemble --private-headers --reloc --dynamic-reloc --section-headers --syms --line-numbers --no-show-raw-insn --source --print-imm-hex "$l9_bin_file_unsanitized" > "$l9_tagged_path_unsanitized/$l9_bin_file_body_unsanitized.s"
      else
         exit 1
      fi
   fi
}


release_64=1
debug_64=1
release_32=0
debug_32=0

is_asm=0
is_extra_debugging=0

for arg in "$@"; do
   if [ "$arg" = "-no_release_64" ]; then
      release_64=0
   fi
   if [ "$arg" = "-no_debug_64" ]; then
      debug_64=0
   fi
   if [ "$arg" = "-release_32" ]; then
      release_32=1
   fi
   if [ "$arg" = "-debug_32" ]; then
      debug_32=1
   fi
   if [ "$arg" = "-asm" ]; then
      is_asm=1
   fi
   if [ "$arg" = "-extra_debugging" ]; then
      is_extra_debugging=1
   fi
done

# TODO: this could be improved upon.  There is no perfect solution AFAIK for getting the script directory, and I'm not too sure how the CDPATH thing works
# Look at BASH_SOURCE[0] as well and possibly select either it or $0
# The output here needs to not be the empty string for glob substitution below:
script_path_initial=`dirname -- "$0"`
# the space after the '= ' character is required
script_path_unsanitized=`CDPATH= cd -- "$script_path_initial" && pwd -P`
if [ ! -f "$script_path_unsanitized/build.sh" ] ; then
   # there are all kinds of reasons why we might not have gotten the script path in $0.  It's more of a convention
   # than a requirement to have either the full path or even the script itself.  There are far more complicated
   # scripts out there that attempt to use various shell specific workarounds, like BASH_SOURCE[0] to best solve
   # the problem, but it's possible in theory to be running over an SSL connection without a script on the local
   # system at all, so getting the directory is a fundamentally unsolved problem.  We can terminate though if
   # we find ourselves in such a weird condition.  This also happens when the "source" command is used.
   printf "Could not find script file root directory for building InterpretML.  Exiting."
   exit 1
fi

root_path_unsanitized="$script_path_unsanitized"
tmp_path_unsanitized="$root_path_unsanitized/tmp"
python_lib_unsanitized="$root_path_unsanitized/python/interpret-core/interpret/lib"
staging_path_unsanitized="$root_path_unsanitized/staging"
src_path_unsanitized="$root_path_unsanitized/shared/ebm_native"
src_path_sanitized=`sanitize "$src_path_unsanitized"`


# a good referenece on writing shared libraries is at: https://akkadia.org/drepper/dsohowto.pdf

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
# TODO: once we have highly efficient tightly looped code, try no -fpic and see if that makes better code.  The compiler can save a register in this case. See https://akkadia.org/drepper/dsohowto.pdf
both_args="$both_args -fpic"
both_args="$both_args -pthread"
both_args="$both_args -DEBM_NATIVE_EXPORTS"
if [ $is_extra_debugging -ne 0 ]; then 
   both_args="$both_args -g"
fi


c_args="-std=c99"

cpp_args="-std=c++11"
cpp_args="$cpp_args -Wold-style-cast"
cpp_args="$cpp_args -fvisibility-inlines-hidden"

common_args="-I$src_path_sanitized/inc"
common_args="$common_args -I$src_path_sanitized/common_c"
common_args="$common_args -I$src_path_sanitized/common_cpp"

bridge_args="$common_args"
bridge_args="$bridge_args -I$src_path_sanitized/bridge_c"
bridge_args="$bridge_args -I$src_path_sanitized/bridge_cpp"

main_args="$bridge_args"
main_args="$main_args -I$src_path_sanitized"

compute_args="$bridge_args"
compute_args="$compute_args -I$src_path_sanitized/compute"
compute_args="$compute_args -I$src_path_sanitized/compute/loss_functions"
compute_args="$compute_args -I$src_path_sanitized/compute/metrics"

# add any other non-include options
common_args="$common_args -Wno-format-nonliteral"

link_args=""

os_type=`uname`

if [ "$os_type" = "Linux" ]; then
   c_compiler=gcc
   cpp_compiler=g++

   # try moving some of these g++ specific warnings into both_args if clang eventually supports them
   both_args="$both_args -Wlogical-op"
   both_args="$both_args -march=core2"

   # the linker wants to have the most dependent .o/.so/.dylib files listed FIRST
   link_args="$link_args -Wl,--version-script=$src_path_sanitized/ebm_native_exports.txt"
   link_args="$link_args -Wl,--exclude-libs,ALL"
   link_args="$link_args -Wl,-z,relro,-z,now"
   link_args="$link_args -static-libgcc"
   link_args="$link_args -static-libstdc++"
   link_args="$link_args -shared"

   printf "%s\n" "Creating initial directories"
   [ -d "$staging_path_unsanitized" ] || mkdir -p "$staging_path_unsanitized"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   [ -d "$python_lib_unsanitized" ] || mkdir -p "$python_lib_unsanitized"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi

   if [ $release_64 -eq 1 ]; then
      ########################## Linux release|x64

      printf "%s\n" "Compiling ebm_native with $c_compiler/$cpp_compiler for Linux release|x64"
      obj_path_unsanitized="$tmp_path_unsanitized/gcc/obj/release/linux/x64/ebm_native"
      bin_path_unsanitized="$tmp_path_unsanitized/gcc/bin/release/linux/x64/ebm_native"
      bin_file="lib_ebm_native_linux_x64.so"
      g_log_file_unsanitized="$obj_path_unsanitized/ebm_native_release_linux_x64_build_log.txt"
      both_args_extra="-m64 -DNDEBUG -O3 -Wl,--wrap=memcpy -Wl,--wrap=exp -Wl,--wrap=log -Wl,--wrap=log2,--wrap=pow"
      c_args_specific="$c_args $both_args $both_args_extra"
      cpp_args_specific="$cpp_args $both_args $both_args_extra"
      # the linker wants to have the most dependent .o/.so/.dylib files listed FIRST
      link_args_specific="$link_args $cpp_args_specific"
   
      g_all_object_files_sanitized=""
      g_compile_out_full=""

      make_initial_paths_simple "$obj_path_unsanitized" "$bin_path_unsanitized"
      compile_directory_c "$c_compiler" "$c_args_specific $common_args" "$src_path_unsanitized/common_c" "$obj_path_unsanitized" "$is_asm" "C"
      compile_directory_c "$c_compiler" "$c_args_specific $bridge_args" "$src_path_unsanitized/bridge_c" "$obj_path_unsanitized" "$is_asm" "C"
      compile_directory_cpp "$cpp_compiler" "$cpp_args_specific $main_args -DZONE_main" "$src_path_unsanitized" "$obj_path_unsanitized" "$is_asm" "main"
      compile_compute "$cpp_compiler" "$cpp_args_specific $compute_args" "$src_path_sanitized" "$src_path_unsanitized" "$obj_path_unsanitized" "$is_asm" "cpu"
      compile_compute "$cpp_compiler" "$cpp_args_specific $compute_args" "$src_path_sanitized" "$src_path_unsanitized" "$obj_path_unsanitized" "$is_asm" "avx512"
      compile_file "$cpp_compiler" "$cpp_args_specific" "$src_path_unsanitized"/special/linux_wrap_functions.cpp "$obj_path_unsanitized" "$is_asm" "NONE"
      link_file "$cpp_compiler" "$link_args_specific" "$bin_path_unsanitized" "$bin_file"
      printf "%s\n" "$g_compile_out_full"
      printf "%s\n" "$g_compile_out_full" > "$g_log_file_unsanitized"
      copy_bin_files "$bin_path_unsanitized" "$bin_file" "$python_lib_unsanitized" "$staging_path_unsanitized"
      copy_asm_files "$obj_path_unsanitized" "$tmp_path_unsanitized" "$staging_path_unsanitized/$bin_file" "asm_release_64" "$is_asm"
   fi

   if [ $debug_64 -eq 1 ]; then
      ########################## Linux debug|x64

      printf "%s\n" "Compiling ebm_native with $c_compiler/$cpp_compiler for Linux debug|x64"
      obj_path_unsanitized="$tmp_path_unsanitized/gcc/obj/debug/linux/x64/ebm_native"
      bin_path_unsanitized="$tmp_path_unsanitized/gcc/bin/debug/linux/x64/ebm_native"
      bin_file="lib_ebm_native_linux_x64_debug.so"
      g_log_file_unsanitized="$obj_path_unsanitized/ebm_native_debug_linux_x64_build_log.txt"
      both_args_extra="-m64 -O1 -Wl,--wrap=memcpy -Wl,--wrap=exp -Wl,--wrap=log -Wl,--wrap=log2,--wrap=pow"
      c_args_specific="$c_args $both_args $both_args_extra"
      cpp_args_specific="$cpp_args $both_args $both_args_extra"
      # the linker wants to have the most dependent .o/.so/.dylib files listed FIRST
      link_args_specific="$link_args $cpp_args_specific"
   
      g_all_object_files_sanitized=""
      g_compile_out_full=""

      make_initial_paths_simple "$obj_path_unsanitized" "$bin_path_unsanitized"
      compile_directory_c "$c_compiler" "$c_args_specific $common_args" "$src_path_unsanitized/common_c" "$obj_path_unsanitized" 0 "C"
      compile_directory_c "$c_compiler" "$c_args_specific $bridge_args" "$src_path_unsanitized/bridge_c" "$obj_path_unsanitized" 0 "C"
      compile_directory_cpp "$cpp_compiler" "$cpp_args_specific $main_args -DZONE_main" "$src_path_unsanitized" "$obj_path_unsanitized" 0 "main"
      compile_compute "$cpp_compiler" "$cpp_args_specific $compute_args" "$src_path_sanitized" "$src_path_unsanitized" "$obj_path_unsanitized" 0 "cpu"
      compile_compute "$cpp_compiler" "$cpp_args_specific $compute_args" "$src_path_sanitized" "$src_path_unsanitized" "$obj_path_unsanitized" 0 "avx512"
      compile_file "$cpp_compiler" "$cpp_args_specific" "$src_path_unsanitized"/special/linux_wrap_functions.cpp "$obj_path_unsanitized" 0 "NONE"
      link_file "$cpp_compiler" "$link_args_specific" "$bin_path_unsanitized" "$bin_file"
      printf "%s\n" "$g_compile_out_full"
      printf "%s\n" "$g_compile_out_full" > "$g_log_file_unsanitized"
      copy_bin_files "$bin_path_unsanitized" "$bin_file" "$python_lib_unsanitized" "$staging_path_unsanitized"
   fi

   if [ $release_32 -eq 1 ]; then
      ########################## Linux release|x86

      printf "%s\n" "Compiling ebm_native with $c_compiler/$cpp_compiler for Linux release|x86"
      obj_path_unsanitized="$tmp_path_unsanitized/gcc/obj/release/linux/x86/ebm_native"
      bin_path_unsanitized="$tmp_path_unsanitized/gcc/bin/release/linux/x86/ebm_native"
      bin_file="lib_ebm_native_linux_x86.so"
      g_log_file_unsanitized="$obj_path_unsanitized/ebm_native_release_linux_x86_build_log.txt"
      both_args_extra="-msse2 -mfpmath=sse -m32 -DNDEBUG -O3"
      c_args_specific="$c_args $both_args $both_args_extra"
      cpp_args_specific="$cpp_args $both_args $both_args_extra"
      # the linker wants to have the most dependent .o/.so/.dylib files listed FIRST
      link_args_specific="$link_args $cpp_args_specific"
      
      g_all_object_files_sanitized=""
      g_compile_out_full=""

      make_initial_paths_simple "$obj_path_unsanitized" "$bin_path_unsanitized"
      check_install "$tmp_path_unsanitized" "g++-multilib"
      compile_directory_c "$c_compiler" "$c_args_specific $common_args" "$src_path_unsanitized/common_c" "$obj_path_unsanitized" 0 "C"
      compile_directory_c "$c_compiler" "$c_args_specific $bridge_args" "$src_path_unsanitized/bridge_c" "$obj_path_unsanitized" 0 "C"
      compile_directory_cpp "$cpp_compiler" "$cpp_args_specific $main_args -DZONE_main" "$src_path_unsanitized" "$obj_path_unsanitized" 0 "main"
      compile_compute "$cpp_compiler" "$cpp_args_specific $compute_args" "$src_path_sanitized" "$src_path_unsanitized" "$obj_path_unsanitized" 0 "cpu"
      compile_compute "$cpp_compiler" "$cpp_args_specific $compute_args" "$src_path_sanitized" "$src_path_unsanitized" "$obj_path_unsanitized" 0 "avx512"
      compile_file "$cpp_compiler" "$cpp_args_specific" "$src_path_unsanitized"/special/linux_wrap_functions.cpp "$obj_path_unsanitized" 0 "NONE"
      link_file "$cpp_compiler" "$link_args_specific" "$bin_path_unsanitized" "$bin_file"
      printf "%s\n" "$g_compile_out_full"
      printf "%s\n" "$g_compile_out_full" > "$g_log_file_unsanitized"
      copy_bin_files "$bin_path_unsanitized" "$bin_file" "$python_lib_unsanitized" "$staging_path_unsanitized"
   fi

   if [ $debug_32 -eq 1 ]; then
      ########################## Linux debug|x86

      printf "%s\n" "Compiling ebm_native with $c_compiler/$cpp_compiler for Linux debug|x86"
      obj_path_unsanitized="$tmp_path_unsanitized/gcc/obj/debug/linux/x86/ebm_native"
      bin_path_unsanitized="$tmp_path_unsanitized/gcc/bin/debug/linux/x86/ebm_native"
      bin_file="lib_ebm_native_linux_x86_debug.so"
      g_log_file_unsanitized="$obj_path_unsanitized/ebm_native_debug_linux_x86_build_log.txt"
      both_args_extra="-msse2 -mfpmath=sse -m32 -O1"
      c_args_specific="$c_args $both_args $both_args_extra"
      cpp_args_specific="$cpp_args $both_args $both_args_extra"
      # the linker wants to have the most dependent .o/.so/.dylib files listed FIRST
      link_args_specific="$link_args $cpp_args_specific"
      
      g_all_object_files_sanitized=""
      g_compile_out_full=""

      make_initial_paths_simple "$obj_path_unsanitized" "$bin_path_unsanitized"
      check_install "$tmp_path_unsanitized" "g++-multilib"
      compile_directory_c "$c_compiler" "$c_args_specific $common_args" "$src_path_unsanitized/common_c" "$obj_path_unsanitized" 0 "C"
      compile_directory_c "$c_compiler" "$c_args_specific $bridge_args" "$src_path_unsanitized/bridge_c" "$obj_path_unsanitized" 0 "C"
      compile_directory_cpp "$cpp_compiler" "$cpp_args_specific $main_args -DZONE_main" "$src_path_unsanitized" "$obj_path_unsanitized" 0 "main"
      compile_compute "$cpp_compiler" "$cpp_args_specific $compute_args" "$src_path_sanitized" "$src_path_unsanitized" "$obj_path_unsanitized" 0 "cpu"
      compile_compute "$cpp_compiler" "$cpp_args_specific $compute_args" "$src_path_sanitized" "$src_path_unsanitized" "$obj_path_unsanitized" 0 "avx512"
      compile_file "$cpp_compiler" "$cpp_args_specific" "$src_path_unsanitized"/special/linux_wrap_functions.cpp "$obj_path_unsanitized" 0 "NONE"
      link_file "$cpp_compiler" "$link_args_specific" "$bin_path_unsanitized" "$bin_file"
      printf "%s\n" "$g_compile_out_full"
      printf "%s\n" "$g_compile_out_full" > "$g_log_file_unsanitized"
      copy_bin_files "$bin_path_unsanitized" "$bin_file" "$python_lib_unsanitized" "$staging_path_unsanitized"
   fi

elif [ "$os_type" = "Darwin" ]; then
   # reference on rpath & install_name: https://www.mikeash.com/pyblog/friday-qa-2009-11-06-linking-and-install-names.html

   # TODO: make these real options instead of forcing them to the same as x64
   release_arm=$release_64
   debug_arm=$debug_64

   # try moving some of these clang specific warnings into both_args if g++ eventually supports them
   c_compiler=clang
   cpp_compiler=clang++
   both_args="$both_args -Wnull-dereference"
   both_args="$both_args -Wgnu-zero-variadic-macro-arguments"

   # the linker wants to have the most dependent .o/.so/.dylib files listed FIRST
   link_args="$link_args -dynamiclib"

   printf "%s\n" "Creating initial directories"
   [ -d "$staging_path_unsanitized" ] || mkdir -p "$staging_path_unsanitized"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   [ -d "$python_lib_unsanitized" ] || mkdir -p "$python_lib_unsanitized"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi

   if [ $release_64 -eq 1 ]; then
      ########################## macOS release|x64

      printf "%s\n" "Compiling ebm_native with $c_compiler/$cpp_compiler for macOS release|x64"
      obj_path_unsanitized="$tmp_path_unsanitized/clang/obj/release/mac/x64/ebm_native"
      bin_path_unsanitized="$tmp_path_unsanitized/clang/bin/release/mac/x64/ebm_native"
      bin_file="lib_ebm_native_mac_x64.dylib"
      g_log_file_unsanitized="$obj_path_unsanitized/ebm_native_release_mac_x64_build_log.txt"
      both_args_extra="-march=core2 -target x86_64-apple-macos10.12 -m64 -DNDEBUG -O3"
      c_args_specific="$c_args $both_args $both_args_extra"
      cpp_args_specific="$cpp_args $both_args $both_args_extra"
      # the linker wants to have the most dependent .o/.so/.dylib files listed FIRST
      link_args_specific="-install_name @rpath/$bin_file $link_args $cpp_args_specific"
   
      g_all_object_files_sanitized=""
      g_compile_out_full=""

      make_initial_paths_simple "$obj_path_unsanitized" "$bin_path_unsanitized"
      compile_directory_c "$c_compiler" "$c_args_specific $common_args" "$src_path_unsanitized/common_c" "$obj_path_unsanitized" "$is_asm" "C"
      compile_directory_c "$c_compiler" "$c_args_specific $bridge_args" "$src_path_unsanitized/bridge_c" "$obj_path_unsanitized" "$is_asm" "C"
      compile_directory_cpp "$cpp_compiler" "$cpp_args_specific $main_args -DZONE_main" "$src_path_unsanitized" "$obj_path_unsanitized" "$is_asm" "main"
      compile_compute "$cpp_compiler" "$cpp_args_specific $compute_args" "$src_path_sanitized" "$src_path_unsanitized" "$obj_path_unsanitized" "$is_asm" "cpu"
      compile_compute "$cpp_compiler" "$cpp_args_specific $compute_args" "$src_path_sanitized" "$src_path_unsanitized" "$obj_path_unsanitized" "$is_asm" "avx512"
      link_file "$cpp_compiler" "$link_args_specific" "$bin_path_unsanitized" "$bin_file"
      printf "%s\n" "$g_compile_out_full"
      printf "%s\n" "$g_compile_out_full" > "$g_log_file_unsanitized"
      copy_bin_files "$bin_path_unsanitized" "$bin_file" "$python_lib_unsanitized" "$staging_path_unsanitized"
      copy_asm_files "$obj_path_unsanitized" "$tmp_path_unsanitized" "$staging_path_unsanitized/$bin_file" "asm_release_64" "$is_asm"
   fi

   if [ $debug_64 -eq 1 ]; then
      ########################## macOS debug|x64

      printf "%s\n" "Compiling ebm_native with $c_compiler/$cpp_compiler for macOS debug|x64"
      obj_path_unsanitized="$tmp_path_unsanitized/clang/obj/debug/mac/x64/ebm_native"
      bin_path_unsanitized="$tmp_path_unsanitized/clang/bin/debug/mac/x64/ebm_native"
      bin_file="lib_ebm_native_mac_x64_debug.dylib"
      g_log_file_unsanitized="$obj_path_unsanitized/ebm_native_debug_mac_x64_build_log.txt"
      both_args_extra="-march=core2 -target x86_64-apple-macos10.12 -m64 -O1 -fsanitize=address,undefined -fno-sanitize-recover=address,undefined -fno-optimize-sibling-calls -fno-omit-frame-pointer"
      c_args_specific="$c_args $both_args $both_args_extra"
      cpp_args_specific="$cpp_args $both_args $both_args_extra"
      # the linker wants to have the most dependent .o/.so/.dylib files listed FIRST
      link_args_specific="-install_name @rpath/$bin_file $link_args $cpp_args_specific"
   
      g_all_object_files_sanitized=""
      g_compile_out_full=""

      make_initial_paths_simple "$obj_path_unsanitized" "$bin_path_unsanitized"
      compile_directory_c "$c_compiler" "$c_args_specific $common_args" "$src_path_unsanitized/common_c" "$obj_path_unsanitized" 0 "C"
      compile_directory_c "$c_compiler" "$c_args_specific $bridge_args" "$src_path_unsanitized/bridge_c" "$obj_path_unsanitized" 0 "C"
      compile_directory_cpp "$cpp_compiler" "$cpp_args_specific $main_args -DZONE_main" "$src_path_unsanitized" "$obj_path_unsanitized" 0 "main"
      compile_compute "$cpp_compiler" "$cpp_args_specific $compute_args" "$src_path_sanitized" "$src_path_unsanitized" "$obj_path_unsanitized" 0 "cpu"
      compile_compute "$cpp_compiler" "$cpp_args_specific $compute_args" "$src_path_sanitized" "$src_path_unsanitized" "$obj_path_unsanitized" 0 "avx512"
      link_file "$cpp_compiler" "$link_args_specific" "$bin_path_unsanitized" "$bin_file"
      printf "%s\n" "$g_compile_out_full"
      printf "%s\n" "$g_compile_out_full" > "$g_log_file_unsanitized"
      copy_bin_files "$bin_path_unsanitized" "$bin_file" "$python_lib_unsanitized" "$staging_path_unsanitized"
   fi

   if [ $release_arm -eq 1 ]; then
      ########################## macOS release|arm

      printf "%s\n" "Compiling ebm_native with $c_compiler/$cpp_compiler for macOS release|arm"
      obj_path_unsanitized="$tmp_path_unsanitized/clang/obj/release/mac/arm/ebm_native"
      bin_path_unsanitized="$tmp_path_unsanitized/clang/bin/release/mac/arm/ebm_native"
      bin_file="lib_ebm_native_mac_arm.dylib"
      g_log_file_unsanitized="$obj_path_unsanitized/ebm_native_release_mac_arm_build_log.txt"
      both_args_extra="-target arm64-apple-macos11 -m64 -DNDEBUG -O3"
      c_args_specific="$c_args $both_args $both_args_extra"
      cpp_args_specific="$cpp_args $both_args $both_args_extra"
      # the linker wants to have the most dependent .o/.so/.dylib files listed FIRST
      link_args_specific="-install_name @rpath/$bin_file $link_args $cpp_args_specific"
   
      g_all_object_files_sanitized=""
      g_compile_out_full=""

      make_initial_paths_simple "$obj_path_unsanitized" "$bin_path_unsanitized"
      compile_directory_c "$c_compiler" "$c_args_specific $common_args" "$src_path_unsanitized/common_c" "$obj_path_unsanitized" "$is_asm" "C"
      compile_directory_c "$c_compiler" "$c_args_specific $bridge_args" "$src_path_unsanitized/bridge_c" "$obj_path_unsanitized" "$is_asm" "C"
      compile_directory_cpp "$cpp_compiler" "$cpp_args_specific $main_args -DZONE_main" "$src_path_unsanitized" "$obj_path_unsanitized" "$is_asm" "main"
      compile_compute "$cpp_compiler" "$cpp_args_specific $compute_args" "$src_path_sanitized" "$src_path_unsanitized" "$obj_path_unsanitized" "$is_asm" "cpu"
      link_file "$cpp_compiler" "$link_args_specific" "$bin_path_unsanitized" "$bin_file"
      printf "%s\n" "$g_compile_out_full"
      printf "%s\n" "$g_compile_out_full" > "$g_log_file_unsanitized"
      copy_bin_files "$bin_path_unsanitized" "$bin_file" "$python_lib_unsanitized" "$staging_path_unsanitized"
      copy_asm_files "$obj_path_unsanitized" "$tmp_path_unsanitized" "$staging_path_unsanitized/$bin_file" "asm_release_arm" "$is_asm"
   fi

   if [ $debug_arm -eq 1 ]; then
      ########################## macOS debug|arm

      printf "%s\n" "Compiling ebm_native with $c_compiler/$cpp_compiler for macOS debug|arm"
      obj_path_unsanitized="$tmp_path_unsanitized/clang/obj/debug/mac/arm/ebm_native"
      bin_path_unsanitized="$tmp_path_unsanitized/clang/bin/debug/mac/arm/ebm_native"
      bin_file="lib_ebm_native_mac_arm_debug.dylib"
      g_log_file_unsanitized="$obj_path_unsanitized/ebm_native_debug_mac_arm_build_log.txt"
      both_args_extra="-target arm64-apple-macos11 -m64 -O1 -fsanitize=address,undefined -fno-sanitize-recover=address,undefined -fno-optimize-sibling-calls -fno-omit-frame-pointer"
      c_args_specific="$c_args $both_args $both_args_extra"
      cpp_args_specific="$cpp_args $both_args $both_args_extra"
      # the linker wants to have the most dependent .o/.so/.dylib files listed FIRST
      link_args_specific="-install_name @rpath/$bin_file $link_args $cpp_args_specific"
   
      g_all_object_files_sanitized=""
      g_compile_out_full=""

      make_initial_paths_simple "$obj_path_unsanitized" "$bin_path_unsanitized"
      compile_directory_c "$c_compiler" "$c_args_specific $common_args" "$src_path_unsanitized/common_c" "$obj_path_unsanitized" 0 "C"
      compile_directory_c "$c_compiler" "$c_args_specific $bridge_args" "$src_path_unsanitized/bridge_c" "$obj_path_unsanitized" 0 "C"
      compile_directory_cpp "$cpp_compiler" "$cpp_args_specific $main_args -DZONE_main" "$src_path_unsanitized" "$obj_path_unsanitized" 0 "main"
      compile_compute "$cpp_compiler" "$cpp_args_specific $compute_args" "$src_path_sanitized" "$src_path_unsanitized" "$obj_path_unsanitized" 0 "cpu"
      link_file "$cpp_compiler" "$link_args_specific" "$bin_path_unsanitized" "$bin_file"
      printf "%s\n" "$g_compile_out_full"
      printf "%s\n" "$g_compile_out_full" > "$g_log_file_unsanitized"
      copy_bin_files "$bin_path_unsanitized" "$bin_file" "$python_lib_unsanitized" "$staging_path_unsanitized"
   fi

else
   printf "%s\n" "OS $os_type not recognized.  We support clang/clang++ on macOS and gcc/g++ on Linux"
   exit 1
fi
