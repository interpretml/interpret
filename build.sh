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

   l3_file_sanitized=`sanitize "$l3_file_unsanitized"`
   l3_file_body_unsanitized=`get_file_body "$l3_file_unsanitized"`
   l3_object_full_file_unsanitized="$l3_obj_path_unsanitized/${l3_file_body_unsanitized}.o"
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
      l3_asm_full_file_unsanitized="$l3_obj_path_unsanitized/${l3_file_body_unsanitized}.s"
      l3_asm_full_file_sanitized=`sanitize "$l3_asm_full_file_unsanitized"`
      l3_compile_specific_asm="$l3_compiler $l3_compiler_args_sanitized -fverbose-asm -S $l3_file_sanitized -o $l3_asm_full_file_sanitized 2>&1"
      l3_compile_out_asm=`eval "$l3_compile_specific_asm"`
   fi
}

compile_directory() {
   l5_compiler="$1"
   l5_compiler_args_sanitized="$2"
   l5_src_path_unsanitized="$3"
   l5_obj_path_unsanitized="$4"
   l5_asm="$5"

   # zsh (default shell in macs) terminates if you try to glob expand zero results, so check first
   find "$l5_src_path_unsanitized" -maxdepth 1 -type f -name '*.cpp' 2>/dev/null | grep -q .
   l5_ret_code=$?
   if [ $l5_ret_code -eq 0 ]; then 
      # use globs with preceeding directory per: https://dwheeler.com/essays/filenames-in-shell.html
      for l5_file_unsanitized in "$l5_src_path_unsanitized"/*.cpp ; do
         # glob expansion returns *.cpp when there are no matches, so we need to check for the existance of the file
         if [ -f "$l5_file_unsanitized" ] ; then
            compile_file "$l5_compiler" "$l5_compiler_args_sanitized" "$l5_file_unsanitized" "$l5_obj_path_unsanitized" "$l5_asm"
         fi
      done
   fi
}

compile_compute() {
   l6_compiler="$1"
   l6_compiler_args_sanitized="$2"
   l6_src_path_unsanitized="$3"
   l6_obj_path_unsanitized="$4"
   l6_asm="$5"
   l6_zone="$6"

   compile_directory "$l6_compiler" "$l6_compiler_args_sanitized" "$l6_src_path_unsanitized/compute/${l6_zone}_ebm" "$l6_obj_path_unsanitized" "$l6_asm"
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

if [ -n "${CXX}" ]; then
   code_path="./shared/libebm"
   tmp_path="./tmp/mk"

   os_type=`uname`
   # TODO: change this to accept libebm_local.so or libebm_local.dylib to allow for weird architectures build using sdists
   if [ "$os_type" = "Linux" ]; then
      final_binary="./python/interpret-core/interpret/lib/libebm_linux_x64.so"
   elif [ "$os_type" = "Darwin" ]; then
      final_binary="./python/interpret-core/interpret/lib/libebm_mac_x64.dylib"
   else
      printf "%s\n" "OS $os_type not recognized.  We support clang/clang++ on macOS and gcc/g++ on Linux"
      exit 1
   fi

   mkdir ./python
   mkdir ./python/interpret-core
   mkdir ./python/interpret-core/interpret
   mkdir ./python/interpret-core/interpret/lib

   extras="-DLIBEBM_EXPORTS -DNDEBUG -I$code_path/inc -I$code_path/unzoned -I$code_path/bridge -I$code_path -I$code_path/compute -I$code_path/compute/objectives -I$code_path/compute/metrics"

   mkdir ./tmp
   mkdir ./tmp/mk
   mkdir ./staging

   printf "Building from environment specified compiler\n"
   printf "%s\n" "CXX=${CXX}"
   printf "%s\n" "CPPFLAGS=${CPPFLAGS}"
   printf "%s\n" "CXXFLAGS=${CXXFLAGS}"

   printf "%s\n" "LDFLAGS=${LDFLAGS}"
   printf "%s\n" "LOADLIBES=${LOADLIBES}"
   printf "%s\n" "LDLIBS=${LDLIBS}"

   ${CXX} -c ${CPPFLAGS} ${CXXFLAGS} ${extras} "$code_path/ApplyTermUpdate.cpp" -o "$tmp_path/ApplyTermUpdate.o"
   ${CXX} -c ${CPPFLAGS} ${CXXFLAGS} ${extras} "$code_path/BoosterCore.cpp" -o "$tmp_path/BoosterCore.o"
   ${CXX} -c ${CPPFLAGS} ${CXXFLAGS} ${extras} "$code_path/BoosterShell.cpp" -o "$tmp_path/BoosterShell.o"
   ${CXX} -c ${CPPFLAGS} ${CXXFLAGS} ${extras} "$code_path/CalcInteractionStrength.cpp" -o "$tmp_path/CalcInteractionStrength.o"
   ${CXX} -c ${CPPFLAGS} ${CXXFLAGS} ${extras} "$code_path/compute_accessors.cpp" -o "$tmp_path/compute_accessors.o"
   ${CXX} -c ${CPPFLAGS} ${CXXFLAGS} ${extras} "$code_path/ConvertAddBin.cpp" -o "$tmp_path/ConvertAddBin.o"
   ${CXX} -c ${CPPFLAGS} ${CXXFLAGS} ${extras} "$code_path/CutQuantile.cpp" -o "$tmp_path/CutQuantile.o"
   ${CXX} -c ${CPPFLAGS} ${CXXFLAGS} ${extras} "$code_path/CutUniform.cpp" -o "$tmp_path/CutUniform.o"
   ${CXX} -c ${CPPFLAGS} ${CXXFLAGS} ${extras} "$code_path/CutWinsorized.cpp" -o "$tmp_path/CutWinsorized.o"
   ${CXX} -c ${CPPFLAGS} ${CXXFLAGS} ${extras} "$code_path/dataset_shared.cpp" -o "$tmp_path/dataset_shared.o"
   ${CXX} -c ${CPPFLAGS} ${CXXFLAGS} ${extras} "$code_path/DataSetBoosting.cpp" -o "$tmp_path/DataSetBoosting.o"
   ${CXX} -c ${CPPFLAGS} ${CXXFLAGS} ${extras} "$code_path/DataSetInteraction.cpp" -o "$tmp_path/DataSetInteraction.o"
   ${CXX} -c ${CPPFLAGS} ${CXXFLAGS} ${extras} "$code_path/DetermineLinkFunction.cpp" -o "$tmp_path/DetermineLinkFunction.o"
   ${CXX} -c ${CPPFLAGS} ${CXXFLAGS} ${extras} "$code_path/debug_ebm.cpp" -o "$tmp_path/debug_ebm.o"
   ${CXX} -c ${CPPFLAGS} ${CXXFLAGS} ${extras} "$code_path/Discretize.cpp" -o "$tmp_path/Discretize.o"
   ${CXX} -c ${CPPFLAGS} ${CXXFLAGS} ${extras} "$code_path/Term.cpp" -o "$tmp_path/Term.o"
   ${CXX} -c ${CPPFLAGS} ${CXXFLAGS} ${extras} "$code_path/GenerateTermUpdate.cpp" -o "$tmp_path/GenerateTermUpdate.o"
   ${CXX} -c ${CPPFLAGS} ${CXXFLAGS} ${extras} "$code_path/InitializeGradientsAndHessians.cpp" -o "$tmp_path/InitializeGradientsAndHessians.o"
   ${CXX} -c ${CPPFLAGS} ${CXXFLAGS} ${extras} "$code_path/InteractionCore.cpp" -o "$tmp_path/InteractionCore.o"
   ${CXX} -c ${CPPFLAGS} ${CXXFLAGS} ${extras} "$code_path/InteractionShell.cpp" -o "$tmp_path/InteractionShell.o"
   ${CXX} -c ${CPPFLAGS} ${CXXFLAGS} ${extras} "$code_path/interpretable_numerics.cpp" -o "$tmp_path/interpretable_numerics.o"
   ${CXX} -c ${CPPFLAGS} ${CXXFLAGS} ${extras} "$code_path/PartitionOneDimensionalBoosting.cpp" -o "$tmp_path/PartitionOneDimensionalBoosting.o"
   ${CXX} -c ${CPPFLAGS} ${CXXFLAGS} ${extras} "$code_path/PartitionRandomBoosting.cpp" -o "$tmp_path/PartitionRandomBoosting.o"
   ${CXX} -c ${CPPFLAGS} ${CXXFLAGS} ${extras} "$code_path/PartitionTwoDimensionalBoosting.cpp" -o "$tmp_path/PartitionTwoDimensionalBoosting.o"
   ${CXX} -c ${CPPFLAGS} ${CXXFLAGS} ${extras} "$code_path/PartitionTwoDimensionalInteraction.cpp" -o "$tmp_path/PartitionTwoDimensionalInteraction.o"
   ${CXX} -c ${CPPFLAGS} ${CXXFLAGS} ${extras} "$code_path/RandomDeterministic.cpp" -o "$tmp_path/RandomDeterministic.o"
   ${CXX} -c ${CPPFLAGS} ${CXXFLAGS} ${extras} "$code_path/random.cpp" -o "$tmp_path/random.o"
   ${CXX} -c ${CPPFLAGS} ${CXXFLAGS} ${extras} "$code_path/sampling.cpp" -o "$tmp_path/sampling.o"
   ${CXX} -c ${CPPFLAGS} ${CXXFLAGS} ${extras} "$code_path/InnerBag.cpp" -o "$tmp_path/InnerBag.o"
   ${CXX} -c ${CPPFLAGS} ${CXXFLAGS} ${extras} "$code_path/Tensor.cpp" -o "$tmp_path/Tensor.o"
   ${CXX} -c ${CPPFLAGS} ${CXXFLAGS} ${extras} "$code_path/TensorTotalsBuild.cpp" -o "$tmp_path/TensorTotalsBuild.o"
   ${CXX} -c ${CPPFLAGS} ${CXXFLAGS} ${extras} "$code_path/unzoned/logging.cpp" -o "$tmp_path/logging.o"
   ${CXX} -c ${CPPFLAGS} ${CXXFLAGS} ${extras} "$code_path/unzoned/unzoned.cpp" -o "$tmp_path/unzoned.o"
   ${CXX} -c ${CPPFLAGS} ${CXXFLAGS} ${extras} "$code_path/compute/cpu_ebm/cpu_64.cpp" -o "$tmp_path/cpu_64.o"

   ${CXX} ${LDFLAGS} -shared \
   "$tmp_path/ApplyTermUpdate.o" \
   "$tmp_path/BoosterCore.o" \
   "$tmp_path/BoosterShell.o" \
   "$tmp_path/CalcInteractionStrength.o" \
   "$tmp_path/compute_accessors.o" \
   "$tmp_path/ConvertAddBin.o" \
   "$tmp_path/CutQuantile.o" \
   "$tmp_path/CutUniform.o" \
   "$tmp_path/CutWinsorized.o" \
   "$tmp_path/dataset_shared.o" \
   "$tmp_path/DataSetBoosting.o" \
   "$tmp_path/DataSetInteraction.o" \
   "$tmp_path/DetermineLinkFunction.o" \
   "$tmp_path/debug_ebm.o" \
   "$tmp_path/Discretize.o" \
   "$tmp_path/Term.o" \
   "$tmp_path/GenerateTermUpdate.o" \
   "$tmp_path/InitializeGradientsAndHessians.o" \
   "$tmp_path/InteractionCore.o" \
   "$tmp_path/InteractionShell.o" \
   "$tmp_path/interpretable_numerics.o" \
   "$tmp_path/PartitionOneDimensionalBoosting.o" \
   "$tmp_path/PartitionRandomBoosting.o" \
   "$tmp_path/PartitionTwoDimensionalBoosting.o" \
   "$tmp_path/PartitionTwoDimensionalInteraction.o" \
   "$tmp_path/RandomDeterministic.o" \
   "$tmp_path/random.o" \
   "$tmp_path/sampling.o" \
   "$tmp_path/InnerBag.o" \
   "$tmp_path/Tensor.o" \
   "$tmp_path/TensorTotalsBuild.o" \
   "$tmp_path/logging.o" \
   "$tmp_path/unzoned.o" \
   "$tmp_path/cpu_64.o" \
   ${LOADLIBES} ${LDLIBS} -o "$final_binary"

   exit 0
fi


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
src_path_unsanitized="$root_path_unsanitized/shared/libebm"
src_path_sanitized=`sanitize "$src_path_unsanitized"`


# a good referenece on writing shared libraries is at: https://akkadia.org/drepper/dsohowto.pdf

# re-enable these warnings when they are better supported by g++ or clang: -Wduplicated-cond -Wduplicated-branches -Wrestrict
all_args="-std=c++11"
all_args="$all_args -Wall -Wextra"
all_args="$all_args -Wunused-result"
all_args="$all_args -Wdouble-promotion"
all_args="$all_args -Wold-style-cast"
all_args="$all_args -Wshadow"
all_args="$all_args -Wformat=2"
all_args="$all_args -Wno-format-nonliteral"
all_args="$all_args -Wno-parentheses"
all_args="$all_args -fvisibility=hidden -fvisibility-inlines-hidden"
all_args="$all_args -fno-math-errno -fno-trapping-math"
# TODO: once we have highly efficient tightly looped code, try no -fpic and see if that makes better code.  The compiler can save a register in this case. See https://akkadia.org/drepper/dsohowto.pdf
# TODO: check no-plt compiler option
all_args="$all_args -fpic"
all_args="$all_args -pthread"
all_args="$all_args -DLIBEBM_EXPORTS"
if [ $is_extra_debugging -ne 0 ]; then 
   all_args="$all_args -g"
fi

all_args="$all_args -I$src_path_sanitized/inc"

unzoned_args=""
unzoned_args="$unzoned_args -I$src_path_sanitized/unzoned"

compute_args=""
compute_args="$compute_args -I$src_path_sanitized/unzoned"
compute_args="$compute_args -I$src_path_sanitized/bridge"
compute_args="$compute_args -I$src_path_sanitized/compute"
compute_args="$compute_args -I$src_path_sanitized/compute/objectives"
compute_args="$compute_args -I$src_path_sanitized/compute/metrics"

main_args="$main_args -I$src_path_sanitized/unzoned"
main_args="$main_args -I$src_path_sanitized/bridge"
main_args="$main_args -I$src_path_sanitized"

link_args=""

os_type=`uname`

if [ "$os_type" = "Linux" ]; then
   cpp_compiler=g++

   # try moving some of these g++ specific warnings into the shared all_args if clang eventually supports them
   all_args="$all_args -Wlogical-op"
   all_args="$all_args -march=core2"

   # the linker wants to have the most dependent .o/.so/.dylib files listed FIRST
   link_args="$link_args -Wl,--version-script=$src_path_sanitized/libebm_exports.txt"
   link_args="$link_args -Wl,--exclude-libs,ALL"
   link_args="$link_args -Wl,-z,relro,-z,now"
   link_args="$link_args -Wl,-O2"
   link_args="$link_args -Wl,--sort-common"
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

      printf "%s\n" "Compiling libebm with $cpp_compiler for Linux release|x64"
      obj_path_unsanitized="$tmp_path_unsanitized/gcc/obj/release/linux/x64/libebm"
      bin_path_unsanitized="$tmp_path_unsanitized/gcc/bin/release/linux/x64/libebm"
      bin_file="libebm_linux_x64.so"
      g_log_file_unsanitized="$obj_path_unsanitized/libebm_release_linux_x64_build_log.txt"
      specific_args="$all_args -m64 -DNDEBUG -O3 -DBRIDGE_AVX2_32 -DBRIDGE_AVX512F_32 -Wl,--wrap=memcpy -Wl,--wrap=exp -Wl,--wrap=log -Wl,--wrap=log2,--wrap=pow,--wrap=expf,--wrap=logf"
      # the linker wants to have the most dependent .o/.so/.dylib files listed FIRST
   
      g_all_object_files_sanitized=""
      g_compile_out_full=""

      make_initial_paths_simple "$obj_path_unsanitized" "$bin_path_unsanitized"
      compile_file "$cpp_compiler" "$specific_args $unzoned_args" "$src_path_unsanitized"/special/linux_wrap_functions.cpp "$obj_path_unsanitized" "$is_asm"
      compile_directory "$cpp_compiler" "$specific_args $unzoned_args" "$src_path_unsanitized/unzoned" "$obj_path_unsanitized" "$is_asm"
      compile_compute "$cpp_compiler" "$specific_args $compute_args" "$src_path_unsanitized" "$obj_path_unsanitized" "$is_asm" "cpu"
      compile_compute "$cpp_compiler" "$specific_args $compute_args -mavx2 -mfma" "$src_path_unsanitized" "$obj_path_unsanitized" "$is_asm" "avx2"
      compile_compute "$cpp_compiler" "$specific_args $compute_args -mavx512f" "$src_path_unsanitized" "$obj_path_unsanitized" "$is_asm" "avx512f"
      compile_directory "$cpp_compiler" "$specific_args $main_args" "$src_path_unsanitized" "$obj_path_unsanitized" "$is_asm"
      link_file "$cpp_compiler" "$link_args $specific_args" "$bin_path_unsanitized" "$bin_file"
      printf "%s\n" "$g_compile_out_full"
      printf "%s\n" "$g_compile_out_full" > "$g_log_file_unsanitized"
      copy_bin_files "$bin_path_unsanitized" "$bin_file" "$python_lib_unsanitized" "$staging_path_unsanitized"
      copy_asm_files "$obj_path_unsanitized" "$tmp_path_unsanitized" "$staging_path_unsanitized/$bin_file" "asm_release_64" "$is_asm"
   fi

   if [ $debug_64 -eq 1 ]; then
      ########################## Linux debug|x64

      printf "%s\n" "Compiling libebm with $cpp_compiler for Linux debug|x64"
      obj_path_unsanitized="$tmp_path_unsanitized/gcc/obj/debug/linux/x64/libebm"
      bin_path_unsanitized="$tmp_path_unsanitized/gcc/bin/debug/linux/x64/libebm"
      bin_file="libebm_linux_x64_debug.so"
      g_log_file_unsanitized="$obj_path_unsanitized/libebm_debug_linux_x64_build_log.txt"
      specific_args="$all_args -m64 -O1 -DBRIDGE_AVX2_32 -DBRIDGE_AVX512F_32 -Wl,--wrap=memcpy -Wl,--wrap=exp -Wl,--wrap=log -Wl,--wrap=log2,--wrap=pow,--wrap=expf,--wrap=logf"
      # the linker wants to have the most dependent .o/.so/.dylib files listed FIRST
   
      g_all_object_files_sanitized=""
      g_compile_out_full=""

      make_initial_paths_simple "$obj_path_unsanitized" "$bin_path_unsanitized"
      compile_file "$cpp_compiler" "$specific_args $unzoned_args" "$src_path_unsanitized"/special/linux_wrap_functions.cpp "$obj_path_unsanitized" 0
      compile_directory "$cpp_compiler" "$specific_args $unzoned_args" "$src_path_unsanitized/unzoned" "$obj_path_unsanitized" 0
      compile_compute "$cpp_compiler" "$specific_args $compute_args" "$src_path_unsanitized" "$obj_path_unsanitized" 0 "cpu"
      compile_compute "$cpp_compiler" "$specific_args $compute_args -mavx2 -mfma" "$src_path_unsanitized" "$obj_path_unsanitized" 0 "avx2"
      compile_compute "$cpp_compiler" "$specific_args $compute_args -mavx512f" "$src_path_unsanitized" "$obj_path_unsanitized" 0 "avx512f"
      compile_directory "$cpp_compiler" "$specific_args $main_args" "$src_path_unsanitized" "$obj_path_unsanitized" 0
      link_file "$cpp_compiler" "$link_args $specific_args" "$bin_path_unsanitized" "$bin_file"
      printf "%s\n" "$g_compile_out_full"
      printf "%s\n" "$g_compile_out_full" > "$g_log_file_unsanitized"
      copy_bin_files "$bin_path_unsanitized" "$bin_file" "$python_lib_unsanitized" "$staging_path_unsanitized"
   fi

   if [ $release_32 -eq 1 ]; then
      ########################## Linux release|x86

      printf "%s\n" "Compiling libebm with $cpp_compiler for Linux release|x86"
      obj_path_unsanitized="$tmp_path_unsanitized/gcc/obj/release/linux/x86/libebm"
      bin_path_unsanitized="$tmp_path_unsanitized/gcc/bin/release/linux/x86/libebm"
      bin_file="libebm_linux_x86.so"
      g_log_file_unsanitized="$obj_path_unsanitized/libebm_release_linux_x86_build_log.txt"
      specific_args="$all_args -DBRIDGE_AVX2_32 -DBRIDGE_AVX512F_32 -msse2 -mfpmath=sse -m32 -DNDEBUG -O3"
      # the linker wants to have the most dependent .o/.so/.dylib files listed FIRST
      
      g_all_object_files_sanitized=""
      g_compile_out_full=""

      make_initial_paths_simple "$obj_path_unsanitized" "$bin_path_unsanitized"
      check_install "$tmp_path_unsanitized" "g++-multilib"
      compile_file "$cpp_compiler" "$specific_args $unzoned_args" "$src_path_unsanitized"/special/linux_wrap_functions.cpp "$obj_path_unsanitized" 0
      compile_directory "$cpp_compiler" "$specific_args $unzoned_args" "$src_path_unsanitized/unzoned" "$obj_path_unsanitized" 0
      compile_compute "$cpp_compiler" "$specific_args $compute_args" "$src_path_unsanitized" "$obj_path_unsanitized" 0 "cpu"
      compile_compute "$cpp_compiler" "$specific_args $compute_args -mavx2 -mfma" "$src_path_unsanitized" "$obj_path_unsanitized" 0 "avx2"
      compile_compute "$cpp_compiler" "$specific_args $compute_args -mavx512f" "$src_path_unsanitized" "$obj_path_unsanitized" 0 "avx512f"
      compile_directory "$cpp_compiler" "$specific_args $main_args" "$src_path_unsanitized" "$obj_path_unsanitized" 0
      link_file "$cpp_compiler" "$link_args $specific_args" "$bin_path_unsanitized" "$bin_file"
      printf "%s\n" "$g_compile_out_full"
      printf "%s\n" "$g_compile_out_full" > "$g_log_file_unsanitized"
      copy_bin_files "$bin_path_unsanitized" "$bin_file" "$python_lib_unsanitized" "$staging_path_unsanitized"
   fi

   if [ $debug_32 -eq 1 ]; then
      ########################## Linux debug|x86

      printf "%s\n" "Compiling libebm with $cpp_compiler for Linux debug|x86"
      obj_path_unsanitized="$tmp_path_unsanitized/gcc/obj/debug/linux/x86/libebm"
      bin_path_unsanitized="$tmp_path_unsanitized/gcc/bin/debug/linux/x86/libebm"
      bin_file="libebm_linux_x86_debug.so"
      g_log_file_unsanitized="$obj_path_unsanitized/libebm_debug_linux_x86_build_log.txt"
      specific_args="$all_args -DBRIDGE_AVX2_32 -DBRIDGE_AVX512F_32 -msse2 -mfpmath=sse -m32 -O1"
      # the linker wants to have the most dependent .o/.so/.dylib files listed FIRST
      
      g_all_object_files_sanitized=""
      g_compile_out_full=""

      make_initial_paths_simple "$obj_path_unsanitized" "$bin_path_unsanitized"
      check_install "$tmp_path_unsanitized" "g++-multilib"
      compile_file "$cpp_compiler" "$specific_args $unzoned_args" "$src_path_unsanitized"/special/linux_wrap_functions.cpp "$obj_path_unsanitized" 0
      compile_directory "$cpp_compiler" "$specific_args $unzoned_args" "$src_path_unsanitized/unzoned" "$obj_path_unsanitized" 0
      compile_compute "$cpp_compiler" "$specific_args $compute_args" "$src_path_unsanitized" "$obj_path_unsanitized" 0 "cpu"
      compile_compute "$cpp_compiler" "$specific_args $compute_args -mavx2 -mfma" "$src_path_unsanitized" "$obj_path_unsanitized" 0 "avx2"
      compile_compute "$cpp_compiler" "$specific_args $compute_args -mavx512f" "$src_path_unsanitized" "$obj_path_unsanitized" 0 "avx512f"
      compile_directory "$cpp_compiler" "$specific_args $main_args" "$src_path_unsanitized" "$obj_path_unsanitized" 0
      link_file "$cpp_compiler" "$link_args $specific_args" "$bin_path_unsanitized" "$bin_file"
      printf "%s\n" "$g_compile_out_full"
      printf "%s\n" "$g_compile_out_full" > "$g_log_file_unsanitized"
      copy_bin_files "$bin_path_unsanitized" "$bin_file" "$python_lib_unsanitized" "$staging_path_unsanitized"
   fi

elif [ "$os_type" = "Darwin" ]; then
   # reference on rpath & install_name: https://www.mikeash.com/pyblog/friday-qa-2009-11-06-linking-and-install-names.html

   # TODO: make these real options instead of forcing them to the same as x64
   release_arm=$release_64
   debug_arm=$debug_64

   # try moving some of these clang specific warnings into the shared all_args if g++ eventually supports them
   cpp_compiler=clang++
   
   all_args="$all_args -Wnull-dereference"
   all_args="$all_args -Wgnu-zero-variadic-macro-arguments"

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

      printf "%s\n" "Compiling libebm with $cpp_compiler for macOS release|x64"
      obj_path_unsanitized="$tmp_path_unsanitized/clang/obj/release/mac/x64/libebm"
      bin_path_unsanitized="$tmp_path_unsanitized/clang/bin/release/mac/x64/libebm"
      bin_file="libebm_mac_x64.dylib"
      g_log_file_unsanitized="$obj_path_unsanitized/libebm_release_mac_x64_build_log.txt"
      specific_args="$all_args -march=core2 -target x86_64-apple-macos10.12 -m64 -DNDEBUG -O3 -DBRIDGE_AVX2_32 -DBRIDGE_AVX512F_32"
      # the linker wants to have the most dependent .o/.so/.dylib files listed FIRST
   
      g_all_object_files_sanitized=""
      g_compile_out_full=""

      make_initial_paths_simple "$obj_path_unsanitized" "$bin_path_unsanitized"
      compile_directory "$cpp_compiler" "$specific_args $unzoned_args" "$src_path_unsanitized/unzoned" "$obj_path_unsanitized" "$is_asm"
      compile_compute "$cpp_compiler" "$specific_args $compute_args" "$src_path_unsanitized" "$obj_path_unsanitized" "$is_asm" "cpu"
      compile_compute "$cpp_compiler" "$specific_args $compute_args -mavx2 -mfma" "$src_path_unsanitized" "$obj_path_unsanitized" "$is_asm" "avx2"
      compile_compute "$cpp_compiler" "$specific_args $compute_args -mavx512f" "$src_path_unsanitized" "$obj_path_unsanitized" "$is_asm" "avx512f"
      compile_directory "$cpp_compiler" "$specific_args $main_args" "$src_path_unsanitized" "$obj_path_unsanitized" "$is_asm"
      link_file "$cpp_compiler" "-install_name @rpath/$bin_file $link_args $specific_args" "$bin_path_unsanitized" "$bin_file"
      printf "%s\n" "$g_compile_out_full"
      printf "%s\n" "$g_compile_out_full" > "$g_log_file_unsanitized"
      copy_bin_files "$bin_path_unsanitized" "$bin_file" "$python_lib_unsanitized" "$staging_path_unsanitized"
      copy_asm_files "$obj_path_unsanitized" "$tmp_path_unsanitized" "$staging_path_unsanitized/$bin_file" "asm_release_64" "$is_asm"
   fi

   if [ $debug_64 -eq 1 ]; then
      ########################## macOS debug|x64

      printf "%s\n" "Compiling libebm with $cpp_compiler for macOS debug|x64"
      obj_path_unsanitized="$tmp_path_unsanitized/clang/obj/debug/mac/x64/libebm"
      bin_path_unsanitized="$tmp_path_unsanitized/clang/bin/debug/mac/x64/libebm"
      bin_file="libebm_mac_x64_debug.dylib"
      g_log_file_unsanitized="$obj_path_unsanitized/libebm_debug_mac_x64_build_log.txt"
      specific_args="$all_args -march=core2 -target x86_64-apple-macos10.12 -m64 -O1 -DBRIDGE_AVX2_32 -DBRIDGE_AVX512F_32 -fsanitize=address,undefined -fno-sanitize-recover=address,undefined -fno-optimize-sibling-calls -fno-omit-frame-pointer"
      # the linker wants to have the most dependent .o/.so/.dylib files listed FIRST
   
      g_all_object_files_sanitized=""
      g_compile_out_full=""

      make_initial_paths_simple "$obj_path_unsanitized" "$bin_path_unsanitized"
      compile_directory "$cpp_compiler" "$specific_args $unzoned_args" "$src_path_unsanitized/unzoned" "$obj_path_unsanitized" 0
      compile_compute "$cpp_compiler" "$specific_args $compute_args" "$src_path_unsanitized" "$obj_path_unsanitized" 0 "cpu"
      compile_compute "$cpp_compiler" "$specific_args $compute_args -mavx2 -mfma" "$src_path_unsanitized" "$obj_path_unsanitized" 0 "avx2"
      compile_compute "$cpp_compiler" "$specific_args $compute_args -mavx512f" "$src_path_unsanitized" "$obj_path_unsanitized" 0 "avx512f"
      compile_directory "$cpp_compiler" "$specific_args $main_args" "$src_path_unsanitized" "$obj_path_unsanitized" 0
      link_file "$cpp_compiler" "-install_name @rpath/$bin_file $link_args $specific_args" "$bin_path_unsanitized" "$bin_file"
      printf "%s\n" "$g_compile_out_full"
      printf "%s\n" "$g_compile_out_full" > "$g_log_file_unsanitized"
      copy_bin_files "$bin_path_unsanitized" "$bin_file" "$python_lib_unsanitized" "$staging_path_unsanitized"
   fi

   if [ $release_arm -eq 1 ]; then
      ########################## macOS release|arm

      printf "%s\n" "Compiling libebm with $cpp_compiler for macOS release|arm"
      obj_path_unsanitized="$tmp_path_unsanitized/clang/obj/release/mac/arm/libebm"
      bin_path_unsanitized="$tmp_path_unsanitized/clang/bin/release/mac/arm/libebm"
      bin_file="libebm_mac_arm.dylib"
      g_log_file_unsanitized="$obj_path_unsanitized/libebm_release_mac_arm_build_log.txt"
      specific_args="$all_args -target arm64-apple-macos11 -m64 -DNDEBUG -O3"
      # the linker wants to have the most dependent .o/.so/.dylib files listed FIRST
   
      g_all_object_files_sanitized=""
      g_compile_out_full=""

      make_initial_paths_simple "$obj_path_unsanitized" "$bin_path_unsanitized"
      compile_directory "$cpp_compiler" "$specific_args $unzoned_args" "$src_path_unsanitized/unzoned" "$obj_path_unsanitized" "$is_asm"
      compile_compute "$cpp_compiler" "$specific_args $compute_args" "$src_path_unsanitized" "$obj_path_unsanitized" "$is_asm" "cpu"
      compile_directory "$cpp_compiler" "$specific_args $main_args" "$src_path_unsanitized" "$obj_path_unsanitized" "$is_asm"
      link_file "$cpp_compiler" "-install_name @rpath/$bin_file $link_args $specific_args" "$bin_path_unsanitized" "$bin_file"
      printf "%s\n" "$g_compile_out_full"
      printf "%s\n" "$g_compile_out_full" > "$g_log_file_unsanitized"
      copy_bin_files "$bin_path_unsanitized" "$bin_file" "$python_lib_unsanitized" "$staging_path_unsanitized"
      copy_asm_files "$obj_path_unsanitized" "$tmp_path_unsanitized" "$staging_path_unsanitized/$bin_file" "asm_release_arm" "$is_asm"
   fi

   if [ $debug_arm -eq 1 ]; then
      ########################## macOS debug|arm

      printf "%s\n" "Compiling libebm with $cpp_compiler for macOS debug|arm"
      obj_path_unsanitized="$tmp_path_unsanitized/clang/obj/debug/mac/arm/libebm"
      bin_path_unsanitized="$tmp_path_unsanitized/clang/bin/debug/mac/arm/libebm"
      bin_file="libebm_mac_arm_debug.dylib"
      g_log_file_unsanitized="$obj_path_unsanitized/libebm_debug_mac_arm_build_log.txt"
      specific_args="$all_args -target arm64-apple-macos11 -m64 -O1 -fsanitize=address,undefined -fno-sanitize-recover=address,undefined -fno-optimize-sibling-calls -fno-omit-frame-pointer"
      # the linker wants to have the most dependent .o/.so/.dylib files listed FIRST
   
      g_all_object_files_sanitized=""
      g_compile_out_full=""

      make_initial_paths_simple "$obj_path_unsanitized" "$bin_path_unsanitized"
      compile_directory "$cpp_compiler" "$specific_args $unzoned_args" "$src_path_unsanitized/unzoned" "$obj_path_unsanitized" 0
      compile_compute "$cpp_compiler" "$specific_args $compute_args" "$src_path_unsanitized" "$obj_path_unsanitized" 0 "cpu"
      compile_directory "$cpp_compiler" "$specific_args $main_args" "$src_path_unsanitized" "$obj_path_unsanitized" 0
      link_file "$cpp_compiler" "-install_name @rpath/$bin_file $link_args $specific_args" "$bin_path_unsanitized" "$bin_file"
      printf "%s\n" "$g_compile_out_full"
      printf "%s\n" "$g_compile_out_full" > "$g_log_file_unsanitized"
      copy_bin_files "$bin_path_unsanitized" "$bin_file" "$python_lib_unsanitized" "$staging_path_unsanitized"
   fi

else
   printf "%s\n" "OS $os_type not recognized.  We support clang/clang++ on macOS and gcc/g++ on Linux"
   exit 1
fi
