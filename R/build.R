#!/bin/Rscript

copy_code <- function(source_path, destination_path) {
   if(!dir.exists(destination_path)) {
      dir.create(destination_path)
   }

   for (file in list.files(source_path)) {
      if(endsWith(file, ".h") || endsWith(file, ".c") || endsWith(file, ".hpp") || endsWith(file, ".cpp")) {
         file.copy(from = file.path(source_path, file), to = file.path(destination_path, file))
      }
   }
}

root_path <- getwd()

script_file <- file.path(root_path, "build.R")
if(!file.exists(script_file)) {
   print(paste0("script ", as.character(script_file), " does not exist"))
   q(status=1)
}

tmp_path <- file.path(root_path, "..", "tmp")
if(!dir.exists(tmp_path)) {
   dir.create(tmp_path)
}

dest_path <- file.path(tmp_path, "R")

if(0 != unlink(dest_path, recursive=TRUE)) {
   print(paste0("could not delete path ", as.character(dest_path)))
   q(status=1)
}

# root directory
dir.create(dest_path)
file.copy(from = file.path(root_path, "DESCRIPTION"), to = file.path(dest_path, "DESCRIPTION"))
file.copy(from = file.path(root_path, "NAMESPACE"), to = file.path(dest_path, "NAMESPACE"))
file.copy(from = file.path(root_path, "cran_formatted_licence.txt"), to = file.path(dest_path, "LICENSE"))

# man directory
dir.create(file.path(dest_path, "man"))
for (file in list.files(file.path(root_path, "man"))) {
   file.copy(from = file.path(root_path, "man", file), to = file.path(dest_path, "man", file))
}

# R directory
dir.create(file.path(dest_path, "R"))
for (file in list.files(file.path(root_path, "R"))) {
   file.copy(from = file.path(root_path, "R", file), to = file.path(dest_path, "R", file))
}

# src directory (R C++ files)
dir.create(file.path(dest_path, "src"))
file.copy(from = file.path(root_path, "src", "interpret_R.cpp"), to = file.path(dest_path, "src", "interpret_R.cpp"))
file.copy(from = file.path(root_path, "src", "interpret-win.def"), to = file.path(dest_path, "src", "interpret-win.def"))
file.copy(from = file.path(root_path, "src", "Makevars.interpret"), to = file.path(dest_path, "src", "Makevars"))

# src/libebm directory (non-R C++ files)
native_path <- file.path(root_path, "..", "shared", "libebm")
cxx_path <- file.path(dest_path, "src", "libebm")
copy_code(native_path, cxx_path)
copy_code(file.path(native_path, "inc"), file.path(cxx_path, "inc"))
copy_code(file.path(native_path, "common_c"), file.path(cxx_path, "common_c"))
copy_code(file.path(native_path, "common_cpp"), file.path(cxx_path, "common_cpp"))
copy_code(file.path(native_path, "bridge_c"), file.path(cxx_path, "bridge_c"))
copy_code(file.path(native_path, "bridge_cpp"), file.path(cxx_path, "bridge_cpp"))
copy_code(file.path(native_path, "compute"), file.path(cxx_path, "compute"))
copy_code(file.path(native_path, "compute", "loss_functions"), file.path(cxx_path, "compute", "loss_functions"))
copy_code(file.path(native_path, "compute", "metrics"), file.path(cxx_path, "compute", "metrics"))
copy_code(file.path(native_path, "compute", "cpu_ebm"), file.path(cxx_path, "compute", "cpu_ebm"))

# we do not use these yet
file.remove(file.path(cxx_path, "compute", "cpu_ebm", "sse2_32.cpp"))
file.remove(file.path(cxx_path, "compute", "cpu_ebm", "sse2_64.cpp"))

staging_path <- file.path(root_path, "..", "staging")

# Create the staging directory
if(!dir.exists(staging_path)) {
   dir.create(staging_path)
}

# Delete the interpret_*.tar.gz files in the staging directory
for (file in list.files(staging_path)) {
   if(startsWith(file, "interpret_") && endsWith(file, ".tar.gz")) {
      file.remove(file.path(staging_path, file))
   }
}

# yuck, but I guess I must..
setwd(staging_path)

# Build the package using R CMD build

if(0 != system("R CMD build ../tmp/R")) {
   print("R CMD build failed")
   setwd(root_path)
   q(status=1)
}

# Check the package using R CMD check

# ideally, we'd also use --as-cran but we would then also want "set _R_CHECK_CRAN_INCOMING_=0"

if(0 != system("R CMD check -o ../tmp/R interpret_*.tar.gz")) {
   print("R CMD check failed")
   setwd(root_path)
   q(status=1)
}

setwd(root_path)
