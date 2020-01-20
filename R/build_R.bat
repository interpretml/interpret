@ECHO OFF
SETLOCAL

SET root_path=%~dp0

robocopy "%root_path%\" "%root_path%..\tmp\R" /S /PURGE /R:2 /NP /XF "%root_path%src\Makevars" /XF "%root_path%src\Makevars.interpret" /XF "%root_path%build_R.bat" /XF cran_lic.txt /XF LICENSE /XD "%root_path%src\native"
ECHO robocopy returned error code %ERRORLEVEL%
IF %ERRORLEVEL% GEQ 8 (
   EXIT /B %ERRORLEVEL%
)

copy /Y "%root_path%src\Makevars.interpret" "%root_path%..\tmp\R\src\Makevars"

copy /Y "%root_path%cran_lic.txt" "%root_path%..\tmp\R\LICENSE"

robocopy "%root_path%..\shared\native" "%root_path%..\tmp\R\src\native" /S /PURGE /R:2 /NP /XF wrap_func.cpp /XF DllMainCore.cpp /XF ebmcore.def /XF ebmcore.vcxproj /XF ebmcore.vcxproj.user /XF ebmcore_exports.txt /XF PrecompiledHeader.cpp
ECHO robocopy returned error code %ERRORLEVEL%
IF %ERRORLEVEL% GEQ 8 (
   EXIT /B %ERRORLEVEL%
)

del "%root_path%..\staging\interpret_*.tar.gz"
mkdir "%root_path%..\staging"
pushd "%root_path%..\staging"
Rcmd build "%root_path%..\tmp\R"
popd

R CMD check -o "%root_path%..\tmp\R" --as-cran "%root_path%..\staging\interpret_*.tar.gz"
