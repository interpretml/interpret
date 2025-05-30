@ECHO OFF
SETLOCAL

REM The free version of Visual Studio (Community) is sufficient for compiling InterpretML for Windows.
REM Visual Studio Community can be downloaded for free here:  https://visualstudio.microsoft.com/vs/

REM If running in a non-Visual Studio window, the required environment variables can be obtained with:
REM "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"

SET root_path=%~dp0

SET bld_default=1

SET release_64=0
SET debug_64=0
SET release_32=0
SET debug_32=0

SET "extra_analysis= "
for %%x in (%*) do (
   IF "%%x"=="-release_64" (
      SET release_64=1
      SET bld_default=0
   )
   IF "%%x"=="-debug_64" (
      SET debug_64=1
      SET bld_default=0
   )
   IF "%%x"=="-release_32" (
      SET release_32=1
      SET bld_default=0
   )
   IF "%%x"=="-debug_32" (
      SET debug_32=1
      SET bld_default=0
   )

   IF "%%x"=="-analysis" (
      SET extra_analysis=/p:EnableClangTidyCodeAnalysis=True /p:RunCodeAnalysis=True
   )
)

IF %bld_default% EQU 1 (
   SET release_64=1
   SET debug_64=1
   SET release_32=1
   SET debug_32=1
)

REM "IF ERRORLEVEL 1" checks for error levels 1 OR MORE!
REM %ERRORLEVEL% seems to be unreliable with MSBuild.exe
REM https://devblogs.microsoft.com/oldnewthing/20080926-00

IF %release_64% EQU 1 (
   MSBuild.exe "%root_path%shared\libebm\libebm.vcxproj" /p:Configuration=Release /p:Platform=x64 %extra_analysis%
   IF ERRORLEVEL 1 (
      ECHO MSBuild for Release x64 FAILED
      EXIT /B 101
   )
)
IF %debug_64% EQU 1 (
   MSBuild.exe "%root_path%shared\libebm\libebm.vcxproj" /p:Configuration=Debug /p:Platform=x64
   IF ERRORLEVEL 1 (
      ECHO MSBuild for Debug x64 FAILED
      EXIT /B 102
   )
)
IF %release_32% EQU 1 (
   MSBuild.exe "%root_path%shared\libebm\libebm.vcxproj" /p:Configuration=Release /p:Platform=Win32 %extra_analysis%
   IF ERRORLEVEL 1 (
      ECHO MSBuild for Release x86 FAILED
      EXIT /B 103
   )
)
IF %debug_32% EQU 1 (
   MSBuild.exe "%root_path%shared\libebm\libebm.vcxproj" /p:Configuration=Debug /p:Platform=Win32
   IF ERRORLEVEL 1 (
      ECHO MSBuild for Debug x86 FAILED
      EXIT /B 104
   )
)

EXIT /B 0
