@ECHO OFF
SETLOCAL

REM The free version of Visual Studio (Community) is sufficient for compiling InterpretML for Windows.
REM Visual Studio Community can be downloaded for free here:  https://visualstudio.microsoft.com/vs/

SET root_path=%~dp0

SET build_32_bit=0
SET build_64_bit=1
SET "extra_analysis= "
for %%x in (%*) do (
   IF "%%x"=="-32bit" (
      SET build_32_bit=1
   )
   IF "%%x"=="-no64bit" (
      SET build_64_bit=0
   )
   IF "%%x"=="-analysis" (
      SET extra_analysis=/p:EnableClangTidyCodeAnalysis=True /p:RunCodeAnalysis=True
   )
)

REM "IF ERRORLEVEL 1" checks for error levels 1 OR MORE!
REM %ERRORLEVEL% seems to be unreliable with MSBuild.exe
REM https://devblogs.microsoft.com/oldnewthing/20080926-00

IF %build_64_bit% EQU 1 (
   MSBuild.exe "%root_path%shared\ebm_native\ebm_native.vcxproj" /p:Configuration=Release /p:Platform=x64 %extra_analysis%
   IF ERRORLEVEL 1 (
      ECHO MSBuild for Release x64 FAILED
      EXIT /B 101
   )
   MSBuild.exe "%root_path%shared\ebm_native\ebm_native.vcxproj" /p:Configuration=Debug /p:Platform=x64
   IF ERRORLEVEL 1 (
      ECHO MSBuild for Debug x64 FAILED
      EXIT /B 102
   )
)
IF %build_32_bit% EQU 1 (
   MSBuild.exe "%root_path%shared\ebm_native\ebm_native.vcxproj" /p:Configuration=Release /p:Platform=Win32 %extra_analysis%
   IF ERRORLEVEL 1 (
      ECHO MSBuild for Release x86 FAILED
      EXIT /B 103
   )
   MSBuild.exe "%root_path%shared\ebm_native\ebm_native.vcxproj" /p:Configuration=Debug /p:Platform=Win32
   IF ERRORLEVEL 1 (
      ECHO MSBuild for Debug x86 FAILED
      EXIT /B 104
   )
)

EXIT /B 0
