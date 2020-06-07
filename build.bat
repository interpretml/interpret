@ECHO OFF
SETLOCAL

REM The free version of Visual Studio (Community) is sufficient for compiling InterpretML for Windows.
REM Visual Studio Community can be downloaded for free here:  https://visualstudio.microsoft.com/vs/

SET root_path=%~dp0

SET build_32_bit=0
for %%x in (%*) do (
   IF "%%x"=="-32bit" (
      SET build_32_bit=1
   )
)

MSBuild.exe "%root_path%shared\ebm_native\ebm_native.vcxproj" /p:Configuration=Release /p:Platform=x64 /p:EnableClangTidyCodeAnalysis=True /p:RunCodeAnalysis=True
IF %ERRORLEVEL% NEQ 0 (
   ECHO MSBuild for Release x64 returned error code %ERRORLEVEL%
   EXIT /B %ERRORLEVEL%
)
MSBuild.exe "%root_path%shared\ebm_native\ebm_native.vcxproj" /p:Configuration=Debug /p:Platform=x64
IF %ERRORLEVEL% NEQ 0 (
   ECHO MSBuild for Debug x64 returned error code %ERRORLEVEL%
   EXIT /B %ERRORLEVEL%
)
IF %build_32_bit% EQU 1 (
   MSBuild.exe "%root_path%shared\ebm_native\ebm_native.vcxproj" /p:Configuration=Release /p:Platform=Win32
   IF %ERRORLEVEL% NEQ 0 (
      ECHO MSBuild for Release x86 returned error code %ERRORLEVEL%
      EXIT /B %ERRORLEVEL%
   )
   MSBuild.exe "%root_path%shared\ebm_native\ebm_native.vcxproj" /p:Configuration=Debug /p:Platform=Win32
   IF %ERRORLEVEL% NEQ 0 (
      ECHO MSBuild for Debug x86 returned error code %ERRORLEVEL%
      EXIT /B %ERRORLEVEL%
   )
)

EXIT /B 0
