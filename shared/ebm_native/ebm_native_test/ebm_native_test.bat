@ECHO OFF
SETLOCAL

SET root_path=%~dp0..\..\..\

SET build_pipeline=0
for %%x in (%*) do (
   IF "%%x"=="-pipeline" (
      SET build_pipeline=1
   )
)

IF %build_pipeline% EQU 0 (
   ECHO Building ebm_native library...
   CALL "%root_path%build.bat" -32bit -analysis
   IF %ERRORLEVEL% NEQ 0 (
      REM build.bat should already have written out an error message
      EXIT /B %ERRORLEVEL%
   )
)

MSBuild.exe "%root_path%shared\ebm_native\ebm_native_test\ebm_native_test.vcxproj" /p:Configuration=Debug /p:Platform=x64
IF %ERRORLEVEL% NEQ 0 (
   ECHO MSBuild for Debug x64 returned error code %ERRORLEVEL%
   EXIT /B %ERRORLEVEL%
)
MSBuild.exe "%root_path%shared\ebm_native\ebm_native_test\ebm_native_test.vcxproj" /p:Configuration=Release /p:Platform=x64
IF %ERRORLEVEL% NEQ 0 (
   ECHO MSBuild for Release x64 returned error code %ERRORLEVEL%
   EXIT /B %ERRORLEVEL%
)
"%root_path%tmp\vs\bin\Debug\win\x64\ebm_native_test\ebm_native_test.exe"
IF %ERRORLEVEL% NEQ 0 (
   ECHO ebm_native_test.exe for Debug x64 failed with error code %ERRORLEVEL%
   EXIT /B %ERRORLEVEL%
)
"%root_path%tmp\vs\bin\Release\win\x64\ebm_native_test\ebm_native_test.exe"
IF %ERRORLEVEL% NEQ 0 (
   ECHO ebm_native_test.exe for Release x64 failed with error code %ERRORLEVEL%
   EXIT /B %ERRORLEVEL%
)


IF %build_pipeline% EQU 1 (
   ECHO Building ebm_native library for 32 bit and static analysis...
   CALL "%root_path%build.bat" -32bit -analysis
   IF %ERRORLEVEL% NEQ 0 (
      REM build.bat should already have written out an error message
      EXIT /B %ERRORLEVEL%
   )
)


MSBuild.exe "%root_path%shared\ebm_native\ebm_native_test\ebm_native_test.vcxproj" /p:Configuration=Debug /p:Platform=Win32
IF %ERRORLEVEL% NEQ 0 (
   ECHO MSBuild for Debug x86 returned error code %ERRORLEVEL%
   EXIT /B %ERRORLEVEL%
)
MSBuild.exe "%root_path%shared\ebm_native\ebm_native_test\ebm_native_test.vcxproj" /p:Configuration=Release /p:Platform=Win32
IF %ERRORLEVEL% NEQ 0 (
   ECHO MSBuild for Release x86 returned error code %ERRORLEVEL%
   EXIT /B %ERRORLEVEL%
)
"%root_path%tmp\vs\bin\Debug\win\Win32\ebm_native_test\ebm_native_test.exe"
IF %ERRORLEVEL% NEQ 0 (
   ECHO ebm_native_test.exe for Debug x86 failed with error code %ERRORLEVEL%
   EXIT /B %ERRORLEVEL%
)
"%root_path%tmp\vs\bin\Release\win\Win32\ebm_native_test\ebm_native_test.exe"
IF %ERRORLEVEL% NEQ 0 (
   ECHO ebm_native_test.exe for Release x86 failed with error code %ERRORLEVEL%
   EXIT /B %ERRORLEVEL%
)

EXIT /B 0
