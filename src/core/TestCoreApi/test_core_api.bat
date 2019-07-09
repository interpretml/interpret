@ECHO OFF
SETLOCAL

SET script_path=%~dp0
SET build_core=1

for %%x in (%*) do (
   IF "%%x"=="nobuildcore" (
      SET build_core=0
   )
   IF "%%x"=="+nobuildcore" (
      SET build_core=0
   )
   IF "%%x"=="-nobuildcore" (
      SET build_core=0
   )
   IF "%%x"=="/nobuildcore" (
      SET build_core=0
   )
)

IF %build_core% NEQ 0 (
   ECHO Building Core library...
   CALL "%script_path%..\..\..\build.bat"
) ELSE (
   ECHO Core library NOT being built
)

MSBuild.exe "%script_path%TestCoreApi.vcxproj" /p:Configuration=Release /p:Platform=x64
IF %ERRORLEVEL% NEQ 0 (
   ECHO MSBuild for Release x64 returned error code %ERRORLEVEL%
   EXIT /B %ERRORLEVEL%
)
MSBuild.exe "%script_path%TestCoreApi.vcxproj" /p:Configuration=Release /p:Platform=Win32
IF %ERRORLEVEL% NEQ 0 (
   ECHO MSBuild for Release x86 returned error code %ERRORLEVEL%
   EXIT /B %ERRORLEVEL%
)
MSBuild.exe "%script_path%TestCoreApi.vcxproj" /p:Configuration=Debug /p:Platform=x64
IF %ERRORLEVEL% NEQ 0 (
   ECHO MSBuild for Debug x64 returned error code %ERRORLEVEL%
   EXIT /B %ERRORLEVEL%
)
MSBuild.exe "%script_path%TestCoreApi.vcxproj" /p:Configuration=Debug /p:Platform=Win32
IF %ERRORLEVEL% NEQ 0 (
   ECHO MSBuild for Debug x86 returned error code %ERRORLEVEL%
   EXIT /B %ERRORLEVEL%
)

"%script_path%..\..\..\tmp\vs\bin\Debug\win\x64\TestCoreApi\test_core_api.exe"
IF %ERRORLEVEL% NEQ 0 (
   ECHO test_core_api.exe for Debug x64 failed with error code %ERRORLEVEL%
   EXIT /B %ERRORLEVEL%
)
"%script_path%..\..\..\tmp\vs\bin\Debug\win\Win32\TestCoreApi\test_core_api.exe"
IF %ERRORLEVEL% NEQ 0 (
   ECHO test_core_api.exe for Debug x86 failed with error code %ERRORLEVEL%
   EXIT /B %ERRORLEVEL%
)
"%script_path%..\..\..\tmp\vs\bin\Release\win\x64\TestCoreApi\test_core_api.exe"
IF %ERRORLEVEL% NEQ 0 (
   ECHO test_core_api.exe for Release x64 failed with error code %ERRORLEVEL%
   EXIT /B %ERRORLEVEL%
)
"%script_path%..\..\..\tmp\vs\bin\Release\win\Win32\TestCoreApi\test_core_api.exe"
IF %ERRORLEVEL% NEQ 0 (
   ECHO test_core_api.exe for Release x86 failed with error code %ERRORLEVEL%
   EXIT /B %ERRORLEVEL%
)

EXIT /B 0
