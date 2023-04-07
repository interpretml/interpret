@ECHO OFF
SETLOCAL

SET root_path=%~dp0..\..\..\

SET build_pipeline=0
SET "extra_analysis= "
for %%x in (%*) do (
   IF "%%x"=="-pipeline" (
      SET build_pipeline=1
   )
   IF "%%x"=="-analysis" (
      SET extra_analysis=-analysis
   )
)

IF %build_pipeline% EQU 0 (
   ECHO Building ebm_native library...
   CALL "%root_path%build.bat" -32bit %extra_analysis%
   IF ERRORLEVEL 1 (
      EXIT /B 201
   )
)

MSBuild.exe "%root_path%shared\ebm_native\ebm_native_test\ebm_native_test.vcxproj" /p:Configuration=Debug /p:Platform=x64
IF ERRORLEVEL 1 (
   ECHO MSBuild for Debug x64 FAILED
   EXIT /B 202
)
MSBuild.exe "%root_path%shared\ebm_native\ebm_native_test\ebm_native_test.vcxproj" /p:Configuration=Release /p:Platform=x64
IF ERRORLEVEL 1 (
   ECHO MSBuild for Release x64 FAILED
   EXIT /B 203
)
"%root_path%tmp\vs\bin\Debug\win\x64\ebm_native_test\ebm_native_test.exe"
IF ERRORLEVEL 1 (
   ECHO ebm_native_test.exe for Debug x64 FAILED
   EXIT /B 204
)
"%root_path%tmp\vs\bin\Release\win\x64\ebm_native_test\ebm_native_test.exe"
IF ERRORLEVEL 1 (
   ECHO ebm_native_test.exe for Release x64 FAILED
   EXIT /B 205
)


IF %build_pipeline% EQU 1 (
   ECHO Building ebm_native library for 32 bit and static analysis...
   CALL "%root_path%build.bat" -32bit %extra_analysis%
   IF ERRORLEVEL 1 (
      EXIT /B 206
   )
)


MSBuild.exe "%root_path%shared\ebm_native\ebm_native_test\ebm_native_test.vcxproj" /p:Configuration=Debug /p:Platform=Win32
IF ERRORLEVEL 1 (
   ECHO MSBuild for Debug x86 FAILED
   EXIT /B 207
)
MSBuild.exe "%root_path%shared\ebm_native\ebm_native_test\ebm_native_test.vcxproj" /p:Configuration=Release /p:Platform=Win32
IF ERRORLEVEL 1 (
   ECHO MSBuild for Release x86 FAILED
   EXIT /B 208
)
"%root_path%tmp\vs\bin\Debug\win\Win32\ebm_native_test\ebm_native_test.exe"
IF ERRORLEVEL 1 (
   ECHO ebm_native_test.exe for Debug x86 FAILED
   EXIT /B 209
)
"%root_path%tmp\vs\bin\Release\win\Win32\ebm_native_test\ebm_native_test.exe"
IF ERRORLEVEL 1 (
   ECHO ebm_native_test.exe for Release x86 FAILED
   EXIT /B 210
)

EXIT /B 0
