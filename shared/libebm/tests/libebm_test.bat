@ECHO OFF
SETLOCAL

SET root_path=%~dp0..\..\..\

SET bld_default=1

SET debug_64=0
SET release_64=0
SET debug_32=0
SET release_32=0

SET existing_debug_64=0
SET existing_release_64=0
SET existing_debug_32=0
SET existing_release_32=0

SET "extra_analysis= "
for %%x in (%*) do (
   IF "%%x"=="-debug_64" (
      SET debug_64=1
      SET bld_default=0
   )
   IF "%%x"=="-release_64" (
      SET release_64=1
      SET bld_default=0
   )
   IF "%%x"=="-debug_32" (
      SET debug_32=1
      SET bld_default=0
   )
   IF "%%x"=="-release_32" (
      SET release_32=1
      SET bld_default=0
   )

   IF "%%x"=="-existing_debug_64" (
      SET existing_debug_64=1
   )
   IF "%%x"=="-existing_release_64" (
      SET existing_release_64=1
   )
   IF "%%x"=="-existing_debug_32" (
      SET existing_debug_32=1
   )
   IF "%%x"=="-existing_release_32" (
      SET existing_release_32=1
   )

   IF "%%x"=="-analysis" (
      SET extra_analysis=-analysis
   )
)

IF %bld_default% EQU 1 (
   SET debug_64=1
   SET release_64=1
   SET debug_32=1
   SET release_32=1
)

IF %debug_64% EQU 1 (
   IF %existing_debug_64% EQU 0 (
      ECHO Building libebm library for 64 bit debug
      CALL "%root_path%build.bat" -debug_64
      IF ERRORLEVEL 1 (
         EXIT /B 201
      )
   )
   MSBuild.exe "%root_path%shared\libebm\tests\libebm_test.vcxproj" /p:Configuration=Debug /p:Platform=x64
   IF ERRORLEVEL 1 (
      ECHO MSBuild for Debug x64 FAILED
      EXIT /B 202
   )
   "%root_path%bld\tmp\vs\bin\Debug\win\x64\libebm_test\libebm_test.exe"
   IF ERRORLEVEL 1 (
      ECHO libebm_test.exe for Debug x64 FAILED
      EXIT /B 204
   )
   IF %existing_debug_64% EQU 1 (
      ECHO Building libebm library for 64 bit debug
      CALL "%root_path%build.bat" -debug_64 %extra_analysis%
      IF ERRORLEVEL 1 (
         EXIT /B 201
      )
   )
)

IF %release_64% EQU 1 (
   IF %existing_release_64% EQU 0 (
      ECHO Building libebm library for 64 bit release
      CALL "%root_path%build.bat" -release_64
      IF ERRORLEVEL 1 (
         EXIT /B 201
      )
   )
   MSBuild.exe "%root_path%shared\libebm\tests\libebm_test.vcxproj" /p:Configuration=Release /p:Platform=x64
   IF ERRORLEVEL 1 (
      ECHO MSBuild for Release x64 FAILED
      EXIT /B 203
   )
   "%root_path%bld\tmp\vs\bin\Release\win\x64\libebm_test\libebm_test.exe"
   IF ERRORLEVEL 1 (
      ECHO libebm_test.exe for Release x64 FAILED
      EXIT /B 205
   )
   IF %existing_release_64% EQU 1 (
      ECHO Building libebm library for 64 bit release
      CALL "%root_path%build.bat" -release_64 %extra_analysis%
      IF ERRORLEVEL 1 (
         EXIT /B 201
      )
   )
)

IF %debug_32% EQU 1 (
   IF %existing_debug_32% EQU 0 (
      ECHO Building libebm library for 32 bit debug
      CALL "%root_path%build.bat" -debug_32
      IF ERRORLEVEL 1 (
         EXIT /B 206
      )
   )
   MSBuild.exe "%root_path%shared\libebm\tests\libebm_test.vcxproj" /p:Configuration=Debug /p:Platform=Win32
   IF ERRORLEVEL 1 (
      ECHO MSBuild for Debug x86 FAILED
      EXIT /B 207
   )
   "%root_path%bld\tmp\vs\bin\Debug\win\Win32\libebm_test\libebm_test.exe"
   IF ERRORLEVEL 1 (
      ECHO libebm_test.exe for Debug x86 FAILED
      EXIT /B 209
   )
   IF %existing_debug_32% EQU 1 (
      ECHO Building libebm library for 32 bit debug
      CALL "%root_path%build.bat" -debug_32 %extra_analysis%
      IF ERRORLEVEL 1 (
         EXIT /B 206
      )
   )
)

IF %release_32% EQU 1 (
   IF %existing_release_32% EQU 0 (
      ECHO Building libebm library for 32 bit release
      CALL "%root_path%build.bat" -release_32
      IF ERRORLEVEL 1 (
         EXIT /B 206
      )
   )
   MSBuild.exe "%root_path%shared\libebm\tests\libebm_test.vcxproj" /p:Configuration=Release /p:Platform=Win32
   IF ERRORLEVEL 1 (
      ECHO MSBuild for Release x86 FAILED
      EXIT /B 208
   )
   "%root_path%bld\tmp\vs\bin\Release\win\Win32\libebm_test\libebm_test.exe"
   IF ERRORLEVEL 1 (
      ECHO libebm_test.exe for Release x86 FAILED
      EXIT /B 210
   )
   IF %existing_release_32% EQU 1 (
      ECHO Building libebm library for 32 bit release
      CALL "%root_path%build.bat" -release_32 %extra_analysis%
      IF ERRORLEVEL 1 (
         EXIT /B 206
      )
   )
)

EXIT /B 0
