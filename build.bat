@ECHO OFF
SETLOCAL

SET script_path=%~dp0

MSBuild.exe "%script_path%src\core\emlcore.sln" /p:Configuration=Release /p:Platform=x64
IF %ERRORLEVEL% NEQ 0 (
   ECHO MSBuild for Release x64 returned error code %ERRORLEVEL%
   EXIT /B %ERRORLEVEL%
)
MSBuild.exe "%script_path%src\core\emlcore.sln" /p:Configuration=Release /p:Platform=x86
IF %ERRORLEVEL% NEQ 0 (
   ECHO MSBuild for Release x86 returned error code %ERRORLEVEL%
   EXIT /B %ERRORLEVEL%
)
MSBuild.exe "%script_path%src\core\emlcore.sln" /p:Configuration=Debug /p:Platform=x64
IF %ERRORLEVEL% NEQ 0 (
   ECHO MSBuild for Debug x64 returned error code %ERRORLEVEL%
   EXIT /B %ERRORLEVEL%
)
MSBuild.exe "%script_path%src\core\emlcore.sln" /p:Configuration=Debug /p:Platform=x86
IF %ERRORLEVEL% NEQ 0 (
   ECHO MSBuild for Debug x86 returned error code %ERRORLEVEL%
   EXIT /B %ERRORLEVEL%
)

EXIT /B 0
