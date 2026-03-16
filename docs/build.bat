@setlocal

SET INTERPRET_DOCS_MODE=docs

jupyter-book build --warningiserror --keep-going .\interpret\
SET RETURNCODE=%ERRORLEVEL%

copy .\extras\* .\interpret\_build\html\ /Y

echo This file stops Github Pages from deleting directories that start with _> .\interpret\_build\html\.nojekyll

endlocal & exit /b %RETURNCODE%
