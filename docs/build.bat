jupyter-book build --warningiserror --keep-going .\interpret\
SET RETURNCODE=%ERRORLEVEL%

mkdir .\interpret\_build\html\images\
copy .\interpret\images\*.png .\interpret\_build\html\images\ /Y

mkdir .\interpret\_build\html\python\examples\images\
copy .\interpret\python\examples\images\*.png .\interpret\_build\html\python\examples\images\ /Y

echo This file stops Github Pages from deleting directories that start with _> .\interpret\_build\html\.nojekyll

exit /b %RETURNCODE%
