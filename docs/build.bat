jupyter-book build .\interpret\
SET RETURNCODE=%ERRORLEVEL%

mkdir .\interpret\_build\html\images\
copy .\interpret\images\*.png .\interpret\_build\html\images\ /Y

mkdir .\interpret\_build\html\examples\images\
copy .\interpret\examples\images\*.png .\interpret\_build\html\examples\images\ /Y

echo This file stops Github Pages from deleting directories that start with _> .\interpret\_build\html\.nojekyll

exit /b %RETURNCODE%
