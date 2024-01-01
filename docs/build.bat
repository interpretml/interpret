jupyter-book build .\interpret\
mkdir .\interpret\_build\html\assets\images\
copy .\interpret\assets\images\*.png .\interpret\_build\html\assets\images\ /Y

echo This file stops Github Pages from deleting directories that start with _> .\interpret\_build\html\.nojekyll
