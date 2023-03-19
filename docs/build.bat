jupyter-book build .\interpret_docs\
mkdir .\interpret_docs\_build\html\assets\images\
copy .\interpret_docs\assets\images\*.png .\interpret_docs\_build\html\assets\images\ /Y

echo This file stops Github Pages from deleting directories that start with _> .\interpret_docs\_build\html\.nojekyll
