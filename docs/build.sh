jupyter-book build ./interpret_docs/
mkdir -p ./interpret_docs/_build/html/assets/images/
cp ./interpret_docs/assets/images/*.png ./interpret_docs/_build/html/assets/images/

# make an empty .nojekyll file because Github Pages will delete directories that start with '_' otherwise
printf "This file stops Github Pages from deleting directories that start with _\n" > ./interpret_docs/_build/html/.nojekyll
