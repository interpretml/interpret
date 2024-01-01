jupyter-book build ./interpret/
mkdir -p ./interpret/_build/html/assets/images/
cp ./interpret/assets/images/*.png ./interpret/_build/html/assets/images/

# make an empty .nojekyll file because Github Pages will delete directories that start with '_' otherwise
printf "This file stops Github Pages from deleting directories that start with _\n" > ./interpret/_build/html/.nojekyll
