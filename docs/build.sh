jupyter-book build --warningiserror ./interpret/
ret_code=$?

mkdir -p ./interpret/_build/html/images/
cp ./interpret/images/*.png ./interpret/_build/html/images/

mkdir -p ./interpret/_build/html/examples/images/
cp ./interpret/examples/images/*.png ./interpret/_build/html/examples/images/

# make an empty .nojekyll file because Github Pages will delete directories that start with '_' otherwise
printf "This file stops Github Pages from deleting directories that start with _\n" > ./interpret/_build/html/.nojekyll

exit $ret_code
