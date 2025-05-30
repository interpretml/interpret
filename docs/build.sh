jupyter-book build --warningiserror --keep-going ./interpret/
ret_code=$?

cp ./extras/* ./interpret/_build/html/

# make an empty .nojekyll file because Github Pages will delete directories that start with '_' otherwise
printf "This file stops Github Pages from deleting directories that start with _\n" > ./interpret/_build/html/.nojekyll

exit $ret_code
