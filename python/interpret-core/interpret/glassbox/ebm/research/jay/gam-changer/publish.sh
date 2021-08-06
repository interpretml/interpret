#!/bin/bash

npm run build

if [ $# -eq 1 ]
then
  surge public https://xiaohk-sparrow-$1.surge.sh
else
  surge public https://xiaohk-sparrow.surge.sh
fi
