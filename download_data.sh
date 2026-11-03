#!/bin/sh
# Downloads WANDS (Wayfair ANnotation DataSet, MIT license) into data/wands/.
set -e
mkdir -p data/wands
for f in product.csv query.csv label.csv; do
  curl -sL -o "data/wands/$f" "https://raw.githubusercontent.com/wayfair/WANDS/main/dataset/$f"
done
echo "WANDS downloaded to data/wands/"
