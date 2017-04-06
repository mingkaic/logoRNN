#!/usr/bin/env bash
rm ./data/testing_bbox.txt

for filename in ./data/testingset/*; do
    echo $filename
    ./bin/bin/opgen "$filename" >> ./data/testing_bbox.txt
done