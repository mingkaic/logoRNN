#!/usr/bin/env bash
rm testing_bbox.txt

for filename in ./data/testset/*; do
    echo $filename
    ./bin/bin/opgen "$filename" >> testing_bbox.txt
done