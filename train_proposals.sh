#!/usr/bin/env bash
rm training_bbox.txt

for filename in ./data/trainingset/*; do
    echo $filename
    ./bin/bin/opgen "$filename" >> training_bbox.txt
done