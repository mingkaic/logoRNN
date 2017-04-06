#!/usr/bin/env bash
rm ./data/training_bbox.txt

for filename in ./data/trainingset/*; do
    echo $filename
    ./bin/bin/opgen "$filename" >> ./data/training_bbox.txt
done