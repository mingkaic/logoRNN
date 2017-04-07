#!/usr/bin/env bash
rm ./data/testing_bbox.txt

colorw=${1:1}
textw=${2:1}
sizew=${3:1}
gapw=${4:1}

for filename in ./data/testingset/*; do
echo $filename
./bin/bin/opgen "$filename" $colorw $textw $sizew $gapw >> ./data/testing_bbox.txt
done