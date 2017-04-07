#!/usr/bin/env bash
rm testres.txt

./test_proposals.sh 1 1 1 1
./train_proposals.sh 1 1 1 1
echo "combination 1 1 1 1" >> testres.txt
python pythontools/assess_bbox.py >> testres.txt


./test_proposals.sh 1 0 0 0
./train_proposals.sh 1 0 0 0
echo "combination 1 0 0 0" >> testres.txt
python pythontools/assess_bbox.py >> testres.txt


./test_proposals.sh 1 1 1 0
./train_proposals.sh 1 1 1 0
echo "combination 1 1 1 0" >> testres.txt
python pythontools/assess_bbox.py >> testres.txt


./test_proposals.sh 1 1 0 1
./train_proposals.sh 1 1 0 1
echo "combination 1 1 0 1" >> testres.txt
python pythontools/assess_bbox.py >> testres.txt


./test_proposals.sh 1 0 1 1
./train_proposals.sh 1 0 1 1
echo "combination 1 0 1 1" >> testres.txt
python pythontools/assess_bbox.py >> testres.txt


./test_proposals.sh 0 1 1 1
./train_proposals.sh 0 1 1 1
echo "combination 0 1 1 1" >> testres.txt
python pythontools/assess_bbox.py >> testres.txt


./test_proposals.sh 1 1 0 0
./train_proposals.sh 1 1 0 0
echo "combination 1 1 0 0" >> testres.txt
python pythontools/assess_bbox.py >> testres.txt


./test_proposals.sh 1 0 1 0
./train_proposals.sh 1 0 1 0
echo "combination 1 0 1 0" >> testres.txt
python pythontools/assess_bbox.py >> testres.txt


./test_proposals.sh 0 1 1 0
./train_proposals.sh 0 1 1 0
echo "combination 0 1 1 0" >> testres.txt
python pythontools/assess_bbox.py >> testres.txt


./test_proposals.sh 1 0 0 1
./train_proposals.sh 1 0 0 1
echo "combination 1 0 0 1" >> testres.txt
python pythontools/assess_bbox.py >> testres.txt


./test_proposals.sh 0 1 0 1
./train_proposals.sh 0 1 0 1
echo "combination 0 1 0 1" >> testres.txt
python pythontools/assess_bbox.py >> testres.txt


./test_proposals.sh 0 0 1 1
./train_proposals.sh 0 0 1 1
echo "combination 0 0 1 1" >> testres.txt
python pythontools/assess_bbox.py >> testres.txt


./test_proposals.sh 0 1 0 0
./train_proposals.sh 0 1 0 0
echo "combination 0 1 0 0" >> testres.txt
python pythontools/assess_bbox.py >> testres.txt


./test_proposals.sh 0 0 1 0
./train_proposals.sh 0 0 1 0
echo "combination 0 0 1 0" >> testres.txt
python pythontools/assess_bbox.py >> testres.txt


./test_proposals.sh 0 0 0 1
./train_proposals.sh 0 0 0 1
echo "combination 0 0 0 1" >> testres.txt
python pythontools/assess_bbox.py >> testres.txt