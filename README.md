# logoRNN
CMPT 414- CV Modeling Term Project

Our project focuses on implementing region based CNN for logo detection

## Object Proposal Generator

### Dependencies

This project requires OpenCV and Caffe

### Build Object Proposal Generator

I recommend building in a separate directory

> mkdir build && cd build

Generate makefiles and build from inside the build directory

> cmake <path/to/logoRNN> && make

### How to Run

The generator is found in <path/to/logoRNN>/bin/bin

Run the generator to get the top right and bottom right corner coordinates

> ./opgen <path/to/image>

The output will return bounding box coordinates.
Each line will be 4 numbers separated by commas in the format top left x, top left y, bottom right x, bottom right y.

Running the generator on `./tests/imgs/test1.jpg` will yield the following:

```
./tests/imgs/test1.jpg 136 0 293 76
./tests/imgs/test1.jpg 217 0 499 231
./tests/imgs/test1.jpg 217 0 499 231
./tests/imgs/test1.jpg 153 71 354 248
./tests/imgs/test1.jpg 371 132 499 227
./tests/imgs/test1.jpg 308 212 499 310
./tests/imgs/test1.jpg 270 26 383 204
./tests/imgs/test1.jpg 217 0 499 231
./tests/imgs/test1.jpg 270 15 412 204
./tests/imgs/test1.jpg 0 0 499 374
```

# FUTURE PLANS

## RCNN setup

1. Build opgen

2. Add folders `trainingset` and `testingset` to `data`

3. run `train_proposals.sh` and `test_proposals.sh` to generate a list of bounding boxes

## Training

With GPU

> ./pythontools/train_net.py --gpu 0 --solver models/solver.prototxt --weights data/models/logos.model

Without GPU

> ./pythontools/train_net.py --cpu-only --solver models/solver.prototxt --weights data/models/logos.model
