# logoRNN
CMPT 414- CV Modeling Term Project

Our project focuses on implementing region based CNN for logo detection

## Object Proposal Generator

### Dependencies

This project requires OpenCV (and Caffe for Fast RCNN)

### Build Object Proposal Generator

I recommend building in a separate directory

> mkdir build && cd build

Generate makefiles and build from inside the build directory,

> cmake <path/to/logoRNN> && cmake --build

Note that the project requires third party library (egbis) found on github 
and building will attempt pull and merge it to local. Building offline will fail. 

### Components

For demonstration purposes, building creates 3 executables:

- opgen
- optest
- viewer

The executables are found in directory `./bin/bin`.

### How to Run

#### Opgen

> ./bin/bin/opgen <path/to/image> colorweight textureweight sizeweight fillweight topk

weight and topk arguments are optional and will default to 1 for weights and 3 for topk

Run the generator to get the top right and bottom right corner of the topk object proposals 

Each line will be 4 numbers separated by commas in the format top left x, top left y, bottom right x, bottom right y.

Running the generator on `./tests/imgs/test1.jpg` will yield the following:

```
./tests/imgs/test1.jpg 136 0 293 76
./tests/imgs/test1.jpg 217 0 499 231
./tests/imgs/test1.jpg 217 0 499 231
```

#### Optest

> ./bin/bin/optest <path/to/image> colorweight textureweight sizeweight fillweight

Similar to Opgen, Optest's weight arguments are optional and default to 1.

Run the test to obtain the following window:

![alt tag](https://github.com/mingkaic/logoRNN/doc/optest.jpg)

#### Viewer

> ./bin/bin/viewer <path/to/image> colorweight textureweight sizeweight fillweight

Run the viewer to obtain a set of windows demonstrating each iteration(level) of grouping:

![alt tag](https://github.com/mingkaic/logoRNN/doc/viewer.jpg)

Warning: Viewer may generate a lot of windows for highly segmented images, I recommend using test images instead

## Experiment

Our scoring test will generate a text file describing how object proposal accuracy for each combination of weight parameters with each parameter being either 0 or 1.

1. Build Opgen

2. Unzip `data`

3. Run `scoringprocess.sh`

# FUTURE PLANS -- INCOMPLETE --

## RCNN setup

1. Build opgen

2. Unzip `data`

3. run `train_proposals.sh` and `test_proposals.sh` to generate a list of bounding boxes

4. Git pull submodule `caffe-fast-rcnn`

## Training

With GPU

> ./pythontools/train_net.py --gpu 0 --solver models/solver.prototxt --weights data/models/logos.model

Without GPU

> ./pythontools/train_net.py --cpu-only --solver models/solver.prototxt --weights data/models/logos.model

## Testing

Yet to be complete