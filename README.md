# logoRNN
CMPT 414- CV Modeling Term Project

Our project focuses on implementing region based CNN for logo detection

## Object Proposal Generator

### Dependencies

This project requires OpenCV and Caffe

### Build

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

Running the generator on `tests/imgs/test1.jpg` will yield the following:

```
136,0,293,76
217,0,499,231
217,0,499,231
153,71,354,248
371,132,499,227
308,212,499,310
270,26,383,204
217,0,499,231
270,15,412,204
0,0,499,374
```

## Integration

Modify `train.py`. 

It generates 
1. dictionary mapping each filename to an array of bounding boxes. 
2. dictionary mapping each filename to its associated label

Do something with these dictionaries (hash maps)

the python script accepts the following argument: path to training img directory, path to labels text file

Example:

> python train.py ./flickr_logos_27_dataset/subset ./flickr_logos_27_dataset_training_set_annotation.txt

## Training

With GPU

> ./pythontools/train_net.py --gpu 0 --solver models/solver.prototxt \
      --weights data/models/logos.model

Without GPU

> ./pythontools/train_net.py --cpu-only --solver models/solver.prototxt \
      --weights data/models/logos.model

## How to Merge
- git clone https://github.com/mingkaic/logoRNN.git
- copy over existing files
- git add .
- git commit -m "comment"
- git push origin master

OR
from your pre-existing project directory
- git init

set up local authentication if you haven't already

- git remote add origin git@github.com:mingkaic/logoRNN.git
- git pull origin master
- git add .
- git commit -m "comment"
- git push origin master