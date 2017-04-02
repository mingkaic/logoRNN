#!/usr/bin/python

import os
import sys
import subprocess

cmd = "./bin/bin/opgen"
imgdir = ""
labelf = ""

if len(sys.argv) > 1:
    imgdir = sys.argv[1]

if len(sys.argv) > 2:
    labelf = sys.argv[2]

proposal = dict();
if os.path.exists(imgdir) and os.path.isdir(imgdir):
    imgs = os.listdir(imgdir)
    for img in imgs:
        imgpath = os.path.join(imgdir, img)
        imgopen = subprocess.Popen((cmd, imgpath), stdout=subprocess.PIPE)
        imgopen.wait()
        boxes = []
        while True:
            line = imgopen.stdout.readline()
            if not line:
                break
            coords = line.split(",")
            tl = (int(coords[0]), int(coords[1]))
            br = (int(coords[2]), int(coords[3]))
            boxes.append((tl, br))
        proposal[img] = boxes
else:
    print "error: directory <"+imgdir+"> not found"

# for training only
label = dict();
real = dict();
if os.path.exists(labelf) and os.path.isfile(labelf):
    with open(labelf) as f:
        while True:
            line = f.readline()
            if not line:
                break
            imglabel = line.split(" ")
            img = imglabel[0]
            label[img] = imglabel[1]
            if img not in real:
                real[img] = [];
            if (len(imglabel) > 2):
                labelenum = int(imglabel[2])
                realbox = ((int(imglabel[3]), int(imglabel[4])), (int(imglabel[5]), int(imglabel[6])))
                real[img].append(realbox)

# do something with dictionaries
