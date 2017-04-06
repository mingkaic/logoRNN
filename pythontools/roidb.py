import os

testing_roidb = []
training_roidb = []

testfboxes = {}
with open('./data/testing_bbox.txt') as testf:
    while True:
        line = testf.readline()
        if not line:
            break
        coords = line.split(" ")
        fname = os.path.basename(coords[0]).split(os.extsep)[0]
        if not fname in testfboxes:
            testfboxes[fname] = []
        testfboxes[fname].append([int(coords[1]), int(coords[2]), int(coords[3]), int(coords[4])])

print testfboxes

trainfboxes = {}
with open('./data/training_bbox.txt') as trainf:
    while True:
        line = trainf.readline()
        if not line:
            break
        coords = line.split(" ")
        fname = os.path.basename(coords[0]).split(os.extsep)[0]
        if not fname in trainfboxes:
            trainfboxes[fname] = []
        trainfboxes[fname].append([int(coords[1]), int(coords[2]), int(coords[3]), int(coords[4])])

print trainfboxes


def get_testing_roidb():
    return testing_roidb

def get_training_roidb():
    return training_roidb