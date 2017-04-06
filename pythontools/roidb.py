import os
import numpy as np
import scipy.sparse
if not cpuonly:
    from utils.cython_bbox import bbox_overlaps
else:
    from roi_data_layer.customoverlap import myoverlap

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

classes = {}
truth = {}
idx = 0
with open('./data/annotation/truth.txt') as tf:
    while True:
        line = tf.readline()
        if not line:
            break
        coords = line.split(" ")
        fname = os.path.basename(coords[0]).split(os.extsep)[0]
        if not fname in truth:
            truth[fname] = []
        cname = coords[1]
        if not cname in classes:
            classes[cname] = idx
            idx = idx + 1
        truth[fname].append([cname, int(coords[3]), int(coords[4]), int(coords[5]), int(coords[6])])

numclasses = len(classes)

def roidb_from_fboxes(dir, fboxes, gt_roidb):
    numtest = len(fboxes)
    roidb = []
    for i, f in enumerate(fboxes):
        boxlist = np.array(fboxes[f])
        nboxes = len(fboxes[f])
        overlaps = np.zeros((nboxes, numtest), dtype=np.float32)

        if gt_roidb is not None:
            gt_boxes = gt_roidb[i]['boxes']
            gt_classes = gt_roidb[i]['gt_classes']
            gt_overlaps = []
            if not cpuonly:
                gt_overlaps = bbox_overlaps(boxlist.astype(np.float), gt_boxes.astype(np.float))
            else:
                gt_overlaps = myoverlap(boxlist.astype(np.float), gt_boxes.astype(np.float))
            argmaxes = gt_overlaps.argmax(axis=1)
            maxes = gt_overlaps.max(axis=1)
            I = np.where(maxes > 0)[0]
            overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]

        overlaps = scipy.sparse.csr_matrix(overlaps)

        imgname = './data/'+dir+'/'+f+'.jpg'
        roidb.append({
            'boxes': boxlist,
            'image': imgname,
            'gt_classes': np.zeros((nboxes,), dtype=np.int32),
            'gt_overlaps' : overlaps,
            'flipped' : False
        })
    return roidb

def gt_roidb_from_fboxes(dir, fboxes):
    gt_roidb = []
    numtest = len(fboxes)
    for f in fboxes:
        boxlist = []
        nboxes = len(truth[f])
        overlaps = np.zeros((nboxes, numtest), dtype=np.float32)
        gt_classes = np.zeros((nboxes,), dtype=np.int32)
        for i, bbs in enumerate(truth[f]):
            gt_classes[i] = classes[bbs[0]]
            boxlist.append([bbs[1], bbs[2], bbs[3], bbs[4]])
        boxlist = np.array(boxlist)
        imgname = './data/'+dir+'/'+f+'.jpg'
        gt_roidb.append({
            'boxes' : boxlist,
            'image': imgname,
            'gt_classes' : gt_classes,
            'gt_overlaps' : overlaps,
            'flipped' : False})

    return gt_roidb

def merge_roidbs(a, b):
    assert len(a) == len(b)
    for i in xrange(len(a)):
        a[i]['boxes'] = np.vstack((a[i]['boxes'], b[i]['boxes']))
        a[i]['gt_classes'] = np.hstack((a[i]['gt_classes'],
                                        b[i]['gt_classes']))
        a[i]['gt_overlaps'] = scipy.sparse.vstack([a[i]['gt_overlaps'],
                                                   b[i]['gt_overlaps']])
    return a

def prepare_roidb(roidb):
    for i in range(len(roidb)):
        gt_overlaps = roidb[i]['gt_overlaps'].toarray()
        # max overlap with gt over classes (columns)
        max_overlaps = gt_overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = gt_overlaps.argmax(axis=1)
        roidb[i]['max_classes'] = max_classes
        roidb[i]['max_overlaps'] = max_overlaps
    return roidb

def get_testing_roidb():
    gt_roidb = gt_roidb_from_fboxes('testingset', testfboxes)
    ss_roidb = roidb_from_fboxes('testingset', testfboxes, gt_roidb)
    roidb = merge_roidbs(gt_roidb, ss_roidb)
    return prepare_roidb(roidb)

def get_training_roidb():
    gt_roidb = gt_roidb_from_fboxes('trainingset', trainfboxes)
    ss_roidb = roidb_from_fboxes('trainingset', trainfboxes, gt_roidb)
    roidb = merge_roidbs(gt_roidb, ss_roidb)
    return prepare_roidb(roidb)