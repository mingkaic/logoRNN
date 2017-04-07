import __builtin__
__builtin__.cpuonly = True
import _init_paths
import numpy as np
from roidb import get_testing_roidb, get_training_roidb

roidb = get_testing_roidb()
roidb2 = get_training_roidb()

correct = 0
total = 0
for roi in roidb:
    overlaps = roi['gt_overlaps']
    for coord in np.argwhere(overlaps != 0):
        olap_val = overlaps.toarray()[coord[0]][coord[1]]
        if olap_val == 1:
            correct = correct + 1
        total = total + 1
for roi in roidb2:
    overlaps = roi['gt_overlaps']
    for coord in np.argwhere(overlaps != 0):
        olap_val = overlaps.toarray()[coord[0]][coord[1]]
        if olap_val == 1:
            correct = correct + 1
        total = total + 1
print correct
print total
print float(correct) / total