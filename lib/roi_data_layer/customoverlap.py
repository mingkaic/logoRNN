import numpy as np

def area(bbox):
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]
    return (x2 - x1) * (y2 - y1)

def overlap_ratio(bboxa, bboxb):
    xa1 = bboxa[0]
    ya1 = bboxa[1]
    xa2 = bboxa[2]
    ya2 = bboxa[3]
    xb1 = bboxb[0]
    yb1 = bboxb[1]
    xb2 = bboxb[2]
    yb2 = bboxb[3]

    iLeft = max(xa1, xb1)
    iRight = min(xa2, xb2)
    iTop = max(ya1, yb1)
    iBottom = min(ya2, yb2)

    si = max(0, iRight - iLeft) * max(0, iBottom - iTop)
    sa = (xa2 - xa1) * (ya2 - ya1)
    sb = (xb2 - xb1) * (yb2 - yb1)
    intersect = sa + sb - si

    totalarea = area(bboxa) + area(bboxb)
    return intersect / totalarea

def myoverlap (bboxa, bboxb):
    olap = np.zeros((len(bboxa), len(bboxb)), dtype=np.float32)
    for i in range(len(bboxa)):
        for j in range(len(bboxb)):
            olap[i][j] = overlap_ratio(bboxa[i], bboxb[j])
    return olap