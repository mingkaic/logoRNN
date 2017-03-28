# logoRNN

region based neural nets works as follows:

1. given an input image, use Selective Search to extract the category-independent region proposals (most likely regions to contain objects)

2. the proposals are input into a CNN to extract a fixed length feature vector

3. category-specific linear SVMs classify the feature vectors

Fast RNN improves the above approach by replacing the CNN and SVM pipeline for R-CNN or RNN.

Object proposals are still necessary to for untrained NN (hence selective search).

## Selective Search [1]

Uijlings and colleges make the following considerations when designing selective search:

- Capture all scales (differing sigma in smoothing operators to capture regions of different details, similar to gaussian pyramid)

- Diversify region identification (differing features)

- Fast: necessary condition for production grade software

The specific segmentation method, by J. Carreira and C. Sminchisescu [2], is beyond the scope of this project, 
so for now, segment by applying canny edge detection then flooding.

1. Hierarchically group segments


    R = {r1, ..., rn}
    S = null
    for (adjacent regions: {ri, rj}
        s(ri, rj) = similarity between ri and rj
        if threshold < s(ri, rj)
            S = S + s(ri, rj)
    
    while (S is not empty)
        s(ri, rj) = max(S)
        rt = ri merge rj
        remove ri and rj similarities in S
        add similarities for rt and neighbors in S
        R = R + rt - ri - rj

2. 

## Fast RNN


## Citation

[1] J. R. Uijlings, K. E. van de Sande, T. Gevers, and A. W.
Smeulders. Selective search for object recognition. IJCV, 2013.

[2] J. Carreira and C. Sminchisescu. Constrained parametric mincuts
for automatic object segmentation. In CVPR, 2010. 2, 3,
8, 9, 10, 11, 13