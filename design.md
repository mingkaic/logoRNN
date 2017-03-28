# logoRNN

region based neural nets works as follows:

1. given an input image, use Selective Search to extract the category-independent region proposals (most likely regions to contain objects)

2. the proposals are input into a CNN to extract a fixed length feature vector

3. category-specific linear SVMs classify the feature vectors

## Selective Search

Uijlings and colleges make the following considerations when designing selective search:

- Capture all scales (differing sigma in smoothing operators to capture regions of different details, similar to gaussian pyramid)

- Diversify region identification (differing features)

- Fast: necessary condition for production grade software

## CNN


## SVM


## Citation

[1] J. R. Uijlings, K. E. van de Sande, T. Gevers, and A. W.
Smeulders. Selective search for object recognition. IJCV, 2013.
