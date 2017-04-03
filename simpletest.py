import sys
sys.path.insert(0, '/Users/maxxface/caffe/python')
import caffe
import cv2
import numpy as np

net = caffe.Net('models/train.prototxt', caffe.TEST)


img = cv2.imread('tests/imgs/test1.jpg', 0)
img_blobinp = img[np.newaxis, np.newaxis, :, :]
net.blobs['data'].reshape(*img_blobinp.shape)
net.blobs['data'].data[...] = img_blobinp

net.forward()
cv2.imwrite('output_image_ .jpg', 255*net.blobs['roi_pool5'].data[0,0])
