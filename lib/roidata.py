import caffe
import numpy as np

class ROIBlobLayer(caffe.Layer):

    def setup(self, bottom, top):
          """Setup the RoIDataLayer."""

          params = eval(self.param_str)
          src_file = params["src_file"]
          self.batch_size = params["batch_size"]
          self.im_shape = params["im_shape"]
          self.crop_size = params.get("crop_size", False)
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)

        self._num_classes = layer_params['num_classes']

        self._name_to_top_map = {
            'data': 0,
            'rois': 1,
            'labels': 2}
        self.imgTuples = readSrcFile(src_file)
        # data blob: holds a batch of N images, each with 3 channels
        # The height and width (100 x 100) are dummy values
        top[0].reshape(1, 3, 100, 100)

        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        top[1].reshape(1, 5)


    def reshape(self, bottom, top):


    def forward(self, bottom, top):
        #in forward, you want to extract the ROI from the image and feed it to the next layer
                # Copy all of the data
        #parse the textfile and segment image. for all regions of interest crop photo accordingly and send the
        # image through.


    def backward(self, top, propagate_down, bottom):
        pass
