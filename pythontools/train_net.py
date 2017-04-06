#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network on a region of interest database."""

import __builtin__
import argparse
import pprint
import numpy as np
import sys
import os

cpuonly = False

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=40000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--cpu-only', dest='cpuonly',
                        help='execute cpu only mode',
                        action='store_true')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    __builtin__.cpuonly = args.cpuonly

    import _init_paths
    import caffe
    from fast_rcnn.config import cfg, cfg_from_file, get_output_dir
    from fast_rcnn.train import train_net#, get_training_roidb
    from roidb import get_training_roidb
    # from datasets.factory import get_imdb

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)
        caffe.set_random_seed(cfg.RNG_SEED)

    # set up caffe
    if not args.cpuonly:
        print "setting gpu mode"
        caffe.set_mode_gpu()
        if args.gpu_id is not None:
            caffe.set_device(args.gpu_id)

    # imdb = get_imdb('voc_2007_trainval')
    # print 'Loaded dataset `{:s}` for training'.format(imdb.name)
    # roidb = get_training_roidb(imdb)
    # output_dir = get_output_dir(imdb, None)
    # print 'Output will be saved to `{:s}`'.format(output_dir)


    roidb = get_training_roidb()
    output_dir = os.path.join('.', 'data', 'out')

    train_net(args.solver, roidb, output_dir,
              pretrained_model=args.pretrained_model,
              max_iters=args.max_iters)
