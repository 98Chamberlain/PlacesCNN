import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'/home/ponu/caffe/python')
import caffe
import os
import scipy.io as sio
import h5py

MODEL_FILE = './graphCNN_deploy.prototxt'
PRETRAINED = './_iter_10000.caffemodel'

caffe.set_device(0)
caffe.set_mode_gpu()
net = caffe.Classifier(MODEL_FILE,PRETRAINED)

hdf5_file = 'graphCNNtesting.h5'
file = h5py.File(hdf5_file,'r')
dataset = file['/data']
# dataset = dataset[1,:]
# file.close()

# da = [[]]
# for i in range(28):
    
dataset_t = dataset[1 ,:]
net.blobs['data'].data[...] = dataset_t
out = net.forward()
# print out['fc1']
h5f = h5py.File( './h5file/h5_file_test.h5','w')
h5f.create_dataset('40',out['fc1'])
h5f.close()
