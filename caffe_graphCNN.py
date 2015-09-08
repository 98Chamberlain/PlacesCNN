import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'/home/ponu/caffe/python')
import caffe
import os
import scipy.io as sio
import h5py

MODEL_FILE = './graphCNN_deploy.prototxt'
PRETRAINED = './_iter_30000.caffemodel'

caffe.set_device(0)
caffe.set_mode_gpu()
net = caffe.Classifier(MODEL_FILE,PRETRAINED)

hdf5_file = 'graphCNNtesting.h5'
file = h5py.File(hdf5_file,'r')
dataset = file['/data']
# dataset = dataset[1,:]
# file.close()

# print dataset[1,:]

# da = [[]]

for i in range(4872):
# for i in range(10):

#     i = i*28
    dataset_t = dataset[i ,:]
    # print dataset_t
    net.blobs['data'].data[...] = dataset_t
    out = net.forward()
    d = out['th_concat']
    # print d[0]

    h5f = h5py.File( os.path.join('./h5file_t/',"h5_file_"+str(i)+".h5"),'w')
    h5f.create_dataset('data',data=d[0])
    h5f.close()
