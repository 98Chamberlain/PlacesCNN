import sys
sys.path.insert(0, '/home/ponu/caffe/python')
import caffe
import os
import numpy as np
import lmdb
from caffe.proto import caffe_pb2
import h5py

LMDB_PATH = '/media/ponu/DATA/Places205_resize/images256'

name_list = ['train_lmdb','val_lmdb','test_lmdb']

for name in name_list:
    lmdb_env = lmdb.open(LMDB_PATH+'/'+name)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe_pb2.Datum()
    cnt = 0
    label_list = []
    for key, value in lmdb_cursor:
        cnt+=1
        datum.ParseFromString(value)
    
        label = datum.label
        # data = caffe.io.datum_to_array(datum)
        label_list.append(label)
    h5f = h5py.File( LMDB_PATH+'/'+name+'_single_label.h5' , 'w' )
    h5f.create_dataset( 'label' , data = label_list )
    h5f.close()
    print "finished get the label in %s total %d" %(name,cnt)

