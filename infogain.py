import csv
import sys
import numpy as np
sys.path.insert(0,'/home/ponu/caffe/python')
import caffe

f = open('H_invalid_1000.csv','r')
i = 0
L = 45
H = np.eye( L , dtype = 'f4')
for row in csv.reader(f):
    H[i] = row
    i = i+1
# print H

blob = caffe.io.array_to_blobproto( H.reshape( (1,1,L,L) ) )
with open( 'infogainH_invalid_1000.binaryproto', 'wb' ) as f :
    f.write( blob.SerializeToString() )
