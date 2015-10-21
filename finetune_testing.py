import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/ponu/caffe/python')
import caffe
import os
import scipy.io as sio
import h5py

path = "/media/ponu/DATA/Places205_resize/images256/"

MODEL_FILE = 'places205CNN_ft_deploy.prototxt'
PRETRAINED =  './finetune_model/_iter_450000.caffemodel'
MEAN_PATH = os.path.join( path , "ft_mean.npy" )


alphabet = os.listdir( path )
path_out = "../images256_h5/"

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

# python convert_protomean.py proto.mean out.npy
if not os.path.exists( MEAN_PATH ):
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open( os.path.join( path , "Places_mean.binaryproto" ) , 'rb' ).read()
    blob.ParseFromString(data)
    arr = np.array( caffe.io.blobproto_to_array(blob) )
    out = arr[0]
    np.save( MEAN_PATH , out )

caffe.set_device(0)
caffe.set_mode_gpu()
net = caffe.Classifier(MODEL_FILE,
                       PRETRAINED,                                    
                       mean=np.load( MEAN_PATH ).mean(1).mean(1),
		               channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256,256))

file = open( os.path.join( path , "LMDB_test.txt" ), 'r')

for line in file:
    test_path = line.split()[0]
    input_image = caffe.io.load_image(os.path.join( path , test_path ))
    prediction = net.predict([input_image])
    print 'predicted class:', prediction[0].argmax()


def feed_forward( imageset_org , num , im_path , im_path_out , im_path_fc8):
    imageset = [0]*num
    for i in xrange(num):
        imageset[i] = imageset_org[i]
    for file in imageset:
        input_image = caffe.io.load_image(os.path.join( im_path , file ))

        prediction = net.predict([input_image])
        print 'predicted class:', prediction[0].argmax()
        feat = net.blobs['fc8'].data[0]

        h5name = os.path.splitext(os.path.basename(file))[0]+'.h5'
        hn = os.path.join(im_path_out , h5name)
        h5f = h5py.File( hn , 'w' )
        h5f.create_dataset('prob',data=prediction[0])
        h5f.close()

        fc8 = os.path.join(im_path_fc8 , h5name)
        h5f = h5py.File( fc8 , 'w' )
        h5f.create_dataset('fc8',data=feat)
        h5f.close()
