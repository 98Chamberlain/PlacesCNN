import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/ponu/caffe/python')
import caffe
import os
import scipy.io as sio
import h5py

MODEL_FILE = 'places205CNN_deploy.prototxt'
PRETRAINED =  'places205CNN_iter_300000.caffemodel'


path = "../../Hitachi/Places205_resize/images256/"
alphabet = os.listdir( path )
path_out = "../images256_h5/"
path_fc8 = "../images256_fc8_h5/"

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

caffe.set_device(0)
caffe.set_mode_gpu()
net = caffe.Classifier(MODEL_FILE,
                       PRETRAINED,                                    
                       mean=np.load('mean.npy').mean(1).mean(1),
		       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256,256))

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
        h5f.create_dataset('dataset_1',data=prediction[0])
        h5f.close()

        fc8 = os.path.join(im_path_fc8 , h5name)
        h5f = h5py.File( fc8 , 'w' )
        h5f.create_dataset('fc8',data=feat)
        h5f.close()

for al in alphabet:
    f = os.listdir( os.path.join( path , al ) )
    for file in f:
        im_path = os.path.join( path , al , file )
        imageset_org = os.listdir( im_path )
        im_path_out = os.path.join( path_out , al , file )
        im_path_fc8 = os.path.join( path_fc8 , al , file )
        if os.path.isdir( os.path.join( im_path , imageset_org[0] ) ):
            for sub_f in imageset_org:
                im_path_tmp = os.path.join( im_path , sub_f )
                imageset_org = os.listdir( im_path_tmp )
                im_path_out_tmp = os.path.join( path_out,al,file,sub_f )
                im_path_fc8_tmp = os.path.join( path_fc8,al,file,sub_f )
                ensure_dir(im_path_out_tmp)
                ensure_dir(im_path_fc8_tmp)
                feed_forward( imageset_org , 128 , im_path_tmp , im_path_out_tmp , im_path_fc8_tmp)
        else:
            ensure_dir(im_path_out)
            ensure_dir(im_path_fc8)
            feed_forward( imageset_org , 128 , im_path , im_path_out , im_path_fc8 )

