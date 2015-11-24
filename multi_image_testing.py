import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/ponu/caffe/python')
import caffe
import os
import scipy.io as sio
import h5py

# task name
task = 'ft_multi_label_inv'

# some path setting
path = '/media/ponu/DATA/Places205_resize/images256'

MODEL_FILE = 'places205CNN_ft_deploy.prototxt'
PRETRAINED =  '/home/ponu/CNNsnapshot_multi_label_inv/_iter_450000.caffemodel'
MEAN_PATH = './mean.npy'
# MEAN_PATH = os.path.join( path , "ft_mean.npy" )

# output path
fc6_out = '/home/ponu/Documents/h5/'+task+'_fc6_h5'
fc7_out = '/home/ponu/Documents/h5/'+task+'_fc7_h5'
fc8_out = '/home/ponu/Documents/h5/'+task+'_fc8_h5'
prob_out = '/home/ponu/Documents/h5/'+task+'_prob_h5'

# testing amount in one forward pass
batch = 64
channel = 3
height = 256
width = 256

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def file_len(fname):
    # with open(fname) as f:
    for i, l in enumerate(fname):
        pass
    return i + 1

# output_path : string
# save_data   : list or array
# label_name  : string
def save_h5( output_path , save_data , label_name ):
    h5f = h5py.File( output_path , 'w')
    h5f.create_dataset( label_name , data = save_data )
    h5f.close()
        
# ensure_dir( path_out )

# convert .mean file to .npy file
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

cnt = 0
h5_list = []
image_set = np.zeros([batch,channel,height,width])
f_len = file_len(file)
file.close()

file = open( os.path.join( path , "LMDB_test.txt" ), 'r')
for line in file:
    cnt += 1
    list_id = (cnt-1)%batch

    # extract the file name
    test_path = line.split()
    test_path = test_path[0]
    (file_name,ext) = os.path.splitext(test_path) # file_name = /path/to/file/filename ext=.jpg

    # path of h5 file
    h5name = file_name+'.h5'
    h5_path = fc6_out + h5name
    
    if not os.path.exists( h5_path ):
        h5_list.append( h5name )
    
        # create image set & path_list
        if ( list_id == batch-1 ) | ( cnt == f_len ) :
            image_path = path+test_path
            input_image = caffe.io.load_image( image_path )
            input_image = input_image[0:width,0:height,0:channel]
            image_set[list_id,:,:,:] = input_image.swapaxes(0,2)
            image_set*=255
            net.blobs['data'].data[...] = image_set
            net.forward()
            for i, f_name in enumerate(h5_list):
            
                fc6_path = fc6_out + f_name
                ensure_dir( os.path.dirname(fc6_path) )
                save_h5( fc6_path , net.blobs['fc6'].data[i] , 'fc6' )
                
                fc7_path = fc7_out + f_name
                ensure_dir( os.path.dirname(fc7_path) )
                save_h5( fc7_path , net.blobs['fc7'].data[i] , 'fc7' )
                
                fc8_path = fc8_out + f_name
                ensure_dir( os.path.dirname(fc8_path) )
                save_h5( fc8_path , net.blobs['fc8'].data[i] , 'fc8' )

                prob_path = prob_out + f_name
                ensure_dir( os.path.dirname(prob_path) )
                save_h5( prob_path , net.blobs['prob'].data[i] , 'prob' )
            h5_list = []
            image_set = np.zeros([batch,channel,height,width])
            print "finished batch, size %d" % (i)

        else:
            image_path = path+test_path
            input_image = caffe.io.load_image( image_path )
            input_image = input_image[0:width,0:height,0:channel]
            image_set[list_id,:,:,:] = input_image.swapaxes(0,2)
                
#                h5f = h5py.File( h5_path , 'w')
#                h5f.create_dataset('prob',data=prediction[0])
#                h5f.close()
            
#                input_image = caffe.io.load_image( image_path )
#                prediction = net.predict([input_image])
#                print 'predicted class:', prediction[0].argmax()
#                h5f = h5py.File( h5_path , 'w')
#                h5f.create_dataset('prob',data=prediction[0])
#                h5f.close()

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
