import numpy as np
import matplotlib.pyplot as plt
import random

import sys
sys.path.append('/home/ponu/caffe-lstm/python')
import caffe
import os
import scipy.io as sio
import h5py
sys.path.append('/home/ponu/caffe-lstm/examples/coco_caption')

path = '/media/ponu/DATA/Places205_resize/images256'

vocabulary = ['<EOS>'] + [line.strip() for line in open('/home/ponu/caffe-lstm/examples/places_caption/h5_data/buffer_100/vocabulary.txt').readlines()]
print len(vocabulary)

iter_num = 12500
net = caffe.Net('/home/ponu/caffe-lstm/examples/places_caption/lrcn_all_deploy.prototxt',
                '/home/ponu/caffe-lstm/examples/places_caption/lrcn/lrcn_iter_%d.caffemodel' % iter_num, caffe.TEST)

def predict_single_word(net, previous_word, image, output='probs'):
    cont = 0 if previous_word == 0 else 1
    cont_input = np.array([cont])
    word_input = np.array([previous_word])
    net.forward(data=image, cont_sentence=cont_input, input_sentence=word_input)
    output_preds = net.blobs['predict'].data[0, 0, :]
    return output_preds

# load image
batch = 1
channel = 3
height = 227
width = 227

file = open( os.path.join( path , "LMDB_test.txt" ), 'r')
print [(k, v.data.shape) for k, v in net.blobs.items()]
cnt = 0
for line in file:
    cnt += 1
    list_id = (cnt-1)%batch

    # extract the file name
    test_path = line.split()
    test_path = test_path[0]
    (file_name,ext) = os.path.splitext(test_path) # file_name = /path/to/file/filename ext=.jpg

    # path of h5 file
    h5name = file_name+'.h5'
    h5_path = './' + h5name

    image_path = path+test_path
    input_image = caffe.io.load_image( image_path )
    input_image = input_image[0:width,0:height,0:channel]

    input_image = input_image.swapaxes(0,2)
    input_image*=255
    input_image[0,:,:] = input_image[0,:,:]-123
    input_image[1,:,:] = input_image[1,:,:]-117
    input_image[2,:,:] = input_image[2,:,:]-104
    input_i = np.zeros([batch,channel,height,width])
    input_i[0,:,:,:] = input_image

    first_word_dist = predict_single_word(net, 0, input_i)
    top_preds = np.argsort(-1 * first_word_dist)
    print top_preds[:10]
    print [vocabulary[index] for index in top_preds[:10]]

    second_word_dist = predict_single_word(net, vocabulary.index('indoor'), input_i)
    print [vocabulary[index] for index in np.argsort(-1 * second_word_dist)[:10]]


