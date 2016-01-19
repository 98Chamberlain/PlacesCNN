import json
import random
import numpy as np
import scipy.io as sio

TRAIN_IMAGE = '../genLMDB/train_with_label.txt'
VAL_IMAGE = '../genLMDB/val_with_label.txt'
TEST_IMAGE = '../genLMDB/test_with_label.txt'

dataset = [('TRAIN','train'),('VAL','val'),('TEST','test')]

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def file_len(fname):
    # with open(fname) as f:
    for i, l in enumerate(fname):
        pass
    return i + 1
	
file = open( TEST_IMAGE, 'r')

train_len = file_len(file)
file.close()
print 'got the length ', train_len

train_path = np.zeros(train_len,dtype=object)
train_label = np.zeros(train_len)
file = open( TEST_IMAGE, 'r')
for i,line in enumerate(file):

    # extract the file name
    txt_split = line.split()
    train_path[i] = txt_split[0]
    train_label[i] = txt_split[1] # label in 205
	
print 'extracted the image path and label'
	
index = [i for i in range(train_len)]
random.shuffle(index)
index = np.array(index)

train_path_shuffle = train_path[index]
train_label_shuffle = train_label[index]

id_file = open( './test_id.txt' , 'w')
for i in index:
    id_file.write("%d\n" %i)

print 'shuffled the data'

# some information 
info = {"year": 2015, \
        "version": 'v1.0', \
		"description": 'This data is for scene classification', \
		"contributor": 'Chen Po-Jen, Ding Jian-Jiun', \
		"url": 'not yet supported', \
		"data_created": '2015-12-29 05:00:00.000000',}

licenses = {"id": 1, "name": 'Places Dataset', "url": 'http://places.csail.mit.edu/',}

print 'set information in json type'

# produce the image list in json type
images = [{} for i in range(train_len)]
for i in range(train_len):
	
	image = {"id": i, "width": 256, "height": 256, "file_name": train_path_shuffle[i], \
	         "license": 1, "flickr_url": 'none', "coco_url": 'none', "date_captured": '2015-12-29 05:00:00.000000',}
	images[i] = image
	
print 'converted the image list to json type'
print images[0]

# produce the annotations list in json type
annotations = [{} for i in range(train_len)]
label_file = open( './scene_label.txt', 'r')
for line in label_file:
    label_list = line.split()
label_list = np.array(label_list)
# label_list = ['root','indoor','outdoor','landscape','sports','home','work_place','store',\
#               'industrial','architecture','plants','water', \
#               'bedroom','dining_room','living_room','kitchen', \
# 			  'conference_room','office','restaurant','supermarket','aquarium','class_room', \
# 			  'church','bridge','skycraper','amusement_park','playground','street', \
# 			  'forest','garden','coast','pond','river','swimming_pool','ocean','canyon','desert','ice','mountain','sky']
mat = sio.loadmat('../gt_scene.mat')
gt_scene = mat['gt_scene'][0] # usage: gt_scene[i] for i-th scene
groundtruth = mat['groundtruth'][0] # usage: groundtruth[i][0] for i-th label
for i in range(train_len):
	
    scn_index = train_label_shuffle[i]
    if scn_index == 94:
        gt = np.array([1,2,6,7])
    elif scn_index == 123:
        gt = np.array([1,3,4,38,39])
    elif (scn_index == 2) | (scn_index == 11) | (scn_index == 73):
        gt = np.array([1,2,9])
    elif (scn_index == 30) | (scn_index == 31):
        gt = np.array([1,2,5])
    elif (sum([48,66,91,103,162,164] == scn_index) >= 1):
        gt = np.array([1,3,4,12,31,35])
    else:
        gt = groundtruth[gt_scene[scn_index-1]-1][0]

    caption = ' '.join(label_list[gt-1])
    annotation = {"id": i, "image_id": i, "caption":caption}
    annotations[i] = annotation

print 'converted the annotation list to json type'
print annotations[0]
	
data = {"info": info, "licenses": licenses, "images": images, "annotations": annotations}
with open('./test_scn.json','w') as outfile:
    json.dump(data, outfile)
