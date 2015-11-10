clear all; close all;

task = 'multi_label';

load('../gt_scene.mat');
load('../total_label.mat');

% in out path setting
IMAGEPATH = '/media/ponu/DATA/Places205_resize/images256/';
train_out = [IMAGEPATH,task,'_train.txt'];
val_out = [IMAGEPATH,task,'_val.txt'];
test_out = [IMAGEPATH,task,'_test.txt'];

train_txt = [IMAGEPATH,'LMDB_train.txt'];
val_txt = [IMAGEPATH,'LMDB_val.txt'];
test_txt = [IMAGEPATH,'LMDB_test.txt'];

% parameter setting
scene_num = 205;
label_num = 40;
data_total = 2000;
prop = 0.8;
train_amt = data_total * prop * prop;
val_amt = data_total * prop * (1-prop);
test_amt = data_total * (1-prop);

% set the ground truth label
gt_label = ones(scene_num,label_num);
gt_label = -gt_label;
for scn_index = 1:205
    i = gt_scene(scn_index);
    if i == 0
        gt = [];
    elseif scn_index == 94
        gt = [1,2,6,7];
    elseif scn_index == 123
        gt = [1,3,4,38,39];
    elseif (scn_index == 2) || (scn_index == 11) || (scn_index == 73)
        gt = [1,2,9];
    elseif (scn_index == 30) || (scn_index == 31)
        gt = [1,2,5];
    elseif (sum([48,66,91,103,162,164] == scn_index) >= 1)
        gt = [1,3,4,12,31,35];
    else
        gt = groundtruth{gt_scene(scn_index)};
    end
    gt_label(scn_index,gt) = 1;
end

genLMDB_multi_label( train_txt , train_out , gt_label , train_amt );
genLMDB_multi_label( val_txt , val_out , gt_label , val_amt );
genLMDB_multi_label( test_txt , test_out , gt_label , test_amt );
