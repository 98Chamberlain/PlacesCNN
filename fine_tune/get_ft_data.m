clear all; close all;

load('../gt_scene.mat');
load('../total_label.mat');

IMAGEPATH = '/media/ponu/DATA/Places205_resize/images256/';
file_id = fopen([IMAGEPATH,'LMDB_test.txt'],'r');
H5PATH = '../../finetune_h5/';

data_total = 2000;
prop = 0.8;
train_amt = data_total * prop * prop;
val_amt = data_total * prop * (1-prop);
test_amt = data_total * (1-prop);

root_s = [1,2,3,5:1:22,24,25,26,28:1:38,40:1:49,51:1:63,65,66,...
    67,69:1:75,77:1:81,83:1:94,96:1:99,101:1:119,121:1:138,140,142,143,145,...
    147,148,150:1:158,160,162,163,164,166:1:172,174:1:185,189,190,194,195,196,...
    198,199,201,202,205];
scene_num = length(root_s);

single_gt = zeros(round(test_amt*scene_num),1);  % single label ground truth
h5 = cell(round(test_amt*scene_num),2); % h5 data / correspond places scene number

cnt = 1; % count the number of total test image
tline = fgetl(file_id);
while ischar(tline)
    
    file_content = textscan(tline, '%s %d');
    file_path = file_content{1}{1};
    single_gt(cnt) = file_content{2};
    [pathstr,name,ext] = fileparts(file_path);
    
    % begin to read h5 file
    scene_idx = floor((cnt-1)/test_amt)+1;
    h5_data = hdf5read([H5PATH,name,'.h5'],'prob');
    h5{cnt,1} = h5_data;
    h5{cnt,2} = root_s(scene_idx); 
    
    % cont. to read another line
    tline = fgetl(file_id);
    cnt = cnt + 1;
    
end

save finetune_data.mat h5 single_gt;
