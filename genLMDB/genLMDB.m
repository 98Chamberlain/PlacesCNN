clear all; close all;

load('../gt_scene.mat');
load('../total_label.mat');

IMAGEPATH = '/media/ponu/DATA/Places205_resize/images256/';
trainID = fopen('LMDB_train.txt','w');
valID = fopen('LMDB_val.txt','w');
testID = fopen('LMDB_test.txt','w');

data_total = 2000;
prop = 0.8;
train_amt = data_total * prop * prop;
val_amt = data_total * prop * (1-prop);
test_amt = data_total * (1-prop);

for i = 1:205
    if gt_scene(i) ~= 0
        image = dir( [IMAGEPATH , total_label{i+40,2},'/*.jpg'] );
        for j = 1:data_total
            if j>=1 && j<=train_amt
                fprintf(trainID,'%s %d\n',[total_label{i+40,2},'/',image(j).name],gt_scene(i));
            elseif j>= train_amt+1 && j<= train_amt+val_amt
                fprintf(valID,'%s %d\n',[total_label{i+40,2},'/',image(j).name],gt_scene(i));
            else
                fprintf(testID,'%s %d\n',[total_label{i+40,2},'/',image(j).name],gt_scene(i));
            end
        end
    end
end
        


        
% [first_dir,num_first] = subdir( IMAGEPATH );
% 
% for i = 1:num_first
%     [second_dir,num_second] = subdir( [ IMAGEPATH,first_dir(i).name,'/' ] );
%     if num_second == 0
%         image = dir( [ IMAGEPATH,first_dir(i).name,'/*.jpg' ] );
%         
% end