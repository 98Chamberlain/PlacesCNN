clear all; close all;

addpath(genpath('../../hex_graph-master'));
addpath('../../Research_Toolkit/SVM/libsvm-3.20/matlab');
addpath(genpath('../'));


% % ----- load the data -----
% load the label name
t = load('../total_label.mat');

% load data
% h5: 69600 x 2 ( test_data*scene_num x (data + correspond scene) )
% single_gt: 69600 x 1 ( single label ground truth )
load('./finetune_data.mat'); 

% load ground truth
% groundtruth = {[1],[1,2],[1,3],[1,3,4],[1,5],[1,2,6],[1,2,7],[1,2,8],[1,2,9],[1,3,10],[1,3,4,11],[1,3,4,12],...
%     [1,2,6,13],[1,2,6,14],[1,2,6,15],[1,2,6,16],[1,2,7,17],[1,2,7,18],[1,2,8,19],[1,2,8,20],[1,2,21],[1,2,22],...
%     [1,3,10,23],[1,3,10,24],[1,3,10,25],[1,3,10,26],[1,3,10,27],[1,3,10,28],[1,3,4,11,29],[1,3,4,11,30],[1,3,4,12,31],...
%     [1,3,4,12,32],[1,3,4,12,33],[1,3,4,12,34],[1,3,4,12,35],[1,3,36],[1,3,37],[1,3,38],[1,3,39],[1,3,40]};
% groundtruth 1*40 cell
% gt_scene 1*205 double
load('../gt_scene.mat'); % gt_scene , groundtruth

% % ----- data setting ----
% set the parameter
data_total = 2000;
prop = 0.8;
train_amt = data_total * prop * prop;
val_amt = data_total * prop * (1-prop);
test_amt = data_total * (1-prop);

% data pre-process
root_s = [1,2,3,5:1:22,24,25,26,28:1:38,40:1:49,51:1:63,65,66,...
    67,69:1:75,77:1:81,83:1:94,96:1:99,101:1:119,121:1:138,140,142,143,145,...
    147,148,150:1:158,160,162,163,164,166:1:172,174:1:185,189,190,194,195,196,...
    198,199,201,202,205];
use_scene = length(root_s);

acc = [];
FP = [];
cnt_gt_label = zeros(40,1);
cnt_ac_label = zeros(40,1);
cnt_FP_label = zeros(40,1);
% for s_id = 1:1
for s_id = 1:use_scene
    disp(['-------------------- now process ',num2str(s_id),'/',num2str(use_scene),' scene --------------------']);
    for data_id = 1:test_amt
        img_id = ( (s_id-1)*test_amt )+data_id;
        scn_index = root_s(s_id);
        data = h5{img_id,1}; % data: 40 x 1
        [~,I] = max(data);
        result = groundtruth{max(data)};

        if scn_index == 94
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
        acc = [acc,length(intersect(result,gt))/length(gt)];
        FP = [FP,(length(setdiff(result,intersect(result,gt))))];
        
        cnt_gt_label(gt) = cnt_gt_label(gt)+1;
        cnt_ac_label(intersect(result,gt)) = cnt_ac_label(intersect(result,gt))+1;
        cnt_FP_label(setdiff(result,intersect(result,gt))) = cnt_FP_label(setdiff(result,intersect(result,gt)))+1;
    
    end
end

% display the result
for s_id = 1:use_scene
disp(['scene ',num2str(s_id),' mean accuracy: ',num2str(mean(acc((s_id-1)*data_len+1:s_id*data_len))),...
        ', sum FP: ',num2str(sum(FP((s_id-1)*data_len+1:s_id*data_len)))])
end
disp(['total mean accuracy: ',num2str(mean(acc)),', sum FP: ',num2str(sum(FP))])

for i = 1:40
    disp(['Label ',num2str(i),' mean accuracy: ',num2str(cnt_ac_label(i)/cnt_gt_label(i)),...
        ', sum FP: ',num2str(cnt_FP_label(i))])
end
disp(['total mean accuracy: ',num2str(sum(cnt_ac_label)/sum(cnt_gt_label)),', sum FP: ',num2str(sum(cnt_FP_label))])

