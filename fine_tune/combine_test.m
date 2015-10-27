clear all; close all;

addpath(genpath('../../hex_graph-master'));
addpath('../../Research_Toolkit/SVM/libsvm-3.20/matlab');
addpath(genpath('../'));
addpath(genpath('../../scene-classification'));

% ----- load data -----
% turn the .h5 data into mat
% get_ft_data
get_original_data

% load the label name
t = load('../total_label.mat');

% load data
% single_gt: 69600 x 1 ( single label ground truth )
% cor_scene: 69600 x 1 ( correspond scene )
% h5: 69600 x 40 ( test_data*scene_num x feature_len )
% ori_h5: 69600 x 205 ( test_data*scene_num x feature_len )
load('./finetune_data.mat');
load('./original_data.mat');

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

% set hierarchy adjacency matrix
adj_mat = genAdj( 40 , 'exclusive');
adj_mat = setHier( adj_mat , 6 , [13,14,15,16] ); % home
adj_mat = setHier( adj_mat , 7 , [17,18] ); % work_place
adj_mat = setHier( adj_mat , 8 , [19,20] ); % store
adj_mat = setHier( adj_mat , 2 , [6,7,8,21,22] ); % indoor
adj_mat = setHier( adj_mat , 10 , [23,24,25,26,27,28] ); % building
adj_mat = setHier( adj_mat , 11 , [29,30] ); % plants
adj_mat = setHier( adj_mat , 12 , [31:1:35] ); % water
adj_mat = setHier( adj_mat , 4 , [11:1:12 36:1:40] ); % landscape
adj_mat = setHier( adj_mat , 3 , [10,4] ); % outdoor
adj_mat = setHier( adj_mat , 1 , [2,3,9,5] ); % root
E_h = logical(adj_mat);

% set Exclusive adjacency matrix
adj_mat = genAdj( 40 , 'exclusive');
adj_mat = setRela( adj_mat , 6 , 7 ); % work_place - home
adj_mat = setRela( adj_mat , 2 , [5,9] ); % indoor - sports/industrial
adj_mat = setRela( adj_mat , 38 , 39 ); % mountain - ice
adj_mat = setRela( adj_mat , 31 , 35 ); % coast - ocean
adj_mat(1,:) = zeros(1,40);
adj_mat(:,1) = zeros(40,1);
E_e = logical(adj_mat);
E_e = E_e | E_e';

% full adj_mat
adj_mat = make_SceneMatrix();

tic;
G = hex_setup(E_h, E_e);
toc

% data pre-process
root_s = [1,2,3,5:1:22,24,25,26,28:1:38,40:1:49,51:1:63,65,66,...
    67,69:1:75,77:1:81,83:1:94,96:1:99,101:1:119,121:1:138,140,142,143,145,...
    147,148,150:1:158,160,162,163,164,166:1:172,174:1:185,189,190,194,195,196,...
    198,199,201,202,205];
use_scene = length(root_s);

% % train SVM model for each label
train_data = h5;
label = zeros(test_amt*use_scene,1);
for i=1:length(train_data)
    s_id = floor((i-1)/test_amt)+1;
    g = size(find(groundtruth{gt_scene(root_s(s_id))}==3),2);
    label(i) = g;
end
[model,mf,nrm] = training_svm( features , label );

acc = [];
FP = [];
cnt_gt_label = zeros(40,1);
cnt_ac_label = zeros(40,1);
cnt_FP_label = zeros(40,1);
% for s_id = 1:1
for s_id = 1:use_scene
    disp(['-------------------- now process ',num2str(s_id),'/',num2str(use_scene),' scene --------------------']);
    for data_id = 1:round(test_amt)
        img_id = floor(( (s_id-1)*test_amt )+data_id);
        scn_index = root_s(s_id);
        data = h5(img_id,:); % data: 40 x 1
        
        [m2,N]=size(data);
        fea_tmp=(data-ones(m2,1)*mf)*nrm;
        [predicted, accuracy, d_values] = svmpredict(1 , fea_tmp , model); % 1 represent containing label 3 (outdoor)
        
        if predicted ~= 1
            if ori_h5(img_id,1)~=0
                data = sumProb_p(ori_h5(img_id,:));
            end
        end
            
        label = single_gt(img_id);
        back_propagate = false;
        [loss, gradients, p_margin, p0] = hex_run(G, data, label, back_propagate);

        feature = 1;
        result = searchBest_hr(adj_mat,(p_margin./max(p_margin)),feature,model,mf,nrm);

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
disp(['scene ',num2str(s_id),' mean accuracy: ',num2str(mean(acc( floor((s_id-1)*test_amt+1:s_id*test_amt)))),', sum FP: ',num2str(sum(FP( floor((s_id-1)*test_amt+1:s_id*test_amt))))])
end
disp(['total mean accuracy: ',num2str(mean(acc)),', sum FP: ',num2str(sum(FP))])

for i = 1:40
    disp(['Label ',num2str(i),' mean accuracy: ',num2str(cnt_ac_label(i)/cnt_gt_label(i)),...
        ', sum FP: ',num2str(cnt_FP_label(i))])
end
disp(['total mean accuracy: ',num2str(sum(cnt_ac_label)/sum(cnt_gt_label)),', sum FP: ',num2str(sum(cnt_FP_label))])

