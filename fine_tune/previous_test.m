clear all; close all;

addpath(genpath('../../hex_graph-master'));
addpath('../../Research_Toolkit/SVM/libsvm-3.20/matlab');
addpath(genpath('../'));
addpath(genpath('../../scene-classification'));


% % ----- load the data -----
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
data_len = size(h5,1);
train_amt = round(data_len*0.8); % 2000 -> 1600
test_amt = data_len - train_amt;

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
% E_e = E_e | E_e';

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

% level information
level = {};
% level{0} = [1];
level{1} = [2,3,5,9];
level{2} = [4,6,7,8,10,21,22];
level{3} = [11:20,23:28,36:40];
level{4} = [29:35];

% children information
children = children_run(adj_mat);

% % train SVM model for each label
train_data = h5;
[model,mf,nrm] = prob_SVM(train_data',test_amt);

acc = [];
FP = [];
cnt_gt_label = zeros(40,1);
cnt_ac_label = zeros(40,1);
cnt_FP_label = zeros(40,1);
% for s_id = 1:1
for s_id = 1:use_scene
    disp(['-------------------- now process ',num2str(s_id),'/',num2str(use_scene),' scene --------------------']);
    for data_id = 1:round(test_amt)
    img_id = floor( ((s_id-1)*test_amt)+data_id );
    scn_index = root_s(s_id);
    sum_prob = h5(img_id,:);
%     label = gt_scene(scn_index);
%     data_t = data(root_s);
%     [~,idx] = max(data_t);
%     use_gt = gt_scene(root_s);
%     label = use_gt(idx);
       
%     % run the hex graph
%     back_propagate = true;
%     [loss, gradients, p_margin, p0] = hex_run(G, sum_prob, label, back_propagate);
%     state = G.c_s_cell{1};
%     state_prob = state * p_margin;
%     [~,I] = max(state_prob);
%     result = find(state(I,:)==1);

% ----- decision tree begin -----
result = [1];
cont = true;
parent_list = 1;
while( cont )
child_list = [];
for p = 1:length(parent_list)
    % child_list = [child_list , children{p}];
    child_list = children{parent_list(p)};
end
child_list = unique(child_list);

% Use SVM to predict each label
predict = [];
for child = 1:length(child_list)
    c = child_list(child);
    [m2,N]=size(data);
    fea_tmp=(data-ones(m2,1)*mf{c})*nrm{c};
    [predicted, accuracy, d_values] = svmpredict(1 , fea_tmp , model{c});
    predict = [predict,predicted];
end
child_list = child_list(find(predict==1));

if ~isempty(child_list)
    result_list = child_list(1);
    % check isvalid
    finished = false;
else
    finished = true;
    cont = false;
end
% begin check
new_result_list = [];
while( ~finished )
    old_result_list = new_result_list;
    for c = 1:length(child_list)
        for r = 1:length(result_list)
            notvalid(r) = (adj_mat( result_list(r),child_list(c) )==1) && (adj_mat( child_list(c),result_list(r) )==1);
        end
        if (sum(notvalid) > 0)
            if p_margin(child_list(c)) > sum(p_margin(result_list))
                result_list = child_list(c);
            end
        else
            result_list = [child_list(c),result_list];
        end
    end
    new_result_list = sort(unique(result_list));
    if isequal(new_result_list,old_result_list)
        finished = true;
        parent_list = new_result_list;
        result = [result,parent_list];
    end
end

label = new_result_list;
back_propagate = true;
[loss, gradients, p_margin, p0] = hex_run(G, sum_prob, label, back_propagate);

end

feature = 1;
result = searchBest_hr(adj_mat,(p_margin./max(p_margin)),feature,model,mf,nrm);
% ----- decision tree end -----

% result_total{data_id} = result;

% % ----- SVM feedback test -----
% label = gt_scene( scn_index );
% 
% [m2,N]=size(data');
% fea_tmp=(data'-ones(m2,1)*mf)*nrm;
% [predicted, accuracy, d_values] = svmpredict(label , fea_tmp , model);
% 
% label = predicted;
% back_propagate = true;
% [loss, gradients, p_margin, p0] = hex_run(G, sum_prob, label, back_propagate);
% 
% feature = 1;
% result = searchBest_hr(adj_mat,(p_margin./max(p_margin)),feature,model,mf,nrm);
% % ----- SVM feedback test end -----
        
%         [~,I] = max(gradients);
%         result = groundtruth{I};
        
        % use past structure to run the multi-label relation
%         feature = 1;
%         model = 1;
%         mf = 1;
%         nrm = 1;
%         result = searchBest_hr(adj_mat,(p_margin./max(p_margin)),feature,model,mf,nrm);

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

