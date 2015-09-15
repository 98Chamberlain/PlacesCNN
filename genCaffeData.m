% parameter
nScene = 205;
nLabel = 40;
total = 128;
train_amt = 100;
test_amt = total-train_amt;

% load fc8_data, prob_data and groundtruth
load('./feature_data.mat');
% prob_data: 205*128*205
% fc8_data : 205*128*205
% gt : 205 * 40   % can load from following code (prob_SVM.m)

% load groundtruth
load('./gt_scene.mat');
% groundtruth 1*40 cell
% gt_scene 1*205 double

% gt = 205 * 40
gt = zeros(nScene,nLabel);
for i = 1:nScene
    if i == 94
        gt(i,[1,2,6,7]) = 1;
    elseif i == 123
        gt(i,[1,3,4,38,39]) = 1;
    elseif (sum([48,66,91,103,162,164] == i) >= 1)
        gt(i,[1,3,4,12,31,35]) = 1;
    elseif (sum([2,11,73] == i) >= 1)
        gt(i,[1,2,9]) = 1;
    elseif (sum([30,31] == i) >= 1)
        gt(i,[1,2,5]) = 1;
    elseif gt_scene(i)~=0
        gt(i,groundtruth{gt_scene(i)}) = 1;
    end
end

root_s = [1,2,3,5:1:22,24,25,26,28:1:38,40:1:49,51:1:63,65,66,...
    67,69:1:75,77:1:81,83:1:94,96:1:99,101:1:119,121:1:138,140,142,143,145,...
    147,148,150:1:158,160,162,163,164,166:1:172,174:1:185,189,190,194,195,196,...
    198,199,201,202,205];

train_amt = 100;

% % Part1: for 40 label training data
% % training data
% tr_prob = fc8_data(:,1:train_amt,root_s);
% tr_prob = tr_prob(:,:);
% gt_tmp = gt(root_s,:);
% for i = 1:174
%     tr_gt(:,(i-1)*100+1:i*100) = repmat(gt_tmp(i,:),[train_amt,1])'; % 40*17400 -> 17400*40
% end
% 
% hdf5write('graphCNNtraining_fc8.h5','/data',tr_prob,'/label',tr_gt);
% tr_gt_neg = tr_gt;
% tr_gt_neg(find(tr_gt_neg==0))=-1;
% hdf5write('graphCNNtraining_fc8_neg.h5','/data',tr_prob,'/label',tr_gt_neg);
% 
% % testing data
% te_prob = fc8_data(:,train_amt+1:128,root_s);
% te_prob = te_prob(:,:);
% for i = 1:174
%     te_gt(:,(i-1)*28+1:i*28) = repmat(gt_tmp(i,:),[28,1])';
% end
% 
% hdf5write('graphCNNtesting_fc8.h5','/data',te_prob,'/label',te_gt);
% te_gt_neg = te_gt;
% te_gt_neg(find(te_gt_neg==0))=-1;
% hdf5write('graphCNNtesting_fc8_neg.h5','/data',te_prob,'/label',te_gt_neg);


% Part2: training data for infogain loss
% ig = 1 * 174
ig = zeros( 1 , nScene );
ig = gt_scene;
ig(94) = 41;
ig(123) = 42;
ig([48,66,91,103,162,164]) = 43;
ig([2,11,73]) = 44;
ig([30,31]) = 45;
ig = ig(1,root_s);

% training data
tr_prob = fc8_data(:,1:train_amt,root_s);
tr_prob = tr_prob(:,:);
for i = 1:174
    tr_gt(1,(i-1)*train_amt+1:i*train_amt) = repmat(ig(1,i),[train_amt,1]);
end
hdf5write('infogainNNtraining_fc8.h5','/data',tr_prob,'/label',tr_gt);

tr_prob = prob_data(:,1:train_amt,root_s);
tr_prob = tr_prob(:,:);
hdf5write('infogainNNtraining_prob.h5','/data',tr_prob,'/label',tr_gt);

% testing data
te_prob = fc8_data(:,train_amt+1:total,root_s);
te_prob = te_prob(:,:);
for i = 1:174
    te_gt(1,(i-1)*test_amt+1:i*test_amt) = repmat(ig(1,i),[test_amt,1]);
end
hdf5write('infogainNNtesting_fc8.h5','/data',te_prob,'/label',te_gt);

te_prob = prob_data(:,train_amt+1:total,root_s);
te_prob = te_prob(:,:);
hdf5write('infogainNNtesting_prob.h5','/data',te_prob,'/label',te_gt);

% gen Infogain matrix
invalid = -10;
len = max(ig);
H = zeros(len,len);

% info = 1 * len
info = cell(1,len);
for i = 1:len
    if i == 41
        info{i} = [1,2,6,7];
    elseif i == 42
        info{i} = [1,3,4,38,39];
    elseif i == 43
        info{i} = [1,3,4,12,31,35];
    elseif i == 44
        info{i} = [1,2,9];
    elseif i == 45
        info{i} = [1,2,5];
    else
        info{i} = groundtruth{i};
    end
end

for r = 1:len
    for c = 1:len
         tmp = length(intersect(info{r},info{c}))/length(info{r});
         if intersect(info{r},info{c})==1
             H(r,c) = invalid;
         else
             H(r,c) = tmp;
         end
    end
end
H(1,:) = 1;
H(:,1) = 1;
csvwrite('./H_invalid_10.csv',H);

% H: 40 * 40 (nlabel)
H_40 = zeros(nLabel,nLabel);
for r = 1:nLabel
    for c = 1:nLabel
        search_matrix = union(r,c);
        cnt = 0;
        for idx = 1:len
            inter = intersect(info{idx},search_matrix);
            if length(inter)>1
                cnt = cnt+1;
            end
        end
        if cnt >= 1
            H_40(r,c) = 1;
        else
            H_40(r,c) = invalid;
        end
    end
end
for i = 1:40 
    H_40(i,i)=1; 
end
csvwrite('./H_40.csv',H_40);