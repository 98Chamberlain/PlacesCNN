% root
root_s = [1,2,3,5:1:22,24,25,26,28:1:38,40:1:49,51:1:63,65,66,...
    67,69:1:75,77:1:81,83:1:94,96:1:99,101:1:119,121:1:138,140,142,143,145,...
    147,148,150:1:158,160,162,163,164,166:1:172,174:1:185,189,190,194,195,196,...
    198,199,201,202,205];

% read data
path = '/media/ponu/DATA/Places205_resize/images256'

tr_label = hdf5read([path,'/train_lmdb_single_label.h5'],'label');
va_label = hdf5read([path,'/val_lmdb_single_label.h5'],'label');
te_label = hdf5read([path,'/test_lmdb_single_label.h5'],'label');

% parameter
nScene = 205;
nLabel = 40;

% load groundtruth
load('./gt_scene.mat');
% groundtruth 1*40 cell
% gt_scene 1*205 double

% gt = 205 * 40
gt = ones(nScene,nLabel);
gt = gt*(-1);
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

% multi_gt = 40 * 40
multi_gt = ones(nLabel,nLabel);
multi_gt = multi_gt*(-1);
for i = 1:nLabel
	multi_gt(i,groundtruth{i}) = 1;
end

train_amt = length(tr_label);
val_amt = length(va_label);
test_amt = length(te_label);

% generate the multi label data ( nLabel * nImage )
tr_gt = zeros( nLabel , train_amt );
for id = 1:train_amt
	tr_gt(:,id) = multi_gt(tr_label(id),:)';
end
hdf5write([path,'/train_lmdb_multi_label.h5'],'/label',tr_gt);

va_gt = zeros( nLabel , val_amt );
for id = 1:val_amt
	va_gt(:,id) = multi_gt(va_label(id),:)';
end
hdf5write([path,'/val_lmdb_multi_label.h5'],'/label',va_gt);

te_gt = zeros( nLabel , test_amt );
for id = 1:test_amt
	te_gt(:,id) = multi_gt(te_label(id),:)';
end
hdf5write([path,'/test_lmdb_multi_label.h5'],'/label',te_gt);
