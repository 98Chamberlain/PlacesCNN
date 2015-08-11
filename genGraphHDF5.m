load('./scene_name.mat');
% load scene_t 205 * 2 cell

tr_data = hdf5read('./graphCNNtraining.h5','data');
% data: 205 * 17400
% label: 40 * 17400
for i = 1:205
    hdf5write(['./train_h5/',scene_t{i,1},'_train.h5'],'data',tr_data(i,:));
end

te_data = hdf5read('./graphCNNtesting.h5','data');
% data: 205 * 4872
% label: 40 * 4872
for i = 1:205
    hdf5write(['./test_h5/',scene_t{i,1},'_test.h5'],'data',te_data(i,:));
end