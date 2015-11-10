function genLMDB_multi_label( file_path , output_path , gt_label , amt )
% gt_label: scene x label ( 205 x 40 )

load('../gt_scene.mat');
load('../total_label.mat');

writeID = fopen(output_path,'w');
readID = fopen(file_path,'r');

% data_total = 2000;
% prop = 0.8;
% train_amt = data_total * prop * prop;
% val_amt = data_total * prop * (1-prop);
% test_amt = data_total * (1-prop);

root_s = [1,2,3,5:1:22,24,25,26,28:1:38,40:1:49,51:1:63,65,66,...
    67,69:1:75,77:1:81,83:1:94,96:1:99,101:1:119,121:1:138,140,142,143,145,...
    147,148,150:1:158,160,162,163,164,166:1:172,174:1:185,189,190,194,195,196,...
    198,199,201,202,205];

cnt = 1; % count the number of total test image
tline = fgetl(readID);
while ischar(tline)
    
    file_content = textscan(tline, '%s %d');
    file_path = file_content{1}{1};
    
    scene_idx = root_s( floor((cnt-1)./amt)+1 );
    
    fprintf(writeID,'%s %d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n',file_path,gt_label(scene_idx,:));
    
    % cont. to read another line
    tline = fgetl(readID);
    cnt = cnt + 1;
    
end


end
