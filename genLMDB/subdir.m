function [directory , numsubdir] = subdir(PATH)
    
    directory = dir( PATH );
    directory(~[directory.isdir]) = [];  %remove non-directories
    tf = ismember( {directory.name}, {'.', '..'});
    directory(tf) = [];  %remove current and parent director
    numsubdir = length(directory);
    
end