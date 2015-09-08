H5FILE = './h5file_t/';

d = dir([H5FILE,'*.h5']);
l = length(d);

test = zeros(40,l);

for i = 1:l
    test(:,i) = h5read([H5FILE,d(i).name],'/data');
end
    