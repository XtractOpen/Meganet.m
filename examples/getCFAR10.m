function [Y_train,C_train,Y_test,C_test] = getCFAR10(n)
nImg = [32,32,3];
ALL_IMG_TRAIN = zeros(prod(nImg),50000,'single');
ALL_LABELS_TRAIN = zeros(10,50000,'single');
TEMP = zeros(10,10000,'single');
for i=1:5
    load(['data_batch_',num2str(i),'.mat']);
    ALL_IMG_TRAIN(:,((i-1)*10000+1):(i*10000)) = data';
    C = double(labels+1);
    C = C + ((0:(10000-1))*10)';
    TEMP(C) = 1;
    ALL_LABELS_TRAIN(:,((i-1)*10000+1):(i*10000)) = TEMP;
    TEMP(:) = 0;
end
ALL_IMG_TRAIN = ALL_IMG_TRAIN ./ max(abs(ALL_IMG_TRAIN(:)));
% figure; imshow(reshape(ALL_IMG_TRAIN(:,75),32,32,3))

IMG_TEST = zeros(prod(nImg),10000,'single');
load('test_batch.mat');
IMG_TEST(:,:) = data'; 
IMG_TEST = IMG_TEST ./ max(abs(IMG_TEST(:)));

IMG_TEST = normalizeData(IMG_TEST')';

ALL_IMG_TRAIN = normalizeData(ALL_IMG_TRAIN')';

% figure; imshow(reshape(IMG_TEST(:,75),32,32,3))
LABELS_TEST = zeros(10,10000,'single');
LABELS_TEST(double(labels') + 1 + (0:(10000-1))*10) = 1;

p = n/size(ALL_IMG_TRAIN,2);
n_train = n;
n_test = ceil(size(IMG_TEST,2)*p);

ptrain = randperm(size(ALL_IMG_TRAIN,2));
Y_train = ALL_IMG_TRAIN(:,ptrain(1:n_train));
C_train = ALL_LABELS_TRAIN(:,ptrain(1:n_train));

ptest = randperm(size(IMG_TEST,2));
Y_test = IMG_TEST(:,ptest(1:n_test));
C_test = LABELS_TEST(:,ptest(1:n_test));
end