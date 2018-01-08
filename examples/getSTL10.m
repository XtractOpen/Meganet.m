function [Y_train,C_train,Y_test,C_test] = getSTL10(n)

%try
%    addpath ../../stl10_matlab/
%catch
%end
load train.mat


nImg = [96,96,3];

ALL_IMG_TRAIN = double(X');
C = double(y);
C = C + ((0:(5000-1))*10)';
TEMP = zeros(10,5000);
TEMP(C) = 1;
ALL_LABELS_TRAIN = TEMP;

ALL_IMG_TRAIN = ALL_IMG_TRAIN - mean(ALL_IMG_TRAIN,2);
s = 1./sqrt(mean(ALL_IMG_TRAIN.^2,2)+1e-2);
ALL_IMG_TRAIN = bsxfun(@times,ALL_IMG_TRAIN,s);

IMG_TEST = ALL_IMG_TRAIN(:,end-499:end);
ALL_IMG_TRAIN = ALL_IMG_TRAIN(:,1:end-500);

LABELS_TEST = ALL_LABELS_TRAIN(:,end-499:end);
ALL_LABELS_TRAIN = ALL_LABELS_TRAIN(:,1:end-500);

p = min(n/size(ALL_IMG_TRAIN,2),1);
n_train = min(n,size(ALL_IMG_TRAIN,2));
n_test = ceil(size(IMG_TEST,2)*p);

ptrain = randperm(size(ALL_IMG_TRAIN,2));
Y_train = ALL_IMG_TRAIN(:,ptrain(1:n_train));
C_train = ALL_LABELS_TRAIN(:,ptrain(1:n_train));

ptest = randperm(size(IMG_TEST,2));
Y_test = IMG_TEST(:,ptest(1:n_test));
C_test = LABELS_TEST(:,ptest(1:n_test));
