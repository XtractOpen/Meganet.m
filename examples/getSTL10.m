function [Y_train,C_train,Y_test,C_test] = getSTL10(n)

% use setupSTL instead of this

addpath('data/stl10_matlab')

load train.mat

if nargin==0
    n=4000; % n cannot be more than 5000
end

nImg = [96,96,3];



ALL_IMG_TRAIN = double(X');
ALL_IMG_TRAIN = reshape(ALL_IMG_TRAIN, 96,96,3,[]); % tensor format
colons = repmat( {':'} , 1 , ndims(ALL_IMG_TRAIN) -1 );

C = double(y);
C = C + ((0:(5000-1))*10)';
TEMP = zeros(10,5000);
TEMP(C) = 1;
ALL_LABELS_TRAIN = TEMP;

ALL_IMG_TRAIN = ALL_IMG_TRAIN - mean(ALL_IMG_TRAIN, ndims(ALL_IMG_TRAIN) );
s = 1./sqrt(mean(ALL_IMG_TRAIN.^2, ndims(ALL_IMG_TRAIN) ) + 1e-2);
% ALL_IMG_TRAIN = bsxfun(@times,ALL_IMG_TRAIN,s);
ALL_IMG_TRAIN = ALL_IMG_TRAIN.*s;


IMG_TEST = ALL_IMG_TRAIN(colons{:},5000-499:5000);
ALL_IMG_TRAIN = ALL_IMG_TRAIN(colons{:},1:5000-500);

LABELS_TEST = ALL_LABELS_TRAIN(:,5000-499:5000);
ALL_LABELS_TRAIN = ALL_LABELS_TRAIN(:,1:5000-500);

n_total = size(ALL_IMG_TRAIN,ndims(ALL_IMG_TRAIN));
p = min( n/n_total , 1);
n_train = min(n,n_total);
n_test = ceil(size(IMG_TEST, ndims(IMG_TEST))*p);

ptrain = randperm( n_total );
Y_train = ALL_IMG_TRAIN(colons{:},ptrain(1:n_train));
C_train = ALL_LABELS_TRAIN(:,ptrain(1:n_train));

ptest = randperm( size(IMG_TEST,ndims(IMG_TEST)) );
Y_test = IMG_TEST(colons{:},ptest(1:n_test));
C_test = LABELS_TEST(:,ptest(1:n_test));
