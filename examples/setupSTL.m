function[Ytrain,Ctrain,Yval,Cval ] = setupSTL(nTrain,nVal)

% baseDir = fileparts(which('startupMeganet.m'));
% stlDir = fullfile(baseDir,'..','stl10_matlab');
% addpath(stlDir);

load train.mat
Ytrain = double(X);
if nTrain<size(Ytrain,1)
    ptrain = randi(size(Ytrain,1),nTrain,1);
else
    ptrain = 1:size(Ytrain,1);
end
Ytrain = Ytrain(ptrain,:);
nex = size(Ytrain,1);
Ytrain = reshape(normalizeData(reshape(Ytrain',96*96,[])')',[],nex);
Ctrain = zeros(nex,10);
ind    = sub2ind(size(Ctrain),(1:size(Ctrain,1))',y(ptrain));
Ctrain(ind) = 1;

load  test.mat
Yval = double(X);
if nVal<size(Yval,1)
    pval = randi(size(Yval,1),nVal,1);
else
    pval = 1:size(Yval,1);
end
Yval = Yval(pval,:);
nv = size(Yval,1);
Yval = reshape(normalizeData(reshape(Yval',96*96,[])')',[],nv);
Cval = zeros(nv,10);
ind    = sub2ind(size(Cval),(1:size(Cval,1))',y(pval));
Cval(ind) = 1;

Ctrain = Ctrain';
Cval   = Cval';