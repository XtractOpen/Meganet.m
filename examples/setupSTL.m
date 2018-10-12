function[Ytrain,Ctrain,Yval,Cval ] = setupSTL(nTrain,nVal)
% [Ytrain,Ctrain,Yval,Cval ] = setupSTL(nTrain,nVal)
%
% Ytrain - tensor (nRows,nCols,nChannels,nTrain)      
% Ctrain - corresponding matrix (10 classes, nTrain)
% Yval   - tensor (nRows,nCols,nChannels,nVal)
% Cval   - corresponding matrix (10 classes, nVal)
%
% images are 96x96 RGB, so nRows=96 , nCols=96 , nChannels=3
%


addpath('data/stl10_matlab')

if nargin==0
    runMinimalExample;
    return;
end


if not(exist('nVal','var')) || isempty(nVal)
    nVal = ceil(nTrain/5);
end

load train.mat
nex = numel(y);
Ctrain = zeros(10,nex);
ind    = sub2ind(size(Ctrain),y,(1:nex)');
Ctrain(ind) = 1;
if nTrain<nex
    % get random permutation that ensures equal number of examples per class
    idx = [];
    for k=1:size(Ctrain,1)
        ik = find(Ctrain(k,:));
        ip = randperm(numel(ik));
        idx = [idx; reshape(ik(ip),1,[])];
    end
    idx = idx(:);
    ptrain = idx(1:nTrain);
else
    ptrain = 1:nex;
end
Ctrain = Ctrain(:,ptrain);
Ytrain = double(X);
Ytrain = Ytrain(ptrain,:);
Ytrain = reshape(normalizeData(reshape(Ytrain',96*96,[])')',96,96,3,numel(ptrain));


if nargout>2
    load  test.mat
    nv = numel(y);
    Cval = zeros(10,nv);
    ind    = sub2ind(size(Cval),y,(1:nv)');
    Cval(ind) = 1;
    if nVal<nv
        idx = [];
        for k=1:10
            ik = find(Cval(k,:));
            ip = randperm(numel(ik));
            idx = [idx; reshape(ik(ip),1,[])];
        end
        idx = idx(:);
        pval = idx(1:nVal);
    else
        pval = 1:nv;
    end
    Cval = Cval(:,pval);
    Yval = double(X);
    Yval = Yval(pval,:);
    Yval = reshape(normalizeData(reshape(Yval',96*96,[])')',96,96,3,numel(pval));
end

function runMinimalExample
[Yt,Ct,Yv,Cv] = feval(mfilename,50,40);
figure(1);clf;
subplot(2,1,1);
montageArray(Yt(:,:,1,:),10);
% montage(Yt(:,:,:,1:10)); RGB handling
axis equal tight
colormap gray
colorbar
title('training images');


subplot(2,1,2);
montageArray(Yv(:,:,1,:),10);
axis equal tight
colormap gray
colorbar
title('validation images');
