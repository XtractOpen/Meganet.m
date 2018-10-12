function[Ytrain,Ctrain,Yval,Cval] = setupCIFAR10(nTrain,nVal,option)
% [Ytrain,Ctrain,Yval,Cval] = setupCIFAR10(nTrain,nVal,option)
%
% Ytrain - tensor (nRows,nCols,nChannels,nTrain)      
% Ctrain - corresponding matrix (10 classes, nTrain)
% Yval   - tensor (nRows,nCols,nChannels,nVal)
% Cval   - corresponding matrix (10 classes, nVal)
%
% images are 32x32 RGB, so nRows=32 , nCols=32 , nChannels=3
%

addpath('data/cifar-10-batches-mat');

if nargin==0
    runMinimalExample;
    return;
end

if not(exist('nTrain','var')) || isempty(nTrain)
    nTrain = 50000;
end

if not(exist('nVal','var')) || isempty(nVal)
    nVal = ceil(nTrain/5);
end

if not(exist('option','var')) || isempty(option)
    option = 1;
end

% Reading in the data
load data_batch_1.mat
data1   = double(data);
labels1 = labels;

load data_batch_2.mat
data2   = double(data);
labels2 = labels;

load data_batch_3.mat
data3   = double(data);
labels3 = labels;

load data_batch_4.mat
data4   = double(data);
labels4 = labels;

load data_batch_5.mat
data5   = double(data);
labels5 = labels;

data   = [data1; data2; data3; data4; data5];
labels = [labels1; labels2; labels3; labels4; labels5];
nex = size(data,1);

if nTrain<nex
    ptrain = randperm(nex,nTrain);
else
    ptrain = 1:nex;
end

[Ytrain,Ctrain] = sortAndScaleData(data(ptrain,:),labels(ptrain),option);
Ytrain = Ytrain';
Ctrain = Ctrain';

% make into tensor, (nRows,nCols,nChannels,nEx) 
Ytrain = reshape(Ytrain,32,32,3,[]); 
% rotate by 90 degrees
Ytrain = permute(Ytrain,[2 1 3 4]);

% Ytrain = reshape(Ytrain,32*32*3,[]);


if nargout>2
    load test_batch.mat
    dataTest   = double(data);
    labelsTest = labels;
    nex = size(dataTest,1);
    if nVal<nex
        pval = randperm(nex,nVal);
    else
        pval = 1:nex;
    end
    [Yval,Cval] = sortAndScaleData(dataTest(pval,:),labelsTest(pval),option);
    Yval = Yval';
    Cval = Cval';
    
    % rotate by 90 degrees
    Yval = reshape(Yval,32,32,3,[]);
    Yval = permute(Yval,[2 1 3 4]);
end




function runMinimalExample
[Yt,Ct,Yv,Cv] = feval(mfilename,50,10,4);
figure(2);clf;
subplot(2,1,1);
montageArray(Yt(:,:,1,:),10);
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

