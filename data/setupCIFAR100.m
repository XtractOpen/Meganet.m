function[Ytrain,Ctrain,Ytest,Ctest] = setupCIFAR100(nTrain,nTest)
% [Ytrain,Ctrain,Ytest,Ctest] = setupCIFAR100(nTrain,nTest,option)
%
% Ytrain - tensor (nRows,nCols,nChannels,nTrain)
% Ctrain - corresponding matrix (100 classes, nTrain)
% Ytest   - tensor (nRows,nCols,nChannels,nTest)
% Ctest   - corresponding matrix (100 classes, nTest)
%
% images are 32x32 RGB, so nRows=32 , nCols=32 , nChannels=3
%

if nargin==0
    runMinimalExample;
    return;
end

if not(exist('nTrain','var')) || isempty(nTrain)
    nTrain = 50000;
end

if not(exist('nTest','var')) || isempty(nTest)
    nTest = ceil(nTrain/5);
end

if not(exist('cifar100-train.mat','file')) || ...
        not(exist('cifar100-test.mat','file')) || ...
        not(exist('cifar100-meta.mat','file'))
     
    warning('CIFAR100 data cannot be found in MATLAB path')
    
    dataDir = [fileparts(which('Meganet.m')) filesep 'data'];
    cifarDir = [dataDir filesep 'CIFAR'];
    if not(exist(cifarDir,'dir'))
        mkdir(cifarDir);
    end
    
    doDownload = input(sprintf('Do you want to download https://www.cs.toronto.edu/~kriz/cifar-100-matlab.tar.gz (around 175 MB) to %s? Y/N [Y]: ',dataDir),'s');
    if isempty(doDownload)  || strcmp(doDownload,'Y')
        if not(exist(dataDir,'dir'))
            mkdir(dataDir);
        end
        imtz = fullfile(dataDir,'cifar-100-matlab.tar.gz');
        if not(exist(imtz,'file')) 
             websave(fullfile(dataDir,'cifar-100-matlab.tar.gz'),'https://www.cs.toronto.edu/~kriz/cifar-100-matlab.tar.gz');
         end
        im  = untar(imtz,dataDir);
        movefile([dataDir filesep 'cifar-100-matlab' filesep 'train.mat'],fullfile(cifarDir,'cifar100-train.mat'));
        movefile([dataDir filesep 'cifar-100-matlab' filesep 'test.mat'],fullfile(cifarDir,'cifar100-test.mat'));
        movefile([dataDir filesep 'cifar-100-matlab' filesep 'meta.mat'],fullfile(cifarDir,'cifar100-meta.mat'));
        delete(imtz)
        rmdir(fullfile([dataDir filesep 'cifar-100-matlab']))
        addpath(cifarDir);
    else
        error('CIFAR100 data not available. Please make sure it is in the current path');
    end
end


% Reading in the data
load cifar100-train.mat
data   = double(data);
labels = fine_labels;
nex    = size(data,1);

if nTrain<nex
    ptrain = randperm(nex,nTrain);
else
    ptrain = 1:nex;
end

% get class probability matrix
labels      = labels(ptrain);
Ctrain      = zeros(100,numel(labels));
ind         = sub2ind(size(Ctrain),labels+1,(1:numel(labels))');
Ctrain(ind) = 1;

% reshape into tensor, (nRows,nCols,nChannels,nEx)
Ytrain = reshape(data(ptrain,:)',32,32,3,[]);
% rotate by 90 degrees
Ytrain = permute(Ytrain,[2 1 3 4]);

if nargout>2
    load cifar100-test.mat
    dataTest   = double(data);
    labels = fine_labels;
    nex = size(dataTest,1);
    if nTest<nex
        ptest = randperm(nex,nTest);
    else
        ptest = 1:nex;
    end
    
    labels      = labels(ptest);
    Ctest      = zeros(100,numel(labels));
    ind         = sub2ind(size(Ctest),labels+1,(1:numel(labels))');
    Ctest(ind) = 1;
    
    
    Ytest = reshape(dataTest(ptest,:)',32,32,3,[]);
    % rotate by 90 degrees
    Ytest = permute(Ytest,[2 1 3 4]);
end




function runMinimalExample
[Ytrain,~,Ytest,~] = feval(mfilename,50,10);
figure(2);clf;
subplot(2,1,1);
montageArray(Ytrain(:,:,1,:),10);
axis equal tight
colormap gray
colorbar
title('training images');

subplot(2,1,2);
montageArray(Ytest(:,:,1,:),10);
axis equal tight
colormap gray
colorbar
title('test images');


