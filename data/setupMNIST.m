function [Ytrain,Ctrain,Ytest,Ctest] = setupMNIST(nTrain,nTest)
% [Ytrain,Ctrain,Ytest,Ctest] = setupMNIST(nTrain,nTest)
%
%
% Output:
%     Ytrain - nTrain 28x28 training images in tensor (28,28,nTrain)
%     Ctrain  - corresponding training classes (10, nTrain)
%     Ytest   - nTest 28x28 test images in tensor (28,28,nTest)
%     Ctest  - corresponding test classes (10, nTest)
%

if nargin==0
    runMinimalExample;
    return;
end

if not(exist('nTrain','var')) || isempty(nTrain)
    nTrain = 50000;
end
if not(exist('nTest','var')) || isempty(nTest)
    nTest = round(nTrain/5);
end

if not(exist('train-images.idx3-ubyte','file')) ||...
        not(exist('train-labels.idx1-ubyte','file')) || ...
        not(exist('t10k-images-idx3-ubyte','file')) || ...
        not(exist('t10k-labels-idx1-ubyte','file'))
    
    warning('MNIST data cannot be found in MATLAB path')
    
    dataDir = [fileparts(which('Meganet.m')) filesep 'data' filesep 'MNIST'];
    if not(exist(dataDir,'dir'))
        mkdir(dataDir);
    end
    
    doDownload = input(sprintf('Do you want to download http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz (around 10 MB) to %s? Y/N [Y]: ',dataDir),'s');
    if isempty(doDownload)  || strcmp(doDownload,'Y')
        if not(exist(dataDir,'dir'))
            mkdir(dataDir);
        end
        imgz = websave(fullfile(dataDir,'train-images.idx3-ubyte.gz'),'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz');
        gunzip(imgz);
        delete(imgz)
        
        imgz = websave(fullfile(dataDir,'train-labels.idx1-ubyte.gz'),'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz');
        gunzip(imgz);
        delete(imgz)
        
        imgz = websave(fullfile(dataDir,'t10k-images-idx3-ubyte.gz'),'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz');
        gunzip(imgz);
        delete(imgz)
        
        imgz = websave(fullfile(dataDir,'t10k-labels-idx1-ubyte.gz'),'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz');
        gunzip(imgz);
        delete(imgz)
        
        addpath(dataDir);
    else
        error('MNNIST data not available. Please make sure it is in the current path');
    end
end

images = reshape ( loadMNISTImages('train-images.idx3-ubyte') , 28,28,1,[] );
labels = loadMNISTLabels('train-labels.idx1-ubyte');

% get class probability matrix
C      = zeros(10,numel(labels));
ind    = sub2ind(size(C),labels+1,(1:numel(labels))');
C(ind) = 1;

idx = randperm(size(C,2));

idTrain = idx(1:nTrain);
Ytrain = images(:,:,:,idTrain);
Ctrain = C(:,idTrain);
if nargout>2
    images = reshape ( loadMNISTImages('t10k-images-idx3-ubyte') , 28,28,1,[] );
    labels = loadMNISTLabels('t10k-labels-idx1-ubyte');
    
    Ctest      = zeros(10,numel(labels));
    ind    = sub2ind(size(Ctest),labels+1,(1:numel(labels))');
    Ctest(ind) = 1;

    idx = randperm(size(Ctest,2));
    idTest = idx(1:nTest);
    Ytest = images(:,:,:,idTest);
    Ctest = Ctest(:,idTest);
end

function runMinimalExample
[Yrain,~,Ytest,~] = feval(mfilename,50,10);
figure(1);clf;
subplot(2,1,1);
montageArray(Yrain,10);
axis equal tight
colormap(flipud(colormap('gray')))
colorbar
title('training images');


subplot(2,1,2);
montageArray(Ytest,10);
axis equal tight
colormap(flipud(colormap('gray')))
colorbar
title('test images');




