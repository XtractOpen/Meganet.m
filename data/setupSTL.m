function[Ytrain,Ctrain,Ytest,Ctest ] = setupSTL(nTrain,nTest)
% [Ytrain,Ctrain,Ytest,Ctest ] = setupSTL(nTrain,nTest)
%
% Ytrain - tensor (nRows,nCols,nChannels,nTrain)      
% Ctrain - corresponding matrix (10 classes, nTrain)
% Ytest  - tensor (nRows,nCols,nChannels,nVal)
% Ctest  - corresponding matrix (10 classes, nVal)
%
% images are 96x96 RGB, so nRows=96 , nCols=96 , nChannels=3
%

if nargin==0
    runMinimalExample;
    return;
end


if not(exist('nTest','var')) || isempty(nTest)
    nTest = ceil(nTrain/5);
end

if not(exist('stl10-train.mat','file')) || ...
        not(exist('stl10-test.mat','file')) 
        
    warning('STL10 data cannot be found in MATLAB path')
    
    dataDir = [fileparts(which('Meganet.m')) filesep 'data'];
    stlDir = [dataDir filesep 'STL10'];
    if not(exist(stlDir,'dir'))
        mkdir(stlDir);
    end
    
    doDownload = input(sprintf('Do you want to download http://ai.stanford.edu/~acoates/stl10/stl10_matlab.tar.gz (around 2.85 GB) to %s? Y/N [Y]: ',dataDir),'s');
    if isempty(doDownload)  || strcmp(doDownload,'Y')
        if not(exist(dataDir,'dir'))
            mkdir(dataDir);
        end
        imtz = fullfile(dataDir,'stl10_matlab.tar.gz');
        if not(exist(imtz,'file')) 
             websave(fullfile(dataDir,'stl10_matlab.tar.gz'),'http://ai.stanford.edu/~acoates/stl10/stl10_matlab.tar.gz');
         end
        im  = untar(imtz,dataDir);
        movefile([dataDir filesep 'stl10_matlab' filesep 'train.mat'],fullfile(stlDir,'stl10-train.mat'));
        movefile([dataDir filesep 'stl10_matlab' filesep 'test.mat'],fullfile(stlDir,'stl10-test.mat'));
        movefile([dataDir filesep 'stl10_matlab' filesep 'unlabeled.mat'],fullfile(stlDir,'stl10-unlabeled.mat'));
        delete(imtz)
        rmdir(fullfile([dataDir filesep 'stl10_matlab']))
        addpath(stlDir);
    else
        error('STL10 data not available. Please make sure it is in the current path');
    end
end

load stl10-train.mat
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
Ytrain = reshape(Ytrain',96,96,3,[]);

if nargout>2
    load  stl10-test.mat
    nv = numel(y);
    Ctest = zeros(10,nv);
    ind    = sub2ind(size(Ctest),y,(1:nv)');
    Ctest(ind) = 1;
    if nTest<nv
        idx = [];
        for k=1:10
            ik = find(Ctest(k,:));
            ip = randperm(numel(ik));
            idx = [idx; reshape(ik(ip),1,[])];
        end
        idx = idx(:);
        ptest = idx(1:nTest);
    else
        ptest = 1:nv;
    end
    Ctest = Ctest(:,ptest);
    Ytest = double(X);
    Ytest = Ytest(ptest,:);
    Ytest = reshape(Ytest',96,96,3,[]);

end

function runMinimalExample
[Ytrain,~,Ytest,~] = feval(mfilename,50,40);
figure(1);clf;
subplot(2,1,1);
montageArray(Ytrain(:,:,:,:),10);
axis equal tight
colormap gray
colorbar
title('training images');


subplot(2,1,2);
montageArray(Ytest(:,:,:,:),10);
axis equal tight
colormap gray
colorbar
title('test images');
