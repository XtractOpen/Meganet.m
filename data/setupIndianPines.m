function [Y,C,nImg] = setupIndianPines(doPlot)
% [Y,C,nImg] = setupIndianPines()
%
% Y      - tensor (nRows,nCols,nChannels,1)
% C      - corresponding matrix (17 classes, nRows*nCols)
% nImg   - [nRows,nCols,nChannels]


if not(exist('doPlot','var')) || isempty(doPlot)
    doPlot = 0;
end

if not(exist('Indian_pines.mat','file')) || ...
        not(exist('Indian_pines_gt.mat','file'))
        
    warning('Indian pines data cannot be found in MATLAB path')
    
    dataDir = [fileparts(which('Meganet.m')) filesep 'data'];
    ipDir = [dataDir filesep 'IndianPines'];
    if not(exist(ipDir,'dir'))
        mkdir(ipDir);
    end
    
    doDownload = input(sprintf('Do you want to download http://www.ehu.eus/ccwintco/uploads/2/22/Indian_pines.mat (around 6 MB) to %s? Y/N [Y]: ',ipDir),'s');
    if isempty(doDownload)  || strcmp(doDownload,'Y')
        if not(exist(ipDir,'dir'))
            mkdir(ipDir);
        end
        imtz = fullfile(ipDir,'Indian_pines.mat');
        websave(fullfile(ipDir,'Indian_pines.mat'),'http://www.ehu.eus/ccwintco/uploads/2/22/Indian_pines.mat');
        imtz = fullfile(ipDir,'Indian_pines_gt.mat');
        websave(fullfile(ipDir,'Indian_pines_gt.mat'),'http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat');
        addpath(ipDir);
    else
        error('IndianPines data not available. Please make sure it is in the current path');
    end
end

load('Indian_pines.mat');
Y = indian_pines;
nImg = size(Y);

load('Indian_pines_gt.mat')
c = indian_pines_gt(:) + 1;
C = full(sparse(1:numel(c),c,ones(numel(c),1),numel(c),max(c)));

if doPlot
    figure(11); clf;
   subplot(1,2,1);
   imagesc(sum(Y,3));
   
   
   subplot(1,2,2);
   imagesc(reshape(C*(1:max(c))',nImg(1:2)));
end

C = reshape(C,nImg(1),nImg(2),[]);
