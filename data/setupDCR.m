
function[Yt,Ct,Yv,Cv,Ytest,Ctest,idx] = setupDCR(nTrain,nVal,seed)
% Direct Current Resistivity:
%
%   -\nabla \cdot (m(x;y) \nabla u) = q     (Poisson's Equation)
%    \nabla u \cdot n = 0                   (Neumann boundary conditions)
%
%   u - electric potential
%   m - model (e.g., conductivity)
%   q - source
%   y - parameters
%
% For details, see the paper:
%
% @article{newman2020train,
% 	title={Train Like a (Var)Pro: Efficient Training of Neural Networks with Variable Projection},
% 	author={Elizabeth Newman and Lars Ruthotto and Joseph Hart and Bart van Bloemen Waanders},
% 	year={2020},
% 	journal={arXiv preprint arxiv.org/abs/2007.13171},
% }
% 
% Loads the following data:
%   Y: parameters, 3 x 10000 
%   C: targets, 882 x 10000


if not(exist('DCR_Data.mat','file')) 
    
    warning('DCR_Data cannot be found in MATLAB path')
    
    dataURL = 'http://www.mathcs.emory.edu/~lruthot/pubs/2020-GNvpro/DCR_Data.mat';
    dataDir = [fileparts(which('Meganet.m')) filesep 'data' filesep 'DCR'];
    
    doDownload = input(sprintf('Do you want to download %s (around 64 MB) to %s? Y/N [Y]: ',dataURL,dataDir),'s');
    
    if isempty(doDownload)  || strcmp(doDownload,'Y')
        
        if not(exist(dataDir,'dir'))
            mkdir(dataDir);
        end
        imtz = fullfile(dataDir,'DCR_Data.mat');
        
        % need to change name!
        websave(imtz,dataURL);
        addpath(dataDir);
    else
        error('DCR_Data data not available. Please make sure it is in the current path');
    end
end

% load data from here for the moment (P1 and D1)
load('DCR_Data.mat');

% split data
nSamples = size(Y,2);

if ~exist('nTrain','var'), nTrain = 400; end
if ~exist('nVal','var'), nVal = 200; end

if (nTrain + nVal) > nSamples
    warning('Requested too much data - choosing 400 training and 200 validation');
end

% for reproducibility
if exist('seed','var'), rng(seed); end


% split data into training, validation, and test
idx = randperm(nSamples);
idxTrain = idx(1:nTrain);
idxVal = idx(nTrain+1:nTrain+nVal);
idxTest = idx(nTrain+nVal+1:end);

meanTrain = mean(C(:,idxTrain),2);

Yt = Y(:,idxTrain);
Ct = C(:,idxTrain) - meanTrain;

Yv = Y(:,idxVal);
Cv = C(:,idxVal) - meanTrain;

Ytest = Y(:,idxTest);
Ctest = C(:,idxTest) - meanTrain;


end
