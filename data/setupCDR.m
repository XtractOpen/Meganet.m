function[Yt,Ct,Yv,Cv,Ytest,Ctest,idx] = setupCDR(nTrain,nVal,seed)
% Convection Diffusion Reaction:
%
%   du/dt = \nabla \cdot (D \nabla u) - v \cdot \nabla u + f + y' * r(u)
%   D \nabla u \cdot n = 0          (Neumann boundary conditions)
%   u = 0                           (initial condition)
% 
%   u - state variable (e.g., concentration)
%   D - diffusion coefficient
%   v - velocity field
%   f - source
%   r - reaction term
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
% Loads the data
%   Y: parameters, 55 x 800 
%   C: targets, 72 x 800


if not(exist('CDR_Data.mat','file')) 
    
    warning('CDR_Data cannot be found in MATLAB path')
    
    dataURL = 'http://www.mathcs.emory.edu/~lruthot/pubs/2020-GNvpro/CDR_Data.mat';
    dataDir = [fileparts(which('Meganet.m')) filesep 'data' filesep 'CDR'];
    
    doDownload = input(sprintf('Do you want to download %s (around 634 KB) to %s? Y/N [Y]: ',dataURL,dataDir),'s');
    
    if isempty(doDownload)  || strcmp(doDownload,'Y')
        
        if not(exist(dataDir,'dir'))
            mkdir(dataDir);
        end
        imtz = fullfile(dataDir,'CDR_Data.mat');
        
        % need to change name!
        websave(imtz,dataURL);
        addpath(dataDir);
    else
        error('CDR_Data data not available. Please make sure it is in the current path');
    end
end


% load data from here for the moment (Y and C)
load('CDR_Data.mat');

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

Yt = Y(:,idxTrain);
Ct = C(:,idxTrain);

Yv = Y(:,idxVal);
Cv = C(:,idxVal);

Ytest = Y(:,idxTest);
Ctest = C(:,idxTest);




end