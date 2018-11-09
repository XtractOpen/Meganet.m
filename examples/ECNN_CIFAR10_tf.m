% =========================================================================
%
% CIFAR10 Neural Network Example similar to the one from TensorFlow tutorial
%
% For TensorFlow.jl code see: 
%    https://github.com/malmaud/TensorFlow.jl/blob/master/examples/mnist_full.jl
% =========================================================================

clear all; clc;

nImg = [32 32];
ncin = 3;

nTrain = 2^10;
nVal   = 2^5;
[Y,C] = setupCIFAR10(nTrain+nVal);
Y    = normalizeData(Y,prod(nImg)*ncin);

% divide into training and validation data
id = randperm(size(C,2));
idt = id(1:nTrain);
idv = id(nTrain+1:end);
Yt  = Y(:,:,:,idt); Ct = C(:,idt);
Yv  = Y(:,:,:,idv); Cv = C(:,idv);


% choose file for results and specify whether or not to retrain
resFile = sprintf('%s.mat',mfilename); 
doTrain = true;

% set GPU flag and precision
useGPU = 0; 
precision='single';

[Yt,Ct,Yv,Cv] = gpuVar(useGPU,precision,Yt,Ct,Yv,Cv);

%% choose convolution
if useGPU
    cudnnSession = convCuDNN2DSession();
    conv = @(varargin)convCuDNN2D(cudnnSession,varargin{:});
else
    conv    = @convMCN;
end

% setup network
act = @reluActivation;
blocks    = cell(0,1);
blocks{end+1} = NN({singleLayer(conv(nImg,[5 5 ncin 32]),'activation', act)});

blocks{end+1} = connector(opPoolMCN([nImg 32],2));
blocks{end+1} = NN({singleLayer(conv(nImg/2,[5 5 32 64]),'activation', act)});

blocks{end+1} = connector(opPoolMCN([nImg/2 64],2));
net    = Meganet(blocks);

% setup loss function for training and validation set
pLoss  = softmaxLoss();
pRegW  = [];
pRegKb = [];
fctn = dnnBatchObjFctn(net,pRegKb,pLoss,pRegW,Yt,Ct,'batchSize',256,'useGPU',useGPU,'precision',precision);
fval = dnnBatchObjFctn(net,[],pLoss,[],Yv,Cv,'batchSize',256,'useGPU',useGPU,'precision',precision);


if doTrain || not(exist(resFile,'file'))
% initialize weights
theta  = 1e-3*vec(randn(nTheta(net),1));
W      = 1e-3*vec(randn(10,prod(sizeFeatOut(net))+1));
[theta,W] = gpuVar(fctn.useGPU,fctn.precision,theta,W);

% setup optimization
opt = sgd();
opt.learningRate = @(epoch) 1e-3/sqrt(epoch);
opt.maxEpochs = 20;
opt.nesterov = false;
opt.ADAM=true;
opt.miniBatch = 128;
opt.momentum = 0.9;
opt.out = 1;

% run optimization
tic;
[xOpt,His] = solve(opt,fctn,[theta(:); W(:)],fval);
[thOpt,WOpt] = split(fctn,xOpt);
toc
save(resFile,'thOpt','WOpt','His')
else
load(resFile)
end

