% =========================================================================
%
% MNIST Neural Network Example similar to the one from TensorFlow tutorial
%
% For TensorFlow.jl code see: 
%    https://github.com/malmaud/TensorFlow.jl/blob/master/examples/mnist_full.jl
% =========================================================================
clear all; clc;


[Y0,C,Ytest,Ctest] = setupCIFAR10(2^10);
nImg = [32 32];
ncin = 3;

% choose file for results and specify whether or not to retrain
resFile = sprintf('%s.mat',mfilename); 
doTrain = true;

% set GPU flag and precision
useGPU = 0; 
precision='single';

[Y0,C] = gpuVar(useGPU,precision,Y0,C);

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
fctn = dnnBatchObjFctn(net,pRegKb,pLoss,pRegW,Y0,C,'batchSize',256,'useGPU',useGPU,'precision',precision);
fval = dnnBatchObjFctn(net,[],pLoss,[],Ytest,Ctest,'batchSize',256,'useGPU',useGPU,'precision',precision);


if doTrain || not(exist(resFile,'file'))
% initialize weights
theta  = 1e-3*vec(randn(nTheta(net),1));
W      = 1e-3*vec(randn(10,nFeatOut(net)+1));
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
[xOpt,His] = solve(opt,fctn,[theta(:); W(:)],fval);
[thOpt,WOpt] = split(fctn,xOpt);
save(resFile,'thOpt','WOpt','His')
else
load(resFile)
end

