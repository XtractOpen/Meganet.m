% =========================================================================
%
% MNIST Neural Network Example similar to the one from TensorFlow tutorial
% but network is much smaller. Optimization is done using iPALM, which is  
% presented in this work:
%
% @article{PockEtAl2017,
%   author = {Pock, Thomas and Sabach, Shoham},
%   title = {{Inertial Proximal Alternating Linearized Minimization (iPALM) for Nonconvex and Nonsmooth Problems}},
%   journal = {arXiv.org},
%   year = {2017},
%   eprint = {1702.02505v1},
%   eprintclass = {math.OC},
%   doi = {10.1137/16M1064064},
% }
%
% For TensorFlow.jl code see:
%    https://github.com/malmaud/TensorFlow.jl/blob/master/examples/mnist_full.jl
% =========================================================================
clear all; clc;


[Y0,C,Ytest,Ctest] = setupMNIST(2^10,2^10);
% nImg = [28 28];
nImg = [size(Y0,1) size(Y0,2)];

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
nc = [4 8];
% setup network
act           = @smoothReluActivationArrayfun;
alphaTh = 1e-3;
blocks        = cell(0,1); pRegThB = cell(0,1);
blocks{end+1} = NN({singleLayer(conv(nImg,[5 5 1 nc(1)]),'activation', act)});
pRegThB{end+1} = tikhonovReg(opEye(nTheta(blocks{end})),alphaTh);

blocks{end+1} = connector(opPoolMCN([nImg nc(1)],2));
blocks{end+1} = NN({singleLayer(conv(nImg/2,[5 5 nc]),'activation', act)});
pRegThB{end+1} = tikhonovReg(opEye(nTheta(blocks{end})),alphaTh);

blocks{end+1} = connector(opPoolMCN([nImg/2 nc(2)],2));
net    = Meganet(blocks);

% setup loss function for training and validation set
pLoss  = softmaxLoss();
pRegTh = blockReg(pRegThB);
fctn = dnnBatchObjFctn(net,[],pLoss,[],Y0,C,'batchSize',256,'useGPU',useGPU,'precision',precision);
fval = dnnBatchObjFctn(net,[],pLoss,[],Ytest,Ctest,'batchSize',256,'useGPU',useGPU,'precision',precision);


if doTrain || not(exist(resFile,'file'))
    % initialize weights
    theta  = 1e-2*vec(randn(nTheta(net),1));
    W      = 1e-2*vec(randn(10,prod(sizeFeatOut(net))+1));
    [theta,W] = gpuVar(fctn.useGPU,fctn.precision,theta,W);
    
    LW = blkdiag(opGrad(nImg/4,10*nc(2),[1;1]),opEye(10));
    alphaW = 1e-2;
    pRegW = tikhonovReg(LW,alphaW,[],'useGPU',useGPU,'precision',precision);
    
    
    
    %% specify blocks and proximal operators
    ids = blkdiag(ones(nTheta(blocks{1}),1),ones(nTheta(blocks{3}),1),ones(numel(W),1));
    Prox = cell(size(ids,2),1);
    P = @(theta) projectKernel([5,5],theta);
    Prox{1} = @(x,tau) proxL2Convex(pRegTh.blocks{1}.B,x,alphaTh,tau,P);
    Prox{2} = @(x,tau)  proxL2Convex(pRegTh.blocks{2}.B,x,alphaTh,tau,P);
    Prox{3} = @(x,tau) proxL2Unconstr(pRegW.B,x,alphaW,tau);
    
    
    
    %% setup optimizat
    opt = iPALM(ids,'Prox',Prox);
    opt.maxEpochs = 40;
    opt.miniBatch = 128;
    opt.Lip = 1e3*[1;1;1];
    opt.out = 1;

    % run optimization
    tic;
    [xOpt,His,Lip] = solve(opt,fctn,[theta(:); W(:)],fval);
    [thOpt,WOpt] = split(fctn,gather(xOpt));
    toc
    save(resFile,'thOpt','WOpt','His','Lip')
else
    load(resFile)
end
return;
%%
ths = split(net,thOpt);
figure(1); clf;
subplot(1,2,1);
montageArray(reshape(ths{1},5,5,[]))
axis equal off

subplot(1,2,2);
montageArray(reshape(ths{3},5,5,[]))
axis equal off
