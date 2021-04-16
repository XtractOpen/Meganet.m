% =========================================================================
%
% CIFAR10 Neural Network Example similar to the one from TensorFlow tutorial
%
% For TensorFlow.jl code see:
%    https://github.com/malmaud/TensorFlow.jl/blob/master/examples/mnist_full.jl
% =========================================================================

clear; clc;

myDir   = pwd;
resFile = sprintf([myDir,'/results/%s'],mfilename);
disp([myDir,'/examples/',mfilename,'.m']);
copyfile([myDir,'/examples/',mfilename,'.m'],[resFile,'.m']);  % save copy of script
diary([resFile,'.txt']);              % save diary of printouts                                                                                                                                                                                                                                                             

%% 
% for reproducibility
dataSeed = 2;
nImg = [32 32];
ncin = 3;

rng(dataSeed);
nTrain = 4e4;
nVal   = 1e4;
nTest  = 1e4;
[Y,C,Ytest,Ctest] = setupCIFAR10(nTrain+nVal,nTest);
Y     = normalizeData(Y,prod(nImg)*ncin);
Ytest = normalizeData(Ytest,prod(nImg)*ncin);

% divide into training and validation data
id = randperm(size(C,2));
disp(id(1:10))
idt = id(1:nTrain);
idv = id(nTrain+1:end);
Yt  = Y(:,:,:,idt); Ct = C(:,idt);
Yv  = Y(:,:,:,idv); Cv = C(:,idv);

% set GPU flag and precision
useGPU    = 1;
precision = 'single';

[Yt,Ct,Yv,Cv] = gpuVar(useGPU,precision,Yt,Ct,Yv,Cv);

%% build network
convOp = getConvOp(useGPU);
poolOp = getPoolOp();

% setup network
act = @reluActivation;
blocks    = cell(0,1);
blocks{end+1} = NN({singleLayer(convOp(nImg,[5 5 ncin 32]),'activation', act)});

blocks{end+1} = connector(poolOp([nImg 32],2));
blocks{end+1} = NN({singleLayer(convOp(nImg/2,[5 5 32 64]),'activation', act)});

blocks{end+1} = connector(poolOp([nImg/2 64],nImg(1)/2));
net    = Meganet(blocks);

% setup loss function for training and validation set
pLoss  = softmaxLoss();

% regularization
alphaTh = logspace(-5,-3,7);
alphaW  = logspace(-3,0,10);

pRegW  = tikhonovReg(opEye(10*(numelFeatOut(net)+1)),1e-2,[]);
pRegKb = tikhonovReg(opEye(nTheta(net)),1e-5,[],'useGPU',useGPU,'precision',precision);

% for reproducibility
weightSeed = 42;
rng(weightSeed);

% initialize weights
[theta,W] = networkInitialization(net,'kaiming_uniform',[],10);

theta = gpuVar(useGPU,precision,theta);

classSolver         = trnewton('atol',1e-10,'rtol',1e-10,'maxIter',400,'out',1);
classSolver.linSol  = GMRES('m',100,'tol',1e-4,'out',0);
fctnInit            = dnnVarProObjFctn(net,pRegKb,pLoss,pRegW,classSolver,Yt,Ct,'useGPU',useGPU,'precision',precision);
fctnInit.pRegW.precision = 'double';


% solve for varpro W
[~,para] = eval(fctnInit,theta);
WVarPro = para.W;
  
net.useGPU = false;
theta = gather(theta);
WVarPro = gather(WVarPro);
classSolver.out = 0;

dummy = 'I am a fake variable';
save(resFile, 'net','theta','W','WVarPro','pRegW','pRegKb','pLoss','dataSeed','nTrain','nVal','nTest','classSolver','dummy')
diary off;

