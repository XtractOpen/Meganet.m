% =========================================================================
%
% CIFAR10 Neural Network Example similar to the one from TensorFlow tutorial
%
% For TensorFlow.jl code see: 
%    https://github.com/malmaud/TensorFlow.jl/blob/master/examples/mnist_full.jl
% =========================================================================

clear; clc;

% for turing
myDir = '/local/home/enewma5/MeganetGNvpro';
addpath(genpath(myDir));
numEpochs  = 200;
batchSize  = 128;
alphaW     = 1e-2;
alphaTh    = 1e-4;
lr         = 5e-3;

extraInfo = sprintf('--numEpochs-%d--batchSize-%d--alphaW-%0.2e--alphaTh-%0.2e--lr-%0.2e--Nesterov',numEpochs,batchSize,alphaW,alphaTh,lr);

% myDir   = pwd;

resFile = sprintf([myDir,'/results/%s-%s'],mfilename,date);
resFile = [resFile,extraInfo];      
disp(resFile)
copyfile([myDir,'/examples/',mfilename,'.m'],[resFile,'.m']);  % save copy of script                                                                                                                                                                                                                                          
diary([resFile,'.txt']);              % save diary of printouts   

% set GPU flag and precision
useGPU    = 1; 
precision = 'single';

% load pre-initialized weights and network
preInit = load('results/ECNN_CIFAR10_networkInit.mat');
net    = preInit.net;
pRegW  = preInit.pRegW;
pRegW.alpha = alphaW;

pRegKb = preInit.pRegKb;
pRegKb.alpha = alphaTh;

pLoss  = preInit.pLoss;
classSolver = preInit.classSolver;

net.useGPU    = useGPU;  net.precision    = precision;
pRegW.useGPU  = useGPU;  pRegW.precision  = precision; 
pRegKb.useGPU = useGPU;  pRegKb.precision = precision; 

theta  = preInit.theta;
% W      = preInit.W;
W      = preInit.WVarPro;
[theta,W] = gpuVar(useGPU,precision,theta,W);

dataSeed = preInit.dataSeed;
nTrain   = preInit.nTrain;
nVal     = preInit.nVal;
nTest    = preInit.nTest;

%% load data
rng(dataSeed);
nImg = [32 32];
ncin = 3;

[Y,C,Ytest,Ctest] = setupCIFAR10(nTrain+nVal,nTest);
Y     = normalizeData(Y,prod(nImg)*ncin);
Ytest = normalizeData(Ytest,prod(nImg)*ncin);

% divide into training and validation data
id = randperm(size(C,2));
idt = id(1:nTrain);
idv = id(nTrain+1:end);
Yt  = Y(:,:,:,idt); Ct = C(:,idt);
Yv  = Y(:,:,:,idv); Cv = C(:,idv);


[Yt,Ct,Yv,Cv] = gpuVar(useGPU,precision,Yt,Ct,Yv,Cv);


%% train

% setup objective functions
fctn = dnnBatchObjFctn(net,pRegKb,pLoss,pRegW,Yt,Ct,'batchSize',256,'useGPU',useGPU,'precision',precision);
fval = dnnBatchObjFctn(net,[],pLoss,[],Yv,Cv,'batchSize',256,'useGPU',useGPU,'precision',precision);

% setup optimization
opt = sgd();
opt.learningRate = lr;
% opt.learningRate = @(epoch) 1e-3/sqrt(epoch);
opt.maxEpochs    = numEpochs;
opt.maxWorkUnits = Inf;
opt.nesterov     = true;
opt.ADAM         = false;
opt.miniBatch    = batchSize;
opt.momentum     = 0.9;
opt.out          = 1;

% ----------------------------------------------------------------------- %
% initialize with VarPro
fctnInit = dnnVarProObjFctn(net,pRegKb,pLoss,pRegW,classSolver,Yt,Ct,'useGPU',useGPU,'precision',precision);
fctnInit.pRegW.precision = 'double';

% solve for varpro W
[~,para] = eval(fctnInit,theta);
W = para.W;
% ----------------------------------------------------------------------- %

% run optimization
rng(2); % for reproducibility
startTime = tic;
[xc,His,xOpt] = solve(opt,fctn,[theta(:); W(:)],fval);
His = gather(His);
endTime = toc(startTime);


fprintf('\npredict labels for last iterate\n')
xOpt = gpuArray(xOpt);
[COpt,POpt] = getLabels(fval,xOpt);
COpt = uint8(gather(COpt));
POpt = single(gather(POpt));

fprintf('\npredict labels for best iterate\n')

[Cc,Pc] = getLabels(fval,xc);
Cc = uint8(gather(Cc));
Pc = single(gather(Pc));

% store test results
ftest = dnnBatchObjFctn(net,[],pLoss,[],Ytest,Ctest,'batchSize',256,'useGPU',useGPU,'precision',precision);
[CtestOpt,PtestOpt] = getLabels(ftest,xOpt);
CtestOpt = uint8(gather(CtestOpt));
PtestOpt = single(gather(PtestOpt));

[Ctestc,Ptestc] = getLabels(ftest,xc);
Ctestc = uint8(gather(Ctestc));
Ptestc = single(gather(Ptestc));

[thOpt,WOpt] = split(fctn,xOpt);
[thOpt,WOpt] = gather(thOpt,WOpt);

[thc,Wc] = split(fctn,xc);
[thc,Wc] = gather(thc,Wc);

net.useGPU = false;

save([resFile,'.mat'], 'net','nTrain', 'His', 'endTime','thOpt','WOpt','COpt','POpt','Cc','Pc','thc','Wc','Ctestc','CtestOpt','Ptestc','PtestOpt');
diary off;
