% =========================================================================
%
% CIFAR10 Neural Network Example similar to the one from TensorFlow tutorial
%
% For TensorFlow.jl code see: 
%    https://github.com/malmaud/TensorFlow.jl/blob/master/examples/mnist_full.jl
% =========================================================================

% saving information

clear; clc;

% for turing
myDir = '/local/home/enewma5/MeganetGNvpro';
addpath(genpath(myDir));
numEpochs  = 2;
numSamples = 4;
maxIter    = 3;
alphaW     = 1e-2;
alphaTh    = 1e-4;
gmresIter  = 30;
shuffle    = 1;

extraInfo = sprintf('--numEpochs-%d--numSamples-%d--maxIter-%d--GMRES-%d--alphaW-%0.2e--alphaTh-%0.2e--shuffle-%d',numEpochs,numSamples,maxIter,gmresIter,alphaW,alphaTh,shuffle);


% myDir   = pwd;

resFile = sprintf([myDir,'/results/%s-%s'],mfilename,date);
resFile = [resFile,extraInfo];
disp(resFile)
copyfile([myDir,'/examples/',mfilename,'.m'],[resFile,'.m']);  % save copy of script                                                                                                                                                                                                                                          
diary([resFile,'.txt']);              % save diary of printouts   

%% 
% set GPU flag and precision
useGPU    = 1; 
precision = 'single';

% load pre-initialized weights and network
preInit = load('results/ECNN_CIFAR10_networkInit.mat');
net    = preInit.net;
pRegW  = preInit.pRegW;
pRegW.alpha = alphaW;

pRegKb = preInit.pRegKb;
pRegKt.alpha = alphaTh;

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

% objective functions
fctn = dnnObjFctn(net,pRegKb,pLoss,pRegW,Yt,Ct,'useGPU',useGPU,'precision',precision);
fval = dnnBatchObjFctn(net,[],pLoss,[],Yv,Cv,'batchSize',256,'useGPU',useGPU,'precision',precision);

% setup optimization
opt              = trnewton();
opt.linSol       = GMRES('m',gmresIter,'tol',1e-2);
opt.out          = 1;
opt.atol         = 5e-10;
opt.rtol         = 5e-10;
opt.maxIter      = maxIter;
opt.maxWorkUnits = Inf;
% opt.Delta        = 1; % need to start with small enough Delta

% ----------------------------------------------------------------------- %
% initialize with VarPro
fctnInit = dnnVarProObjFctn(net,pRegKb,pLoss,pRegW,classSolver,Yt,Ct,'useGPU',useGPU,'precision',precision);
fctnInit.pRegW.precision = 'double';

% solve for varpro W
[~,para] = eval(fctnInit,theta);
W = para.W;
% ----------------------------------------------------------------------- %

% time optimization
startTime = tic;

% stochastic variant
% numEpochs  = 2;
% numSamples = 4;
batchSize  = floor(sizeLastDim(Yt) / numSamples);
disp(batchSize);
rng(42);
HIS = cell(1,numEpochs);
for i = 1:numEpochs
    % batchSize = floor(sizeLastDim(Yt) / numSamples);
    His = cell(1,numSamples);
    fprintf('Epoch %d\n',i);

    % current trust region radius
    delta = opt.Delta;
    
    if shuffle
        origIdx = randperm(sizeLastDim(Yt));
    else
        origIdx = 1:sizeLastDim(Yt);
    end
    
    for k = 1:numSamples
        tic;
        fprintf('Sample %d\n',k);
        idx    = origIdx((k-1)*batchSize+1:k*batchSize);
        disp(idx(1:10));
        fctn.Y = Yt(:,:,:,idx);
        fctn.C = Ct(:,idx);
        opt.Delta = delta;

        [x,his] = solve(opt,fctn,[theta(:);W(:)],fval);
        [theta,W] = split(fctn,x);
        [theta,W] = gpuVar(fctn.useGPU,fctn.precision,theta,W);

        His{k} = gather(his);

        ii = find(strcmp(His{k}.str,'Delta'));
        delta = double(His{k}.his(end,ii));
        toc;
    end
    HIS{i} = His;
end

endTime = toc(startTime);


fprintf('\npredict labels for last iterate\n')

% performance on validation data
[COpt,POpt] = getLabels(fval,[theta(:);W(:)]);
COpt = uint8(gather(COpt));
POpt = single(gather(POpt));

% performance on testdata
Ytest = normalizeData(Ytest,prod(nImg)*ncin);
ftest = dnnBatchObjFctn(net,[],pLoss,[],Ytest,Ctest,'batchSize',256,'useGPU',useGPU,'precision',precision);
[CtestOpt,PtestOpt] = getLabels(ftest,[theta(:);W(:)]);
CtestOpt = uint8(gather(CtestOpt));
PtestOpt = single(gather(PtestOpt));

[thOpt,WOpt] = gather(theta,W);

net.useGPU = false;
save([resFile,'.mat'], 'net', 'nTrain', 'HIS', 'endTime','thOpt','WOpt','COpt','POpt','CtestOpt','PtestOpt');

diary off;
