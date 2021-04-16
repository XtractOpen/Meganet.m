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
numEpochs  = 2;
numSamples = 4;
maxIter    = 30;
alphaW     = 5e-3;
alphaTh    = 1e-4;
wolfeIter  = 30;
shuffle    = 0;

extraInfo = sprintf('--numEpochs-%d--numSamples-%d--maxIter-%d--Wolfe-%d--alphaW-%0.2e--alphaTh-%0.2e--shuffle-%d',numEpochs,numSamples,maxIter,wolfeIter,alphaW,alphaTh,shuffle);

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
pRegKb.alpha = alphaTh;

pLoss  = preInit.pLoss;

net.useGPU    = useGPU;  net.precision    = precision;
pRegW.useGPU  = useGPU;  pRegW.precision  = precision; 
pRegKb.useGPU = useGPU;  pRegKb.precision = precision; 

theta  = preInit.theta;
theta  = gpuVar(useGPU,precision,theta);

classSolver = preInit.classSolver;

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

fctn = dnnVarProObjFctn(net,pRegKb,pLoss,pRegW,classSolver,Yt,Ct,'useGPU',useGPU,'precision',precision);
fctn.pRegW.precision = 'double';              

fval = dnnBatchObjFctn(net,[],pLoss,[],Yv,Cv,'batchSize',256,'useGPU',useGPU,'precision',precision);


% setup optimization
opt                 = lBFGS();
opt.LS              = Wolfe('maxFunEvals',wolfeIter);
% opt.LS              = Armijo2('maxFunEvals',20);
opt.out             = 1;
opt.maxIter         = maxIter;
opt.maxWorkUnits    = Inf;
opt.atol            = 1e-16;
opt.rtol            = 1e-16;

% run optimization
startTime = tic;

% stochastic variant
% numEpochs  = 2;
% numSamples = 4;
batchSize = floor(sizeLastDim(Yt) / numSamples);

HIS = cell(1,numEpochs);
rng(42);
for i = 1:numEpochs
    fprintf('Epoch %d\n',i);
    
    disp(batchSize)
    His = cell(1,numSamples);
    
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
        
        [theta,his] = solve(opt,fctn,theta,fval);
        theta = gpuVar(fctn.useGPU,fctn.precision,theta);
        His{k} = gather(his);
        toc;
    end
    HIS{i} = His;
end

[~,para] = eval(fctn,theta);
W  = para.W;

endTime = toc(startTime);

fprintf('\npredict labels for last iterate\n')
tic;
[COpt,POpt] = getLabels(fval,[theta(:);W(:)]);
COpt = uint8(gather(COpt));
POpt = single(gather(POpt));
toc

% store test results
Ytest = normalizeData(Ytest,prod(nImg)*ncin);                                                                                 
ftest = dnnBatchObjFctn(net,[],pLoss,[],Ytest,Ctest,'batchSize',256,'useGPU',useGPU,'precision',precision);
[CtestOpt,PtestOpt] = getLabels(ftest,[theta(:);W(:)]);
CtestOpt = uint8(gather(CtestOpt));
PtestOpt = single(gather(PtestOpt));

[thOpt,WOpt] = gather(theta,W);

net.useGPU = false;
save([resFile,'.mat'], 'net', 'nTrain', 'HIS', 'endTime','thOpt','WOpt','COpt','POpt','CtestOpt','PtestOpt');

diary off;
