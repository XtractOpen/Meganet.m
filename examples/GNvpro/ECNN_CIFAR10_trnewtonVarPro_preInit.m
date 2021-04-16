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
alphaW     = 5e-3;
alphaTh    = 1e-4;
gmresIter  = 30;
shuffle    = 0;

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
pRegW.alpha  = alphaW;

pRegKb = preInit.pRegKb;
pRegKb.alpha  = alphaTh;
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

clear Y C id idt idv

[Yt,Ct,Yv,Cv] = gpuVar(useGPU,precision,Yt,Ct,Yv,Cv);

%% train
fctn = dnnVarProObjFctn(net,pRegKb,pLoss,pRegW,classSolver,Yt,Ct,'useGPU',useGPU,'precision',precision);
fctn.pRegW.precision = 'double';      

fval = dnnBatchObjFctn(net,[],pLoss,[],Yv,Cv,'batchSize',256,'useGPU',useGPU,'precision',precision);


% setup optimization
opt              = trnewton();
opt.linSol       = GMRES('m',gmresIter,'tol',1e-2);
opt.out          = 1;
opt.atol         = 5e-10;
opt.rtol         = 5e-10;
opt.maxIter      = maxIter;
opt.maxWorkUnits = Inf;
% opt.Delta        = 25; % can start with larger Delta, but not for non-varpro case


% run optimization
startTime = tic;

% stochastic variant
% numEpochs  = 2;
% numSamples = 4;
batchSize  = floor(sizeLastDim(Yt) / numSamples);
disp(batchSize);
rng(42);
HIS = cell(1,numEpochs);
for i = 1:numEpochs
    
    fprintf('Epoch %d\n',i);

    delta = opt.Delta;
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
        opt.Delta = delta;
        
%         Z = forwardProp(net,theta,fctn.Y);
%         [~,para] = eval(fctn,theta);
%         W  = para.W;
%         [thetaTmp1,WTmp1,ZTmp1] = gather(theta,W,Z);
        
        [theta,his] = solve(opt,fctn,theta,fval);
        theta = gpuVar(fctn.useGPU,fctn.precision,theta);
        His{k} = gather(his);

        ii = find(strcmp(His{k}.str,'Delta'));
        delta = double(His{k}.his(end,ii));
        
        Z = forwardProp(net,theta,fctn.Y);
        [~,para] = eval(fctn,theta);
        W  = para.W;

%         [thetaTmp2,WTmp2,ZTmp2] = gather(theta,W,Z);
%         save(['intermediateResult',num2str(i),'-',num2str(k)],'thetaTmp1','thetaTmp2','WTmp1','WTmp2','ZTmp1','ZTmp2');
        toc;
    end
    
    % opt.maxIter =  opt.maxIter - 1;
    
    
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
