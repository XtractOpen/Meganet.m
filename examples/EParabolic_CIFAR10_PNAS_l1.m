
% =========================================================================
%
% STL10 Parabolic CNN Example described in the manuscript
%
% Ruthotto, Haber, Deep Neural Networks motivated by Partial Differential Equations,
% submitted to PNAS, April 2018
%
% This code was run using Meganet.m commit version b7ff9df
% 
% Trained weights and other outputs saved from this file are available at
%
%		www.mathcs.emory.edu/~lruthot/pubs/2018-PNAS-PDECNN
%
% =========================================================================
function EParabolic_CIFAR10_PNAS_l1(nex,doTest)
%%
if not(exist('doTest','var')) || isempty(doTest)
    doTest = 0
end

[Y0,C,Ytest,Ctest] = setupCIFAR10(nex,10000);

nImg = [32 32];
cin = 3;

%%
% choose file for results and specify whether or not to retrain
resFile = sprintf('%s-nex-%d',mfilename,nex);

% set GPU flag and precision
useGPU = 0;
precision='single';

[Y0,C,Ytest,Ctest] = gpuVar(useGPU,precision,Y0,C,Ytest,Ctest);

%% choose convolution
if useGPU
    cudnnSession = convCuDNN2DSession();
    conv = @(varargin)convCuDNN2D(cudnnSession,varargin{:});
else
    conv = @convMCN;
end

%% setup network
normLayerOpen  = @batchNormLayer;
normLayerResNN = @tvNormLayer;
normLayerConn  = @batchNormLayer;

miniBatchSize = 125;
act1 = @reluActivation;
actc = @reluActivation;
act  = @reluActivation;

nf = [16 64 128 256 256];
nt = [3 3 3 3];
h  = [1 1 1 1];

blocks    = cell(0,1); pRegTheta = cell(0,1);
alpha = [3e-4*h(1);5e-4];

%% Block to open the network
nL = normLayerOpen([(nImg(1:2)) nf(1) miniBatchSize]);
K  = conv(nImg,[3 3 cin nf(1)]);
blocks{end+1} = NN({singleLayer(K,'activation', act1,'normLayer',nL)},'useGPU',useGPU,'precision',precision);
regD = gpuVar(useGPU,precision,[ones(nTheta(K),1); zeros(nTheta(nL),1)]);
pRegTheta{end+1} = tikhonovReg(opDiag(regD),alpha(2));
%% UNIT
for k=1:numel(h)
    nImgc = nImg/(2^(k-1));
    % implicit layer
    K = conv(nImgc,[3 3 nf(k) nf(k)]);
    
    nL = normLayerResNN([(nImgc(1:2)) nf(k) miniBatchSize]);
    layer = doubleSymLayer(K,'activation',act,'normLayer1',nL,'normLayer2',[]);
    blocks{end+1} = ResNN(layer,nt(k),h(k),'useGPU',useGPU,'precision',precision);
    pRegTheta{end+1} = l1Reg(opTimeDer(nTheta(blocks{end}),nt(k),h(k)),alpha);

    % Connector block
    nL = normLayerConn([(nImgc(1:2)) nf(k+1) miniBatchSize]);
    Kc = conv(nImgc,[1,1,nf(k),nf(k+1)]);
    blocks{end+1} = NN({singleLayer(Kc,'activation',actc,'normLayer',nL)},'useGPU',useGPU,'precision',precision);
    regD = gpuVar(useGPU,precision,[ones(nTheta(Kc),1); zeros(nTheta(nL),1)]);
    pRegTheta{end+1} = tikhonovReg(opDiag(regD),alpha(2));
    
    if k<numel(h)
        blocks{end+1} = connector(opPoolMCN([nImgc nf(k+1)],2));
        pRegTheta{end+1} = tikhonovReg(opEye(nTheta(blocks{end})),alpha(2));
    end
end

%% Connector block
B = gpuVar(useGPU,precision,kron(eye(nf(k+1)),ones(prod(nImgc),1)));
blocks{end+1} = connector(B'/prod(nImgc));
pRegTheta{end+1} = tikhonovReg(opEye(nTheta(blocks{end})),alpha(2));

%% Put it all together
net   = Meganet(blocks);
pLoss = softmaxLoss();

theta  = initTheta(net);
W      = 0.1*vec(randn(10,nFeatOut(net)+1));
W = min(W,.2);
W = max(W,-.2);

% RegOpW = blkdiag(opGrad(nImgc/2,nf(end)*10,ones(2,1)),opEye(10));
RegOpW = blkdiag(opEye(numel(W)));
RegOpW.precision = precision;
RegOpW.useGPU = useGPU;

% pRegTheta.precision = precision;
% pRegTheta.useGPU = useGPU;
pRegTh = blockReg(pRegTheta,'useGPU',useGPU,'precision',precision);
pRegW  = tikhonovReg(RegOpW,5e-4,[],'useGPU',useGPU,'precision',precision);
%% Prepare optimization
fctn = dnnBatchObjFctn(net,pRegTh,pLoss,pRegW,Y0,C,'batchSize',miniBatchSize,'useGPU',useGPU,'precision',precision);
fval = dnnBatchObjFctn(net,[],pLoss,[],Ytest,Ctest,'batchSize',miniBatchSize,'useGPU',useGPU,'precision',precision);
sum(fval.C,2) 
%%
opt = sgd();
opt.nesterov     = false;
opt.ADAM         = false;
opt.miniBatch    = miniBatchSize;
opt.out          = 1;

%% do learning
lr     =[0.1*ones(60,1); 0.01*ones(20,1); 0.001*ones(20,1); 0.001*ones(20,1)];
if doTest
    lr = lr(1:2);
end

opt.learningRate     = @(epoch) lr(epoch);
opt.maxEpochs    = numel(lr);
opt.P = @(x) min(max(x,-1),1);
x0 = [theta(:);W(:)];

%%
% W0  = [1;0;0;0;1;0]
dFile = [resFile '-lvl-' num2str(k) '.txt'];
if exist(dFile,'file'); delete(dFile); end
diary(dFile);
diary on

fprintf('--- %s:  ---\n',mfilename);
fprintf('number of training data: %d\n',size(Y0,2))
fprintf('number of validation data: %d\n',size(Ytest,2))
fprintf('number of filters:         [%s]\n',num2str(nf))
fprintf('alpha :                    [%1.2e %1.2e]\n',alpha)
fprintf('number of time steps:      [%s]\n',num2str(nt))
tic;
[xc,His,xOpt] = solve(opt,fctn,x0,fval);
time = toc,
fval = dnnBatchObjFctn(net,[],pLoss,[],Ytest,Ctest,'batchSize',miniBatchSize,'useGPU',useGPU,'precision',precision);
fprintf('\npredict labels for last iterate\n')
tic;
[COpt,POpt] = getLabels(fval,xOpt);
COpt = uint8(gather(COpt));
POpt = single(gather(COpt));
toc
fprintf('\npredict labels for best iterate\n')
tic;
[Cc,Pc] = getLabels(fval,xc);
Cc = uint8(gather(Cc));
Pc = single(gather(Pc));
toc;

xOpt = gather(xOpt);
xc   = gather(xc);
save([dFile '.mat'], 'net', 'nex', 'nf', 'nt', 'His', 'time','xOpt','xc','COpt','Cc','Pc','POpt');
diary off



