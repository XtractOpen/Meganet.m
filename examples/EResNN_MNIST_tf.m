% =========================================================================
%
% MNIST Residual Neural Network Example
%
% =========================================================================
clear all; clc;

nImg = [28 28];
[Y0,C,Ytest,Ctest] = setupMNIST(2^12); % increase number of examples for more accuracy

% choose dynamics in resnet
% dynamic = 'parabolic';
dynamic = 'singleLayer';
% dynamic = 'doubleLayer';


% choose file for results and specify whether or not to retrain
resFile = sprintf('%s-%s.mat',mfilename,dynamic);
doTrain = true;

% set GPU flag and precision
useGPU = 0;
precision='single';

[Y0,C] = gpuVar(useGPU,precision,Y0,C);

%% choose convolution
convOp = getConvOp(useGPU);

% setup network
miniBatchSize=128;
if useGPU
    act = @smoothReluActivationArrayfun;
else
    act = @smoothReluActivation;
end
% act = @tanhActivation;
nc      = 8;
nt      = 4;
h       = .1;

B = gpuVar(useGPU,precision,kron(eye(nc),ones(prod(nImg),1)));
blocks    = cell(0,1); RegOps = cell(0,1);
blocks{end+1} = NN({singleLayer(convOp(nImg,[1 1 1 nc]),'activation', act,'Bin',B)});
regD = gpuVar(useGPU,precision,[ones(nTheta(blocks{end}.layers{1}.K),1); zeros(size(B,2),1)]);
RegOps{end+1} = opDiag(regD);

switch dynamic
    case 'singleLayer'
        K = convOp(nImg,[3 3 nc nc]);
        layer = singleLayer(K,'Bin',B,'activation',act);
        blocks{end+1} = ResNN(layer,nt,h);
    case 'doubleLayer'
        K     = convOp(nImg,[3 3 nc nc]);
        layer = doubleLayer(K,K,'Bin1',B,'activation1', act,'activation2',@identityActivation,'normLayer1',nL);
        blocks{end+1} = ResNN(layer,nt,h);
    case 'parabolic'
        K     = convOp(nImg,[3 3 nc nc]);
        layer = doubleSymLayer(K,'Bin1',B,'activation1', act,'activation2',@identityActivation,'normLayer1',nL);
        blocks{end+1} = ResNN(layer,nt,h);
end
RegOps{end+1} = opTimeDer(nTheta(blocks{end}),nt,h,useGPU,precision);

net    = Meganet(blocks,'useGPU',useGPU,'precision',precision);

% setup loss function for training and validation set
pLoss  = softmaxLoss();

% setup regularizers
regDW  = gpuVar(useGPU,precision,[ones(10*numelFeatOut(net),1); zeros(10,1)]);
RegOpW = opDiag(regDW);

RegOpW.precision = precision;
RegOpW.useGPU = useGPU;
RegOpTh = blkdiag(RegOps{:});
RegOpTh.precision = precision;
RegOpTh.useGPU = useGPU;

pRegW  = tikhonovReg(RegOpW,2e-4,[],'useGPU',useGPU,'precision',precision);
pRegKb = tikhonovReg(RegOpTh,2e-4,[],'useGPU',useGPU,'precision',precision);

fctn = dnnBatchObjFctn(net,pRegKb,pLoss,pRegW,Y0,C,'batchSize',256,'useGPU',useGPU,'precision',precision);
fval = dnnBatchObjFctn(net,[],pLoss,[],Ytest,Ctest,'batchSize',256,'useGPU',useGPU,'precision',precision);

%% run training
if doTrain || not(exist(resFile,'file'))
    % initialize weights
    theta = initTheta(net);
    W     = 0.1*randn(10,numelFeatOut(net)+1);
    W     = max(min(0.2,W),-0.2);
    
    [theta,W] = gpuVar(fctn.useGPU,fctn.precision,theta,W);
    
    % setup optimization
     opt = sgd();
     opt.learningRate = @(epoch) .1/sqrt(.5*epoch);
     opt.maxEpochs    = 50;
     opt.nesterov     = false;
     opt.ADAM         = false;
     opt.miniBatch    = miniBatchSize;
     opt.momentum     = 0.0;
%      opt = sd();
     opt.out         = 1;
    
    % run optimization
    [xOpt,His] = solve(opt,fctn,[theta(:); W(:)],fval);
    [thOpt,WOpt] = split(fctn,gather(xOpt));
    save(resFile,'thOpt','WOpt','His')
else
    load(resFile)
end
return
%%
xOpt = gpuVar(useGPU,precision,[thOpt;WOpt]);
id = randperm(size(Y0,2)); id = id(1:miniBatchSize); % pick two random images

[Y,Yall] = forwardProp(net,xOpt,Y0(:,id));
th = split(net,gpuVar(useGPU,precision,thOpt));
%%
fig = figure(1); clf;
fig.Name = 'input';
montageArray(reshape(gather(Y0(:,id(1:2))),28,28,[]),1);
axis equal tight
%%
[Yin] = forwardProp(net.blocks{1},th{1},Y0(:,id));
fig = figure(2); clf;
fig.Name = 'inputResNN';
montageArray(reshape(gather(Yin(:,1:2)),28,28,[]),nc);
axis equal tight
colormap(flipud(colormap('gray')))


%%
[Yres] = forwardProp(net.blocks{2},th{2},Yin);
fig = figure(3); clf;
fig.Name = 'outputResNN';
montageArray(reshape(gather(Yres(:,1:2)),28,28,[]),nc);
axis equal tight
colormap(flipud(colormap('gray')))


%%
[Ycoup] = forwardProp(net.blocks{3},th{3},Yres);
fig = figure(3); clf;
fig.Name = 'outputCoup';
montageArray(reshape(gather(Ycoup(:,1:2)),28,28,[]),nc);
axis equal tight
colormap(flipud(colormap('gray')))

%%
[Yout] = forwardProp(net.blocks{4},th{4},Ycoup);
fig = figure(4); clf;
fig.Name = 'outputResNN';
montageArray(reshape(gather(Yout(:,1:2))',1,1,[]),nc);
axis equal tight
colormap(flipud(colormap('gray')))


%%
doPrint = 0;
%%
if doPrint
    figDir = '/Users/lruthot/Dropbox/TeX-Base/images/DeepLearning/architectures';
    for k=1:4
        fig=figure(k);
        axis off;
        printFigure(gcf,fullfile(figDir,['singleBlock-' fig.Name '.png']),...
            'printOpts','-dpng','printFormat','.png');
    end
end
        