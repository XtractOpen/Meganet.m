% =========================================================================
%
% MNIST Resisual Neural Network Example
%
% For Meganet-tf code type:
%
% python main.py --model resnet --dataset MNIST --batch_size 64 --max_steps 5000 
%                --num_units 1 --num_blocks 7 --output_channel 6 --test_batch_size 10000 
%                --no_data_augment --no_batchnorm --init_lrn_rate 0.01
%                --hp_weight_smoothness_rate 0.0 --hp_weight_decay_rate 0.0002
%
% The TF code achieved around 96.13% test accuracy, however, a different
% normalization scheme is being used. 
% =========================================================================
clear all; clc;

% normalize examples after each block
doInstNorm = 1;

nImg = [28 28];
fprintf('loading data using Meganet2\n')
[Y0,C,Ytest,Ctest] = setupMNIST(2^11);
Y0  = normalizeData(Y0)'; Ytest = normalizeData(Ytest)'; C = C'; Ctest = Ctest';

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
nc  = 6;
nt  = 6;
h   = 1.0;

B = gpuVar(useGPU,precision,kron(eye(nc),ones(prod(nImg),1)));
blocks    = cell(0,1); RegOps = cell(0,1);
blocks{end+1} = NN({singleLayer(conv(nImg,[3 3 1 nc]),'activation', act,'Bin',B)});
regD = gpuVar(useGPU,precision,[ones(nTheta(blocks{end}.layers{1}.K),1); zeros(size(B,2),1)]);
RegOps{end+1} = opDiag(regD);

if doInstNorm
    blocks{end+1} = NN({instNormLayer(nFeatOut(blocks{end}))});
    RegOps{end+1} = opEye(0);
end


K = conv(nImg,[3 3 nc nc]);
blocks{end+1} = ResNN(doubleLayer(K,K,'Bin1',B,'Bin2',B,'activation1', act,'activation2',@identityActivation),nt,h);
regD = gpuVar(useGPU,precision,repmat([ones(2*nTheta(blocks{end}.layer.K1),1); zeros(2*size(B,2),1)],nt,1));
RegOps{end+1} = opDiag(regD);

if doInstNorm
    blocks{end+1} = NN({instNormLayer(nFeatOut(blocks{end}))});
    RegOps{end+1} = opEye(0);
end


blocks{end+1} = NN({singleLayer(conv(nImg,[1 1 nc nc]),'activation', act,'Bin',B)});
regD = gpuVar(useGPU,precision,[ones(nTheta(blocks{end}.layers{1}.K),1); zeros(size(B,2),1)]);
RegOps{end+1} = opDiag(regD);

if doInstNorm
    blocks{end+1} = NN({instNormLayer(nFeatOut(blocks{end}))});
    RegOps{end+1} = opEye(0);
end

% final layer takes average of each channel
blocks{end+1} = connector((B/prod(nImg))');
RegOps{end+1} = opEye(nTheta(blocks{end}));
net    = Meganet(blocks,'useGPU',useGPU,'precision',precision);

% setup loss function for training and validation set
pLoss  = softmaxLoss();

% setup regularizers
regDW = gpuVar(useGPU,precision,[ones(10*nFeatOut(net),1); zeros(10,1)]);
RegOpW = blkdiag( opDiag(regDW));
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
    W     = 0.1*randn(10,nFeatOut(net)+1);
    W     = max(min(0.2,W),-0.2);
    
    [theta,W] = gpuVar(fctn.useGPU,fctn.precision,theta,W);
    
    % setup optimization
    opt = sgd();
    if doInstNorm
        opt.learningRate = @(epoch) .1/sqrt(epoch);
    else
        opt.learningRate =.01;
    end
    opt.maxEpochs    = 50;
    opt.nesterov     = false;
    opt.ADAM         = false;
    opt.miniBatch    = 64;
    opt.momentum     = 0.9;
    opt.out          = 1;
    
    % run optimization
    [xOpt,His] = solve(opt,fctn,[theta(:); W(:)],fval);
    [thOpt,WOpt] = split(fctn,xOpt);
    save(resFile,'thOpt','WOpt','His')
else
    load(resFile)
end

