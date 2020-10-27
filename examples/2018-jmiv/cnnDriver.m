% =========================================================================
%
% Main driver for training CNN examples used in 
%
% Ruthotto L, Haber E: Deep Neural Networks motivated by PDEs,
%           Journal of Mathematical Imaging and Vision, 10.1007/s10851-019-00903-1, 2018
%
% Other References:
%
% Haber E, Ruthotto L: Stable Architectures for Deep Neural Networks,
%      Inverse Problems, 2017
%
% Chang B, Meng L, Haber E, Ruthotto L, Begert D, Holtham E:
%      Reversible Architectures for Arbitrarily Deep Residual Neural Networks,
%      AAAI Conference on Artificial Intelligence 2018
%
% Inputs:
%
%     dataset   - choice of data set, 'stl10','cifar10','cifar100'
%     dynamic   - choice of dynamic,'parabolic','leapfrog','hamiltionian','resnet'
%     ntrain    - number of examples used in training (max  = 5000)
%     nval      - number of examples used for validation (max = 5000-ntrain)
%     nf        - width of blocks in network
%     nt        - number of time steps in the PDE blocks
%     useGPU    - flag for GPU computing, default = 0
%     precision - flag for precision, default = 'single'
%     opt       - optimizer
%     augment   - function handle for data augmentation
%     alpha     - regularization parameters, alpha(1) - weight decay for W
%                                            alpha(2) - weight decay for theta
%                                            alpha(3) - smoothness of theta
%     normLayer - function handle for normalization layer used in opening and connectors
%     resFile   - filename for results, set to [] to de-activate
%     
%
% Output:
%
%     xc        - weights at last iteration
%     His       - convergence history
%     xOptAcc   - weights with best validation accuracy (if validation set nval>0)
%
% Examples:
%
%     cnnDriver();  runs minimal example
% =========================================================================

function [xc,His,xOptAcc] = cnnDriver(dataset,dynamic,ntrain,nval,nf,nt,h,useGPU,precision,opt,augment,alpha,normLayer,resFile)

if nargin==0
    opt = sgd('learningRate',1e-1,'maxEpochs',4,'out',1,'miniBatch',100);
    feval(mfilename,'cifar100','parabolic',100,100,[32 64 112 112],3,.1,0,'single',opt,[],[],@batchNormLayer2);
    return
end

% set default options
if not(exist('dataset','var')) || isempty(dataset)
    dataset = 'cifar10';
end

if not(exist('dynamic','var')) || isempty(dynamic)
    dynamic = 'parabolic';
end


if not(exist('ntrain','var')) || isempty(ntrain)
    ntrain = 40000;
end
if not(exist('nval','var')) || isempty(nval)
    nval = 10000;
end

if not(exist('nf','var')) || isempty(nf)
    nf = [32 64 112 112];
end

if not(exist('nt','var')) || isempty(nt)
    nt = 3;
end
if not(exist('h','var')) || isempty(h)
    h  = 1; 
end

if not(exist('useGPU','var')) || isempty(useGPU)
    useGPU = 0;
end

if not(exist('precision','var')) || isempty(precision)
    precision = 'single';
end

if not(exist('augment','var')) || isempty(augment)
    augment = @(Y) randomCrop(randomFlip(Y,.5),4);
end
if not(exist('alpha','var')) || isempty(alpha)
    alpha = [4e-4; 4e-4; 1e-4];
end
if not(exist('normLayer','var')) || isempty(normLayer)
    normLayer = @batchNormLayer2;
end

if not(exist('resFile','var'))
    resFile = sprintf('%s-%s-%s-%s',datestr(now,'YYYY-mm-dd-HH-MM-ss'), mfilename,dynamic,dataset);
end

if not(exist('opt','var')) || isempty(opt)
    opt = sgd();
    opt.nesterov     = false;
    opt.ADAM         = false;
    opt.miniBatch    = 128; % miniBatchSize;
    opt.out          = 1;
    lr     =[0.1*ones(50,1); 0.01*ones(20,1); 0.001*ones(20,1); 0.0001*ones(10,1)];
    opt.learningRate     = @(epoch) lr(epoch);
    opt.maxEpochs    = numel(lr);
end

%%
switch dataset
    case 'cifar10'
        [Y0,C] = setupCIFAR10(ntrain+nval,0);
    case 'stl10'
        [Y0,C] = setupSTL(ntrain+nval,0);
    case 'cifar100'
        [Y0,C] = setupCIFAR100(ntrain+nval,0);
end
nImg   = [size(Y0,1) size(Y0,2)];
cin    = size(Y0,3);
Y0     = normalizeData(Y0,prod(nImg)*cin);


% split into training and validation
id = randperm(ntrain+nval); idtrain = id(1:ntrain); idval = id(ntrain+(1:nval));

Ytrain = Y0(:,:,:,idtrain);
Ctrain = C(:,idtrain);
Yval   = Y0(:,:,:,idval);
Cval   = C(:,idval);

[Ytrain,Ctrain,Yval,Cval] = gpuVar(useGPU,precision,Ytrain,Ctrain,Yval,Cval);

%% choose convolution
conv = @convMCN;

%% setup network
miniBatchSize = 125;
act1 = @reluActivation;
actc = @reluActivation;
act  = @reluActivation;

if isscalar(nt)
    nt  = nt*ones(numel(nf)-1,1);
end
if isscalar(h)
    h =   h*ones(numel(nf)-1,1);
end


blocks    = cell(0,1); pRegTheta = cell(0,1);

%% Block to open the network
nL = normLayer([nImg(1:2) nf(1) miniBatchSize],'isWeight',1);
K  = conv(nImg,[3 3 cin nf(1)]);
blocks{end+1} = NN({singleLayer(K,'activation', act1,'normLayer',nL,'storeInterm',true)},'useGPU',useGPU,'precision',precision);
regD = gpuVar(useGPU,precision,[ones(nTheta(K),1); zeros(nTheta(nL),1)]);
pRegTheta{end+1} = tikhonovReg(opDiag(regD),alpha(2));
%% UNIT
for k=1:numel(h)
    nImgc = nImg/(2^(k-1));
    
    % PDE block
    switch dynamic
        case 'parabolic'
            K = conv(nImgc,[3 3 nf(k) nf(k)]);
            nL = tvNormLayer([nImgc(1:2) nf(k) miniBatchSize],'isWeight',1);
            layer = doubleSymLayer(K,'activation',act,'normLayer1',nL,'storeInterm',true);
            blocks{end+1} = ResNN(layer,nt(k),h(k),'useGPU',useGPU,'precision',precision);
            regD = gpuVar(useGPU,precision,repmat([ones(nTheta(K),1); zeros(nTheta(nL),1)],nt(k),1));
        case 'leapfrog'
            K = conv(nImgc,[3 3 nf(k) nf(k)]);
            nL = tvNormLayer([nImgc(1:2) nf(k) miniBatchSize],'isWeight',1);
            layer = doubleSymLayer(K,'activation',act,'normLayer1',nL,'storeInterm',true);
            blocks{end+1} = LeapFrogNN(layer,nt(k),h(k),'useGPU',useGPU,'precision',precision);
            regD = gpuVar(useGPU,precision,repmat([ones(nTheta(K),1); zeros(nTheta(nL),1)],nt(k),1));
        case 'hamiltonian'
            K = conv(nImgc,[3 3 nf(k)/2 nf(k)/2]);
            nL = tvNormLayer([nImgc(1:2) nf(k)/2 miniBatchSize],'isWeight',1);
            layer = doubleSymLayer(K,'activation',act,'normLayer1',nL,'storeInterm',true);
            blocks{end+1} = DoubleHamiltonianNN(layer,layer,nt(k),h(k),'useGPU',useGPU,'precision',precision);
            regD = gpuVar(useGPU,precision,repmat([ones(nTheta(K),1); zeros(nTheta(nL),1)],2*nt(k),1));
    end
    pRegTheta{end+1} = l1Reg(opTimeDer(nTheta(blocks{end}),nt(k),h(k)),[alpha(3); alpha(2)],[],'B2',opDiag(regD));
    
    % Connector block
    nL = normLayer([nImgc(1:2) nf(k+1) miniBatchSize], 'isWeight',1);
    Kc = conv(nImgc,[1,1,nf(k),nf(k+1)]);
    blocks{end+1} = NN({singleLayer(Kc,'activation',actc,'normLayer',nL)},'useGPU',useGPU,'precision',precision);
    regD = gpuVar(useGPU,precision,[ones(nTheta(Kc),1); zeros(nTheta(nL),1)]);
    pRegTheta{end+1} = tikhonovReg(opDiag(regD),alpha(2));
    
    if k<numel(h)
        % average pooling, downsample by factor of 2
        blocks{end+1} = connector(opPoolMCN([nImgc nf(k+1)],2));
        pRegTheta{end+1} = tikhonovReg(opEye(nTheta(blocks{end})),alpha(2));
    else
        % average across all pixels in channel
        blocks{end+1} = connector(opPoolMCN([nImgc nf(k+1)],nImgc(1:2)));
        pRegTheta{end+1} = tikhonovReg(opEye(nTheta(blocks{end})),alpha(2));
    end
end
%% Put it all together
net   = Meganet(blocks);
pLoss = softmaxLoss();

theta  = initTheta(net);
W      = 0.1*vec(randn(size(C,1),prod(sizeFeatOut(net))+1));
W = min(W,.2);
W = max(W,-.2);

% RegOpW = blkdiag(opGrad(nImgc/2,nf(end)*10,ones(2,1)),opEye(10));
RegOpW = blkdiag(opEye(numel(W)));
RegOpW.precision = precision;
RegOpW.useGPU = useGPU;

pRegTh = blockReg(pRegTheta,'useGPU',useGPU,'precision',precision);
pRegW  = tikhonovReg(RegOpW,alpha(1),[],'useGPU',useGPU,'precision',precision);

%% Prepare optimization
fctn = dnnBatchObjFctn(net,pRegTh,pLoss,pRegW,Ytrain,Ctrain,'batchSize',miniBatchSize,'useGPU',useGPU,'precision',precision,'dataAugment',augment);
if nval>0
    fval = dnnBatchObjFctn(net,[],pLoss,[],Yval,Cval,'batchSize',miniBatchSize,'useGPU',useGPU,'precision',precision);
else
    fval = [];
end
%% do learning
x0 = [theta(:);W(:)];

if not(isempty(resFile))
    dFile = [resFile '.txt'];
    if exist(dFile,'file'); delete(dFile); end
    diary(dFile);
    diary on
end
% print some status
fprintf('------- %s: %s CNN for %s ----------\n',mfilename,dynamic,dataset);
fprintf('no. of examples (train / val) :      %d / %d\n',sizeLastDim(Ytrain),sizeLastDim(Yval))
fprintf('no. of time steps:                   [%s]\n', sprintf('%d ',nt));
fprintf('time step size (h):                  [%s]\n', sprintf('%1.2f ',h));
fprintf('no. of channels:                     [%s]\n', sprintf('%d ',nf));
fprintf('augmentation:                         %s \n', func2str(augment));
fprintf('reg 1: weight decay for W:            %1.2e \n', alpha(1));
fprintf('reg 2: weight decay for theta:        %1.2e \n', alpha(2));
fprintf('reg 3: smoothness of theta:           %1.2e \n', alpha(3));
fprintf('normlayer:                            %s \n', func2str(normLayer));
fprintf('resfile:                              %s \n', resFile);
fprintf('start:                                %s \n', datestr(now,'YYYY-mm-dd HH:MM:ss'))
tic;
[xc,His,xOptAcc,xOptLoss] = solve(opt,fctn,x0,fval);
time = toc,
net.useGPU=0;
xc = gather(xc);
pRegTh.useGPU = 0;
pRegW.useGPU=0;
if not(isempty(resFile))
    save([dFile '.mat'],'xc','xOptAcc','xOptLoss', 'net', 'ntrain','nval','nf','nt','opt', 'alpha', 'augment', ...
        'nf','idtrain','idval', 'His', 'time','useGPU','precision','pRegW','pRegTh','normLayer');
    diary off
end

