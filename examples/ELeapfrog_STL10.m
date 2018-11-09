% =========================================================================
%
% STL10 Leopfrog CNN Example
%
% References:
%
% Ruthotto L, Haber E: Deep Neural Networks motivated by PDEs,
%           arXiv:1804.04272 [cs.LG], 2018
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
%     ntrain    - number of examples used in training (max  = 5000)
%     nval      - number of examples used for validation (max = 5000-ntrain)
%     nf0       - width of first ResNN layer (others will be multiple of this)
%     nt        - number of time steps in ResNNs
%     useGPU    - flag for GPU computing, default = 0
%     precision - flag for precision, default = 'single'
%     opt       - optimizer
%     resFile   - filename for results, set to [] to de-activate
%
% Output: 
% 
%     xc        - weights at last iteration
%     His       - convergence history
%     xOpt      - weights with best validation accuracy
%
% Examples:
%
%     ELeapfrog_STL10();                             runs minimal example
%     ELeapfrog_STL10(4000,1000,32,3,opt);           better architecure
% =========================================================================

function [xc,His,xOpt] = ELeapfrog_STL10(ntrain,nval,nf0,nt,useGPU,precision,opt,resFile)

if nargin==0
    opt = sgd('learningRate',1e-1,'maxEpochs',4,'out',1,'miniBatch',100);
    feval(mfilename,100,100,2,3,0,'single',opt,[]);
    return
end

% set default options
if not(exist('ntrain','var')) || isempty(ntrain)
    ntrain = 4000;
end
if not(exist('nval','var')) || isempty(nval)
    ntrain = 1000;
end

if not(exist('nf0','var')) || isempty(nf0)
    ntrain = 32;
end

if not(exist('nt','var')) || isempty(nt)
    nt = 3;
end

if not(exist('useGPU','var')) || isempty(useGPU)
    useGPU = 0;
end

if not(exist('precision','var')) || isempty(precision)
    precision = 'single';
end

if not(exist('resFile','var'))
    resFile = sprintf('%s-%s',datestr(now,'YYYY-mm-dd-HH-MM-ss'), mfilename);
end


if not(exist('opt','var')) || isempty(opt)
    opt = sgd();
    opt.nesterov     = false;
    opt.ADAM         = false;
    opt.miniBatch    = 125;
    opt.out          = 1;
    lr     =[0.1*ones(50,1); 0.01*ones(20,1); 0.001*ones(20,1); 0.0001*ones(10,1)];
    opt.learningRate     = @(epoch) lr(epoch);
    opt.maxEpochs    = numel(lr);
end

%%
[Y0,C] = setupSTL(ntrain+nval,0);
nImg = [size(Y0,1) size(Y0,2)];
cin = size(Y0,3);


% split into training and validation
id = randperm(ntrain+nval); idtrain = id(1:ntrain); idval = id(ntrain+(1:nval));

Ytrain = Y0(:,:,:,idtrain);
Ctrain = C(:,idtrain);
Yval   = Y0(:,:,:,idval);
Cval   = C(:,idval);



[Ytrain,Ctrain,Yval,Cval] = gpuVar(useGPU,precision,Ytrain,Ctrain,Yval,Cval);

%% choose convolution
if useGPU
    cudnnSession = convCuDNN2DSession();
    conv = @(varargin)convCuDNN2D(cudnnSession,varargin{:});
else
    conv = @convMCN;
end

%% setup network
% normLayer = @getTVNormLayer
normLayer = @batchNormLayer;

miniBatchSize = 125;
act1 = @reluActivation;
actc = @reluActivation;
act  = @reluActivation;

nf  = nf0*[1;4;8;16;16];
nt  = nt*[1;1;1;1];
h  = [1;1;1;1];

blocks    = cell(0,1); RegOps = cell(0,1);

%% Block to open the network
nL = normLayer([nImg(1:2) nf(1) miniBatchSize],'isWeight',1);
blocks{end+1} = NN({singleLayer(conv(nImg,[3 3 cin nf(1)]),'activation', act1,'normLayer',nL)},'useGPU',useGPU,'precision',precision);
RegOps{end+1} = opEye(nTheta(blocks{end}));
%% UNIT
for k=1:numel(h)
    nImgc = nImg/(2^(k-1));
    
    % PDE network
    K = conv(nImgc,[3 3 nf(k) nf(k)]);
    nL = tvNormLayer([nImgc(1:2) nf(k) miniBatchSize],'isWeight',1);
    layer = doubleSymLayer(K,'activation',act,'normLayer1',nL);
    blocks{end+1} = LeapFrogNN(layer,nt(k),h(k),'useGPU',useGPU,'precision',precision);
    regD = gpuVar(useGPU,precision,repmat([ones(nTheta(K),1); zeros(nTheta(nL),1)],nt(k),1));
    RegOps{end+1} = opDiag(regD);

    % Connector block
    nL = normLayer([nImgc(1:2) nf(k+1) miniBatchSize], 'isWeight',1);
    Kc = conv(nImgc,[1,1,nf(k),nf(k+1)]);
    blocks{end+1} = NN({singleLayer(Kc,'activation',actc,'normLayer',nL)},'useGPU',useGPU,'precision',precision);
    regD = gpuVar(useGPU,precision,[ones(nTheta(Kc),1); zeros(nTheta(nL),1)]);
    RegOps{end+1} = opDiag(regD);
    
    if k<numel(h)
        % average pooling, downsample by factor of 2
        blocks{end+1} = connector(opPoolMCN([nImgc nf(k+1)],2));
        RegOps{end+1} = opEye(nTheta(blocks{end}));
    else
        % average across all pixels in channel
        blocks{end+1} = connector(opPoolMCN([nImgc nf(k+1)],nImgc(1:2)));
        RegOps{end+1} = opEye(nTheta(blocks{end}));
    end
end
%% Put it all together
net   = Meganet(blocks);
pLoss = softmaxLoss();

theta  = initTheta(net);
W      = 0.1*vec(randn(10,prod(sizeFeatOut(net))+1));
W = min(W,.2);
W = max(W,-.2);

% RegOpW = blkdiag(opGrad(nImgc/2,nf(end)*10,ones(2,1)),opEye(10));
RegOpW = blkdiag(opEye(numel(W)));
RegOpW.precision = precision;
RegOpW.useGPU = useGPU;

RegOpTh = blkdiag(RegOps{:});
RegOpTh.precision = precision;
RegOpTh.useGPU = useGPU;

pRegW  = tikhonovReg(RegOpW,4e-4,[],'useGPU',useGPU,'precision',precision);
pRegKb = tikhonovReg(RegOpTh,4e-4,[],'useGPU',useGPU,'precision',precision);

%% Prepare optimization
fctn = dnnBatchObjFctn(net,pRegKb,pLoss,pRegW,Ytrain,Ctrain,'batchSize',miniBatchSize,'useGPU',useGPU,'precision',precision);
fval = dnnBatchObjFctn(net,[],pLoss,[],Yval,Cval,'batchSize',miniBatchSize,'useGPU',useGPU,'precision',precision);
%% do learning
x0 = [theta(:);W(:)];

if not(isempty(resFile))
    dFile = [resFile '.txt'];
    if exist(dFile,'file'); delete(dFile); end
    diary(dFile);
    diary on
end
% print some status
fprintf('------- %s ----------\n',mfilename);
fprintf('no. of examples (train / val) :      %d / %d\n',sizeLastDim(Ytrain),sizeLastDim(Yval))
fprintf('no. of time steps:                   [%s]\n', sprintf('%d ',nt));
fprintf('time step size (h):                  [%s]\n', sprintf('%1.2f ',h));
fprintf('no. of channels:                     [%s]\n', sprintf('%d ',nf));
fprintf('resfile:                              %s \n', resFile);
fprintf('start:                                %s \n', datestr(now,'YYYY-mm-dd HH:MM:ss'))
tic;
[xc,His,xOpt] = solve(opt,fctn,x0,fval);
time = toc,
xc = gather(xc);
xOpt = gather(xOpt);
if not(isempty(resFile))
    save([dFile '.mat'],'xc','xOpt', 'net', 'ntrain','nval','nf0','nt','opt', ...
        'nf','idtrain','idval', 'His', 'time','useGPU','precision');    
    diary off
end

