
addpath(genpath('.'))
clear all;


tic;

doSave  = 1;

resFile = sprintf('resultsPeaksADAM/%s-%s-%s-%s',date,mfilename);
resFile = resFile(1:end-1);
if doSave
    copyfile([mfilename,'.m'],[resFile,'.m']);  % save copy of script                                                                                                                                                                
    % diary([resFile,'.txt']);              % save diary of printouts                                                                                                                                                                           
end
% parameters to vary                                                                                                                                                                                                                          
batchSize    = [10,5,1];
learningRate = [1e-3,1e-2];
width        = [8];
alphaW       = [1e0];

paramStruct = cell(1,numel(batchSize)*numel(learningRate)*numel(width)*numel(alphaW));

count = 1;
for b = 1:length(batchSize)
    for lr = 1:length(learningRate)
        opt                 = sgd();
        opt.out             = 0;
        opt.maxEpochs       = 50;
        opt.learningRate    = learningRate(lr);
        opt.miniBatch       = batchSize(b);
        opt.ADAM            = 1;
        opt.atol            = 1e-16;
        opt.rtol            = 1e-16;

        for a = 1:length(alphaW)
            for w = 1:length(width)
                paramStruct{count} = createParameterStructADAM('width',width(w),...,
                    'opt',opt,'alphaW',alphaW(a));
                paramStruct{count}.filename = [resFile,'--',paramStruct{count}.filename(6:end)];
                
                paramStruct{count}.dynamic    = 'antiSym-ResNN';
                paramStruct{count}.depth      = 8;
                paramStruct{count}.finalTime  = 5;
                paramStruct{count}.multiLevel = 1;
                paramStruct{count}.alphaTh    = 5e-6;
                
                count = count + 1;
            end
        end
    end
end


parfor i = 1:length(paramStruct)
    disp(['starting object ',num2str(i)]);
    peaksTrainADAMFunction(paramStruct{i},doSave);
    disp(['finished object ',num2str(i)]);
end

% output = peaksTrainADAMFunction(paramStruct{1},0);  

toc;


%% 
function[output] = peaksTrainADAMFunction(paramStruct,doSave,seed)

if ~exist('doSave','var'), doSave = 0; end
if ~exist('seed','var'), seed = 20; end

% for reproducibility
rng(seed);

%% 
% peaks function
f = @(x) peaks(x(1,:),x(2,:));

% dimension of input features
nFeatIn = 2;

% number 
nTrain  = 2000;
nVal    = 1000;

% domain
a = -3;
b = 3;

% training and validation data
Yt = a + (b - a) * rand(nFeatIn,nTrain);    
Ct = f(Yt);

Yv = a + (b - a) * rand(nFeatIn,nVal);      
Cv = f(Yv);

%%
% setup network and initialize
T  = paramStruct.finalTime;    % final time
nt = paramStruct.depth;     % number of time steps (depth)
nc = paramStruct.width;    % number of channels (width)

dynamic = paramStruct.dynamic;

% first block
block1 = NN({singleLayer(dense([nc,size(Yt,1)]));

% second block (ResNN, keeps size fixed)
switch dynamic
    case 'ResNN'
        K       = dense([nc,nc]);
        layer   = singleLayer(K,'Bin',eye(nc));
        block2  = ResNN(layer,nt,T/nt);
    case 'antiSym-ResNN'
        K       = getDenseAntiSym([nc,nc]);
        layer   = singleLayer(K,'Bin',eye(nc));
        tY      = linspace(0,T,nt);
        block2  = ResNNrk4(layer,tY,tY);
    case 'leapfrog'
        K      = dense([nc,nc]);
        layer  = doubleSymLayer(K,'Bout',ones(nc,1));
        block2 = LeapFrogNN(layer,nt,T/nt);
    case 'hamiltonian'
        K       = dense([nc,nc]);
        block2  = HamiltonianNN(@tanhActivation,K,eye(nc),nt,T/nt);
    otherwise
        error('Example %s not yet implemented',dynamic);
end
net = Meganet({block1,block2});

% initialize
th0 = initTheta(net);
W0  = randn(size(Ct,1),numelFeatOut(net)+1);

%%
% loss 
pLoss = regressionLoss('addBias',1);

% initial evaluation
trainInitMisfit = getMisfit(pLoss,W0,forwardProp(net,th0(:),Yt),Ct);
valInitMisfit   = getMisfit(pLoss,W0,forwardProp(net,th0(:),Yv),Cv);

% ----------------------------------------------------------------------- %
% regularization
alphaTh  = paramStruct.alphaTh; % regularization on theta
alphaW   = paramStruct.alphaW;

reg1    = tikhonovReg(opEye(nTheta(block1)),alphaTh);
reg2    = tikhonovReg(opTimeDer(nTheta(block2),nt,nt / T),alphaTh);
pRegTh  = blockReg({reg1,reg2}); 
pRegW   = tikhonovReg(opEye(numel(W0)),alphaW);

% ----------------------------------------------------------------------- %
% optimizer
opt = paramStruct.opt;

% ----------------------------------------------------------------------- %
% objective functions
fctn = dnnObjFctn(net,pRegTh,pLoss,pRegW,Yt,Ct);
fval = dnnObjFctn(net,[],pLoss,[],Yv,Cv);

%% 
% train with slimTik
rng(seed);

startTime = tic;
[thW,His,~,infoOptLoss] = solve(opt,fctn,[th0(:);W0(:)],fval);
endTime = toc(startTime);

HIS{1} = His;

% store results
[theta,W] = split(fctn,thW);
T         = reshape(W,size(Ct,1),[]);
[~,para]  = eval(fctn,[theta(:);T(:)]);

% final output
output.para  = para;
output.theta = theta;
output.W     = W;

% optimal output
output.HIS      = HIS;
output.paraOpt  = infoOptLoss.paraOptLoss;
output.thOpt    = infoOptLoss.xOptLoss(1:nTheta(net));
output.WOpt     = reshape(infoOptLoss.xOptLoss(nTheta(net)+1:end),size(Ct,1),[]);

% clear data and store objective function (contains history)
fctn.Y      = [];
fctn.C      = [];
output.fctn = fctn;

output.info.endTime = endTime;

if doSave
    save([paramStruct.filename,'.mat'],'paramStruct','output','trainInitMisfit','valInitMisfit');
end


end


function[paramStruct] = createParameterStructADAM(varargin)

paramStruct = struct(...
    'width',8,'depth',8,'finalTime',5,...,
    'dynamic','ResNN','multiLevel',1,...
    'opt',sgd(),...
    'alphaTh',1e-3,'alphaW',1e-2);

% set defaults
for k = 1:2:length(varargin)     % overwrites default parameter
    paramStruct.(varargin{k}) = varargin{k + 1};
end

paramStruct.filename = sprintf('ADAM_width-%0.2d_lr-%0.2e_batch-%0.2d_alphaW-%0.2e',...
                paramStruct.width,...
                paramStruct.opt.learningRate,paramStruct.opt.miniBatch,...
                paramStruct.alphaW);

end

