%% Indian Prines with L-BFGS with VarPro (EVP_IndianPines_LBFGSvpro.m)

% For details, see the paper:

% @article{newman2020train,
% 	title={Train Like a (Var)Pro: Efficient Training of Neural Networks with Variable Projection},
% 	author={Elizabeth Newman and Lars Ruthotto and Joseph Hart and Bart van Bloemen Waanders},
% 	year={2020},
% 	journal={arXiv preprint arxiv.org/abs/2007.13171},
% }

%% Saving information
clear; clc;

% file to save results and training flag
doSave  = 0; % save file

if doSave
    resFile = sprintf('%s-%s-%s',date,datestr(now,'hh-MM-ss','local'),mfilename);
    copyfile(['examples/GNvpro/',mfilename,'.m'],[resFile,'.m']);  % save copy of script
    diary([resFile,'.txt']);              % save diary of printouts
end


%% Setup data

% random seed for reproducibility
seed = 42;
rng(seed); 

% create original data
[YOrig,COrig,nY]    = setupIndianPines(false);

nImg            = nY(1:2); % image resolution
YOrig           = reshape(YOrig,prod(nImg),[]);
COrig           = reshape(COrig,prod(nImg),[]);
numClasses      = size(COrig,2) - 1;

% find pixels not in the background
idxForeground   = find((COrig*(1:size(COrig,2))'>1));
Ydata           = YOrig(idxForeground,:)';
Ydata           = normalizeData(Ydata,size(Ydata,1));
Cdata           = COrig(idxForeground,2:end)';

% number of examples and training data
nSamples    = size(Ydata,2);
nTrain      = 5000;
nVal        = nSamples - nTrain;
    
% shuffle pixels
idxShuffle  = randperm(nSamples);
Y           = Ydata(:,idxShuffle);
C           = Cdata(:,idxShuffle);

% pick equal number of examples per class
Yt = []; Yv = []; Ytest = [];
Ct = []; Cv = []; Ctest = [];

idt = [];  idv = [];  idTest = []; % indices
for k = 1:size(Cdata,1)   
    idk = find(Cdata(k,:));
    nb = min(numel(idk)-10,fix(nTrain/size(Cdata,1)));
    
    % take ~10% of training data for testing
    idxTrain = idk(1:nb);
    nTest = floor(0.2 * nb);
    idx2 = randperm(nb, nTest);

    idxTest = idxTrain(idx2);
    idxTrain(idx2) = [];
    
    % test data
    Ytest = cat(2,Ytest,Y(:,idxTest));
    Ctest = cat(2,Ctest,C(:,idxTest));
    idTest = cat(2,idTest,idxTest);
    
    % training data
    Yt = cat(2,Yt,Y(:,idxTrain));
    Ct = cat(2,Ct,C(:,idxTrain));
    idt = cat(2,idt,idxTrain);
    
    % validation data
    Yv = cat(2,Yv,Y(:,idk(1+nb:end)));
    Cv = cat(2,Cv,C(:,idk(1+nb:end)));
    idv = cat(2,idv,idk(nb+1:end));
    
end

%% Select parameters

rng(20);

% network
dynamic = 'antiSym-ResNN';
% dynamic = 'ResNN';
% dynamic = 'hamiltonian';

T       = 4;    % final time
nt      = 4;    % number of time steps (will not prolongate with nt=1)
nc      = 32;    % number of channels (width)
nLevels = 3;    % number of levels

% regularization
alpha1 = 1e-3; % theta
alpha2 = 1e-3; % W

%% Setup network

% first block
block1 = NN({singleLayer(dense([nc,size(Yt,1)],'Bin',eye(nc)))});

% second block (ResNN, keeps size fixed)
switch dynamic
    case 'ResNN'
        K       = dense([nc,nc]);
        layer   = singleLayer(K,'Bin',eye(nc));
        tY      = linspace(0,T,nt);
%         block2  = ResNNrk4(layer,tY,tY);
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
h = T / nt;

% combine both blocks
net = Meganet({block1,block2});

%% setup regression and Newton solver for this subproblem
pLoss = softmaxLoss();
classSolver = trnewton('atol',1e-10,'rtol',1e-10,'maxIter',100);
classSolver.linSol = GMRES('m',500,'tol',1e-10,'out',0);

%% regularization
reg1    = tikhonovReg(opEye(nTheta(block1)),alpha1);
reg2    = tikhonovReg(opTimeDer(nTheta(block2),nt,nt/T),alpha1);
pRegTh  = blockReg({reg1,reg2});
pRegW   = tikhonovReg(opEye((sizeFeatOut(net)+1)*size(Ct,1)),alpha2);

%% setup outer optimization scheme
opt                 = lBFGS();
opt.out             = 1;
opt.maxIter         = Inf;
opt.maxWorkUnits    = 500;
opt.atol            = 1e-16;
opt.rtol            = 1e-16;
opt.LS              = Wolfe();

%% setup objective function with training and validation data
fctn = dnnVarProObjFctn(net,pRegTh,pLoss,pRegW,classSolver,Yt,Ct);
fval = dnnObjFctn(net,[],pLoss,[],Yv,Cv);

%% solve the problem
th0 = initTheta(net);
HIS = cell(1,nLevels);
startTime = tic;

for level = 1:nLevels
    [th0,his]  = solve(opt,fctn,th0,fval);
    HIS{level} = his;
    
    % only for resnet right now
    if level < nLevels
        [net,th0] = prolongateWeights(fctn.net,th0);
        net.blocks{2}.tY = net.blocks{2}.ttheta;
        nt = numel(net.blocks{2}.tY)-1;
        fctn.net  = net;

        if exist('fval','var') && ~isempty(fval)
            fval.net = net;
        end

        % reset regularization
        alpha1  = alpha1 / 2;
        reg1    = tikhonovReg(opEye(nTheta(net.blocks{1})),alpha1);
        reg2    = tikhonovReg(opTimeDer(nTheta(net.blocks{2}),nt+1,T/nt),alpha1);
        pRegTh  = blockReg({reg1,reg2});
        fctn.pRegTheta = pRegTh;
        
    end
end

thOpt = th0;

endTime = toc(startTime);
disp(['Elapsed Time is ',num2str(endTime),' seconds.']);

[Jc,para] = eval(fctn,thOpt);
WOpt      = reshape(para.W,size(Ct,1),[]);

if doSave
    save(resFile,'net','thOpt','WOpt','HIS');
end

%% Numerical results

% training
YNt = forwardProp(net,thOpt,Yt);
errTrain = nnz(Ct - getLabels(pLoss,WOpt,YNt)) / 2;
accTrain = 100 * (1 - errTrain / size(YNt,2));

% validation
YNv = forwardProp(net,thOpt,Yv);
errVal = nnz(Cv - getLabels(pLoss,WOpt,YNv)) / 2;
accVal = 100 * (1 - errVal / size(YNv,2));

% test
YNtest = forwardProp(net,thOpt,Ytest);
errTest = nnz(Ctest - getLabels(pLoss,WOpt,YNtest)) / 2;
accTest = 100 * (1 - errTest / size(YNtest,2));


fprintf('Train acc = %0.2f\n  Val acc = %0.2f\n Test acc = %0.2f\n',[accTrain,accVal,accTest]);


return;

%% Plot convergence

plotVar = 'F';

xIdx = find(strcmp(HIS{1}.str,'TotalWork'),1);
yIdx = find(strcmp(HIS{1}.str,plotVar));

fig = figure(1); clf;
fig.Name = plotVar;

ax = gca;
ax.ColorOrderIndex = 1;
colOrd = get(ax,'ColorOrder');
plotStyle = {'--','-'};

workUnits = 0;
for i = 1:length(HIS)
    
    for j = 1:length(yIdx)
        if ~strcmp(plotVar,'accuracy')
            semilogy(workUnits+HIS{i}.his(:,xIdx),HIS{i}.his(:,yIdx(j)),...
                plotStyle{mod(j,2)+1},'LineWidth',2,'Color',colOrd(j,:));
        else
            plot(workUnits+HIS{i}.his(:,xIdx),HIS{i}.his(:,yIdx(j)),...
                plotStyle{mod(j,2)+1},'LineWidth',2,'Color',colOrd(j,:));
        end
            
        hold on;
    end
    
    workUnits = workUnits + HIS{i}.his(end,xIdx);
    xline(workUnits,'--');
    
end

legend('training','validation');
xlabel('Work Units','FontWeight','bold');
ylabel(plotVar,'FontWeight','bold');
set(gcf,'Color','w');
set(gca,'FontSize',14);

hold off;

%% Visualization

myColorMap = parula(256);
myColorMap(1,:) = 0;  % 1 is white, 0 is black
colormap(myColorMap);

fig = figure(1); clf;
fig.Name = 'Original';

    Ctrue = zeros(prod(nImg),1);
    Ctrue(idxForeground) = vec((1:size(Cdata,1))*Cdata);

    h = imagesc(reshape(Ctrue,nImg));
    colormap(myColorMap);
    axis('image');
    axis('off');

    
fig = figure(2); clf;


    YN = forwardProp(net,thOpt,Ydata);
    [Cp,P] = getLabels(pLoss,WOpt,YN);
    err = nnz(Cdata - Cp) / 2;
    acc = 100 * (1 - err / size(YN,2));

fig.Name = ['Prediction: acc = ',num2str(acc)];
    
    Cimg = zeros(prod(nImg),1);
    Cimg(idxForeground) = vec((1:size(Cp,1))*Cp);

    h = imagesc(reshape(Cimg,nImg));
    colormap(myColorMap);
    axis('image');
    axis('off');
    
fig = figure(3); clf;
fig.Name = 'Difference';
    
    h = imagesc(reshape(abs(Ctrue - Cimg),nImg));
    colormap(myColorMap);
    axis('image');
    axis('off');

    
% accuracy
ConfMat = zeros(numClasses);

for i = 1:nSamples
    j1 = find(Cp(:,i));  % predicted class
    j2 = find(Cdata(:,i)); % true class
    
    ConfMat(j1,j2) = ConfMat(j1,j2) + 1;
    
end

classTotal = sum(Cdata,2)';

fig = figure(4); clf;
fig.Name = 'Confusion Matrix';
    h = imagesc(ConfMat ./ classTotal);
    
    colormap(myColorMap);
    colorbar;
   





