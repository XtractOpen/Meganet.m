%% Circle Example with Gauss-Newton, no VarPro (EVP_Circle_GN.m)

% For details, see the paper:
%
% E.Newman, L. Ruthotto, J. Hart, and B. van Bloemen Waanders. 
% Train Like a (Var)Pro: Efficient Training of Neural Networks with
% Variable Projection, (2020), https://arxiv.org/abs/2007.13171

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

nTrain = 5000;
nVal   = 1000;
nTest  = 1000;

[Y,C] = setupCircle(nTrain+nVal+nTest);

idx = randperm(size(Y,2));
Yt = Y(:,idx(1:nTrain));
Ct = C(1,idx(1:nTrain));

Yv = Y(:,idx(nTrain+1:nTrain+nVal));
Cv = C(1,idx(nTrain+1:nTrain+nVal));

Ytest = Y(:,idx(nTrain+nVal+1:end));
Ctest = C(1,idx(nTrain+nVal+1:end));

%% Select parameters

rng(20);

% regularization
alpha1 = 1e-6; % theta
alpha2 = 1e-8; % W

%% Setup network

layerWidths = [4,4,2];
inFeat = size(Y,1);

layers = cell(1,length(layerWidths));
for i = 1:length(layerWidths)
    if i > 1
        inFeat = layerWidths(i-1);
    end
    K = dense([layerWidths(i),inFeat]);
    layer = singleLayer(K,'Bout',eye(layerWidths(i),1));
    layers{i} = layer;
end
net = NN(layers,'useGPU',false);

% T  = 1;
% nt = 2;
% nc = 4;
% h  = T / nt;
% block1 = NN({singleLayer(dense([nc,size(Yt,1)],'Bin',eye(nc,1)))});
% 
% K       = dense([nc,nc]);
% layer   = singleLayer(K,'Bin',eye(nc,1));
% tY      = linspace(0,T,nt);
% block2  = ResNN(layer,nt,T/nt);
% 
% block3 = NN({singleLayer(dense([2,nc],'Bin',eye(2,1)))});
% 
% 
% % combine both blocks
% net = Meganet({block1,block2,block3});


%% setup regression and Newton solver for this subproblem
pLoss = logRegressionLoss();

%% regularization
pRegTh = tikhonovReg(opEye(nTheta(net)),alpha1);

% reg1    = tikhonovReg(opEye(nTheta(block1)),alpha1);
% reg2    = tikhonovReg(opTimeDer(nTheta(block2),nt,nt/T),alpha1);
% reg3    = tikhonovReg(opEye(nTheta(block3)),alpha1);
% pRegTh  = blockReg({reg1,reg2,reg3});

pRegW  = tikhonovReg(opEye((sizeFeatOut(net)+1)*size(Ct,1)),alpha2);

%% setup outer optimization scheme
opt                 = trnewton();
opt.out             = 1;
opt.maxIter         = 50;
opt.maxWorkUnits    = Inf;  % per level
opt.atol            = 1e-16;
opt.rtol            = 1e-16;
opt.linSol          = GMRES('m',10,'tol',1e-3,'out',0);

%% setup objective function with training and validation data
fctn = dnnObjFctn(net,pRegTh,pLoss,pRegW,Yt,Ct);
fval = dnnObjFctn(net,[],pLoss,[],Yv,Cv);

%% solve the problem
th0 = initTheta(net);

% initialize W0 with varpro
classSolver         = trnewton('atol',1e-10,'rtol',1e-10,'maxIter',100);
classSolver.linSol  = GMRES('m',[],'tol',1e-10,'out',0);
fctnInit            = dnnVarProObjFctn(net,pRegTh,pLoss,pRegW,classSolver,Yt,Ct);

[~,para] = eval(fctnInit,th0);
W0 = para.W;

% standard normal
% W0 = randn(size(Ct,1)*(sizeFeatOut(net)+pLoss.addBias),1);

% concatenate weights
th0 = [th0(:);W0(:)];

HIS = cell(1,1);
startTime = tic;
[th0,his]  = solve(opt,fctn,th0,fval);
HIS{1} = his;

thOpt = th0;

endTime = toc(startTime);
disp(['Elapsed Time is ',num2str(endTime),' seconds.']);

[Jc,para] = eval(fctn,thOpt);
WOpt      = reshape(thOpt(nTheta(net)+1:end),size(Ct,1),[]);
thOpt     = thOpt(1:nTheta(net));


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

plotVar = 'accuracy';

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

fig = figure(1); clf;
fig.Name = 'Ground Truth';

    [Y0,C0,ccb,xx,yy] = setupPeaks(5000,5);
    [~,ca] = contourf(xx,yy,ccb);
    cmap = [ .6 .77 .89; .92 .71 .63; 0.969 .875 .651;  0.796 0.671 0.824;.785 .867 .675;];
    colormap(cmap)
    set(ca,'EdgeColor','none');
    hold on;

    %viewFeatures2D(Y0,C0);
    viewFeatures2D(Y,C);

    axis('image');
    axis('square');
    axis('off');

    
fig = figure(2); clf;

    [Yrand,Crand] = setupPeaks(1000);

    YN = forwardProp(net,thOpt,Y);
    [Cp,P] = getLabels(pLoss,WOpt,YN);
    err = nnz(C - Cp) / 2;
    acc = 100 * (1 - err / size(YN,2));

fig.Name = ['Prediction: acc = ',num2str(acc)];

    viewContour2D([-3 3 -3 3],thOpt,WOpt,net,pLoss);

    axis square; axis off;
    hold on;

    viewFeatures2D(Y,C);

    hold off;
    

   





