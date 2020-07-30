%% Circle Example with Gauss-Newton with VarPro (EVP_Circle_GNvpro.m)

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

% network
% dynamic = 'antiSym-ResNN';
% dynamic = 'ResNN';
% dynamic = 'hamiltonian';

nLevels = 1;    % number of levels

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

%% setup regression and Newton solver for this subproblem
pLoss = logRegressionLoss();
classSolver = trnewton('atol',1e-10,'rtol',1e-10,'maxIter',100);
classSolver.linSol = GMRES('m',[],'tol',1e-10,'out',0);

%% regularization
pRegTh = tikhonovReg(opEye(nTheta(net)),alpha1);
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
        fctn.net  = net;

        if exist('fval','var') && ~isempty(fval)
            fval.net = net;
        end

        % reset regularization
        alpha1  = alpha1 / 2;
        reg1    = tikhonovReg(opEye(nTheta(net.blocks{1})),alpha1);
        reg2    = tikhonovReg(opTimeDer(nTheta(net.blocks{2}),net.blocks{2}.nt,T/net.blocks{2}.nt),alpha1);
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
    

   





