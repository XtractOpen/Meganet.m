%% DCR with L-BFGS with VarPro (EVP_DCR_LBFGSvpro.m)

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

%% Setup data and select parameters

rng(42);
[Yt,Ct,Yv,Cv,Ytest,Ctest,idx] = setupDCR(8000,1000);

rng(20);

% network
dynamic = 'antiSym-ResNN';
% dynamic = 'ResNN';
% dynamic = 'hamiltonian';

T       = 1;    % final time
nt      = 2;    % number of time steps (will not prolongate with nt=1)
nc      = 16;   % number of channels (width)
nLevels = 3;    % number of levels

% regularization
alpha1 = 1e-10; % theta
alpha2 = 1e-10; % W

%% setup network


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
h = T/nt;

% combine both blocks
net = Meganet({block1,block2});

%% setup regression and Newton solver for this subproblem
pLoss = regressionLoss();

%% setup outer optimization scheme
opt                 = lBFGS();
opt.out             = 1;
opt.maxIter         = Inf;
opt.maxWorkUnits    = 400;
opt.atol            = 1e-16;
opt.rtol            = 1e-16;

%% setup objective function with training and validation data
fctn = dnnVarProRegressionObjFctn(net,pLoss,Yt,Ct,'alpha1',alpha1,'alpha2',alpha2);
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
        fctn.alpha1 = fctn.alpha1 / 2;
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
WYt = reshape(WOpt,size(Ct,1),[]) * [YNt; ones(1,size(YNt,2))];
relErrTrain = sqrt(sum((WYt - Ct).^2,1)) ./ sqrt(sum(Ct.^2,1));

% validation
YNv = forwardProp(net,thOpt,Yv);
WYv = reshape(WOpt,size(Ct,1),[]) * [YNv; ones(1,size(YNv,2))];
relErrVal = sqrt(sum((WYv - Cv).^2,1)) ./ sqrt(sum(Cv.^2,1));

% test
YNtest = forwardProp(net,thOpt,Ytest);
WYtest = reshape(WOpt,size(Ct,1),[]) * [YNtest; ones(1,size(YNtest,2))];
relErrTest = sqrt(sum((WYtest - Ctest).^2,1)) ./ sqrt(sum(Ctest.^2,1));

fprintf('%-8smean\t+/-std\t\tmin\tmax\n','')
fprintf('Train:\t%0.4f\t+/-%0.4f\t%0.4f\t%0.4f\n',mean(relErrTrain),std(relErrTrain),min(relErrTrain),max(relErrTrain));
fprintf('Val:\t%0.4f\t+/-%0.4f\t%0.4f\t%0.4f\n',mean(relErrVal),std(relErrVal),min(relErrVal),max(relErrVal));
fprintf('Test:\t%0.4f\t+/-%0.4f\t%0.4f\t%0.4f\n',mean(relErrTest),std(relErrTest),min(relErrTest),max(relErrTest));

if doSave
    save(resFile,'relErrTrain','relErrVal','relErrTest','-append');
end

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
        semilogy(workUnits+HIS{i}.his(:,xIdx),HIS{i}.his(:,yIdx(j)),...
            plotStyle{mod(j,2)+1},'LineWidth',2,'Color',colOrd(j,:));
        hold on;
    end
    
    workUnits = workUnits + HIS{i}.his(end,xIdx);
    xline(workUnits,'--')
    
end

legend('training','validation');
xlabel('Work Units','FontWeight','bold');
ylabel(plotVar,'FontWeight','bold');
set(gcf,'Color','w');
set(gca,'FontSize',14);

hold off;

%% Visualization of approximation

viewI = @(x) log10(abs(x));

fig = figure(1); clf;
fig.Name = 'Orig';
    imagesc(viewI([Ct Cv Ctest]));
    colorbar
    %title('true labels')
    axis equal tight;
    axis('off');
    hold on;
    cax = caxis;
    plot(size(Yt,2)*[1;1],[1;size(Ct,1)],'-r','LineWidth',3);
    plot((size(Yt,2)+size(Yv,2))*[1;1],[1;size(Cv,1)],'-r','LineWidth',3);

    set(gca,'FontSize',16);
    set(gcf,'Color','w');

    % disp(cax);

    hold off;

    % truesize(size(Ct)+[0,20]);


fig = figure(2); clf;
fig.Name = 'Prediction';

    imagesc(viewI([WYt WYv WYtest]));
    colorbar
    %title('true labels')
    axis equal tight;
    axis('off');
    hold on;
    caxis(cax);
    plot(size(Yt,2)*[1;1],[1;size(Ct,1)],'-r','LineWidth',3);
    plot((size(Yt,2)+size(Yv,2))*[1;1],[1;size(Cv,1)],'-r','LineWidth',3);

    set(gca,'FontSize',16);
    set(gcf,'Color','w');

    % export_fig(['img/pred_',fileName,'.jpg']);
    

    hold off;
    
    %truesize(size(Ct)+[0,20]);

fig = figure(3); clf;
fig.Name = 'Difference';

    imagesc(viewI(abs([Ct Cv Ctest] - [WYt WYv WYtest])));
    colorbar
    %title('true labels')
    axis equal tight;
    axis('off');
    hold on;
    %caxis([0,1]);
    caxis(cax);
    plot(size(Yt,2)*[1;1],[1;size(Ct,1)],'-r','LineWidth',3);
    plot((size(Yt,2)+size(Yv,2))*[1;1],[1;size(Cv,1)],'-r','LineWidth',3);
    set(gca,'FontSize',16);
    set(gcf,'Color','w');

    hold off;
    
    %truesize(size(Ct)+[0,20]);


