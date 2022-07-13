% Academic test problem for classification based on MATLAB's peaks function
%
% The optimization uses Variable Projection as described in:
%
% @article{newman2020train,
% 	title={Train Like a (Var)Pro: Efficient Training of Neural Networks with Variable Projection},
% 	author={Elizabeth Newman and Lars Ruthotto and Joseph Hart and Bart van Bloemen Waanders},
% 	year={2020},
% 	journal={arXiv preprint arxiv.org/abs/2007.13171},
% }

close all; clear all;

% rng(42);
[Ytrain,Ctrain] = setupPeaks(1000,5);
[Yv,Cv] = setupPeaks(1000,5);

dynamic = 'antiSym-ResNN';
% dynamic = 'ResNN';

% rng(20); %seed random number generator
figure(1); clf;
subplot(1,2,1);
viewFeatures2D(Ytrain,Ctrain);
title('input features');
axis equal
axis tight
%% setup network
T  = 5;   % final time
nt = 8;  % number of time steps
h = T/nt;
nc = 16;   % number of channels (width)


% first block (single layer that opens up)
block1 = NN({singleLayer(dense([nc,2]))});

% second block (ResNN, keeps size fixed)
switch dynamic
    case 'ResNN'
        K      = dense([nc,nc]);
        layer  = singleLayer(K,'Bout',ones(nc,1));
        block2 = ResNN(layer,nt,T/nt);
    case 'antiSym-ResNN'
        K      = getDenseAntiSym([nc,nc]);
        layer  = singleLayer(K,'Bout',ones(nc,1));
        tY      = linspace(0,T,nt);
        block2  = ResNNrk4(layer,tY);
    case 'leapfrog'
         K      = dense([nc,nc]);
       layer  = doubleSymLayer(K,'Bout',ones(nc,1));
        block2 = LeapFrogNN(layer,nt,T/nt);
    case 'hamiltonian'
        K      = dense([nc/2,nc/2]);
        layer  = doubleSymLayer(K,'Bout',ones(nc/2,1));
        block2 = DoubleHamiltonianNN(layer,layer,nt,T/nt);
    otherwise
        error('Example %s not yet implemented',dynamic);
end

% combine both blocks
net = Meganet({block1,block2});
%% regularization
reg1 = tikhonovReg(opEye(nTheta(block1)),1e-3);
% reg2 = tikhonovReg(opTimeDer(nTheta(block2),nt,h),1e-3);
% pRegTh = blockReg({reg1,reg2});
pRegTh = tikhonovReg(opEye(nTheta(net)),1e-3);
regOpW = opEye((sizeFeatOut(net)+1)*size(Ctrain,1));
pRegW = tikhonovReg(regOpW,1e-3);

%% setup classification and Newton solver for this subproblem
pLoss = softmaxLoss();
classSolver = trnewton('atol',1e-10,'rtol',1e-10);

%% setup outer optimization scheme
opt      = trnewton('linSol',GMRES('m',20));
opt.out  = 2;
opt.maxIter=30;

%% setup objective function with training and validation data
fctn = dnnVarProObjFctn(net,pRegTh,pLoss,pRegW,classSolver,Ytrain,Ctrain);
fval = dnnObjFctn(net,[],pLoss,[],Yv,Cv);

%% solve the problem
th0       = initTheta(net);
tic;
thetaOpt  = solve(opt,fctn,th0,fval);
toc
[Jc,para] = eval(fctn,thetaOpt);
WOpt      = reshape(para.W,[],5);

%% plot results
[Yn,tmp] = forwardProp(net,thetaOpt,Yv);
figure(1);
subplot(1,2,2);
viewContour2D([-3 3 -3 3],thetaOpt,WOpt,net,pLoss);
axis equal
hold on
viewFeatures2D(Yv,Cv);
title('classification result');
