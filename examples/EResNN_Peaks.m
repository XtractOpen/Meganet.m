% close all; clear all;

[Ytrain,Ctrain] = setupPeaks(5000,5);
[Yv,Cv] = setupPeaks(200,5);

% dynamic = 'antiSym-ResNN';
dynamic = 'ResNN';

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
nc = 8;   % number of channels (width)


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
        block2 = ResNN(layer,nt,T/nt);
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
h      = block2.h;

% combine both blocks
net = Meganet({block1,block2});
%% regularization
alpha  = 5e-6;
reg1 = tikhonovReg(opEye(nTheta(block1)),alpha);
reg2 = tikhonovReg(opTimeDer(nTheta(block2),nt,h),alpha);
pRegTh = blockReg({reg1,reg2});
regOpW = opEye((sizeFeatOut(net)+1)*size(Ctrain,1));
pRegW = tikhonovReg(regOpW,1e-10);

%% setup classification and Newton solver for this subproblem
pLoss = softmaxLoss();
classSolver = newton();
classSolver.maxIter=10;
classSolver.linSol.maxIter=10;

%% setup outer optimization scheme
opt      = newton();
opt.out  = 2;
opt.atol = 1e-16;
opt.maxIter=40;
opt.LS.maxIter=20;
opt.linSol.maxIter=20;

%% setup objective function with training and validation data
fctn = dnnVarProObjFctn(net,pRegTh,pLoss,pRegW,classSolver,Ytrain,Ctrain);
fval = dnnObjFctn(net,[],pLoss,[],Yv,Cv);

%% solve the problem
th0       = 1e-1*initTheta(net);
thetaOpt  = solve(opt,fctn,th0,fval);
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
