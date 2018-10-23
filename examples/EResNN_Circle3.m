close all; clear all;

[Ytrain,Ctrain,Yv,Cv] = setupBox('ntrain',1000,'nval',200);

Ctrain = Ctrain(1,:); 
Cv = Cv(1,:);
rng(20)
figure(1); clf;
subplot(1,3,1);
viewFeatures2D(Ytrain,Ctrain)
title('input features');
%% setup network
T = 20;
nt = 16;

% first block
block1 = NN({singleLayer(dense([3,2]))});


K     = dense([3,3]);
layer = singleLayer(K,'Bout',ones(3,1));
layer.activation=@reluActivation;
block2   = ResNN(layer,nt,T/nt);
nt = block2.nt;
h = block2.h;
net = Meganet({block1,block2});


%% setup classifier
pLoss = softmaxLoss();
pLoss = logRegressionLoss();
%% solve the coupled problem
regOp1 = opEye(nTheta(block1));
regOp2 = opTimeDer(nTheta(block2),nt,h);
pRegK = tikhonovReg(blkdiag(regOp1,regOp2),1e-2,[]);
regOpW = opEye((sizeFeatOut(net)+1)*size(Ctrain,1));
pRegW = tikhonovReg(regOpW,1e-2);

classSolver = newton();
classSolver.maxIter=4;
classSolver.linSol.maxIter=4;
opt = newton();
opt.out=2;
opt.atol=1e-16;
opt.linSol.maxIter=20;
opt.maxIter=100;
opt.LS.maxIter=20;
fctn = dnnVarProObjFctn(net,pRegK,pLoss,pRegW,classSolver,Ytrain,Ctrain);
fval = dnnObjFctn(net,[],pLoss,[],Yv,Cv);


% th0 = 1e0*max(randn(nTheta(net),1),0);
th0 = [initTheta(block1); repmat(1e-1*[randn(9,1);0],nt,1)];
% W0  = [1;0;0;0;1;0]
thetaOpt = solve(opt,fctn,th0,fval);
[Jc,para] = eval(fctn,thetaOpt);
WOpt = para.W;
%%
[Yn,tmp] = forwardProp(net,thetaOpt,Yv);
figure(1);
subplot(1,3,2);
viewFeatures3D(Yn,Cv);
title('output features')
%%
subplot(1,3,3);
viewContour2D([-2 2 -1 1],thetaOpt,WOpt,net,pLoss);
hold on
viewFeatures2D(Yv,Cv);
return;
%%
