clear all;

[Ytrain,Ctrain,Yv,Cv] = setupCircle(1000,200);
Ctrain = Ctrain(1,:); 
Cv = Cv(1,:);
rng(20)
figure(1); clf;
subplot(1,3,1);
viewFeatures2D(Ytrain,Ctrain)
axis equal
title('input features');
%% setup network
T = 20;
nt = 8;
K     = dense([2,2]);
layer = singleLayer(K,'Bin',eye(2));
net   = ResNN(layer,nt,T/nt);
nt = net.nt;
h = net.h;

%% setup classifier
pLoss = logRegressionLoss();
%% solve the coupled problem
regOp = opTimeDer(nTheta(net),nt,h);
pRegK = tikhonovReg(regOp,1e-2,[]);
regOpW = opEye((prod(sizeFeatOut(net))+1)*size(Ctrain,1));
pRegW = tikhonovReg(regOpW,1e-2);
%%
classSolver = trnewton('atol',1e-10,'rtol',1e-10,'maxIter',100);


opt = trnewton();
opt.linSol=GMRES('m',20,'tol',1e-2);
opt.out  = 1;
opt.atol=1e-10;
opt.rtol=1e-10;
opt.maxIter=Inf;
opt.maxWorkUnits=2000;

fctn = dnnVarProObjFctn(net,pRegK,pLoss,pRegW,classSolver,Ytrain,Ctrain);
fval = dnnObjFctn(net,[],pLoss,[],Yv,Cv);

%%
th0 = initTheta(net);
thetaOpt = solve(opt,fctn,th0,fval);
[Jc,para] = eval(fctn,thetaOpt);
WOpt = para.W;
%%
[Yn,tmp] = forwardProp(net,thetaOpt,Yv);
figure(1);
subplot(1,3,2);
viewFeatures2D(Yn,Cv);
axis equal
title('output features')
subplot(1,3,3);
viewContour2D([-2 2 -1 1],thetaOpt,WOpt,net,pLoss);
hold on
viewFeatures2D(Yv,Cv);
axis equal
return;


