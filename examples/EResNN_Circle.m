clear all;

[Ytrain,Ctrain,Yv,Cv] = setupCircle(1000,200);
Ytrain = [Ytrain; zeros(1,size(Ytrain,2))];
Yv = [Yv; zeros(1,size(Yv,2))];
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
nt = 16;
K     = dense([3,3]);
layer = singleLayer(K,'Bin',ones(3,1));
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
classSolver = newton();
classSolver.maxIter=4;
classSolver.linSol.maxIter=4;

opt = newton();
opt.out=2;
opt.atol=1e-16;
opt.linSol.maxIter=20;
opt.maxIter=77;
opt.LS.maxIter=20;
fctn = dnnVarProObjFctn(net,pRegK,pLoss,pRegW,classSolver,Ytrain,Ctrain);
fval = dnnObjFctn(net,[],pLoss,[],Yv,Cv);

%%
% th0 = 1e0*max(randn(nTheta(net),1),0);
th0 = repmat(1e-2*[randn(9,1);0],nt,1);
% W0  = [1;0;0;0;1;0]
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
%%
figure(2); clf;
imagesc(reshape(thetaOpt,5,[]))

