% close all; clear all;

[Ytrain,Ctrain] = setupPeaks(1000,5);
[Yv,Cv] = setupPeaks(2000,5);

dynamic = 'aResNNrk4';
% dynamic = 'hamiltonian';

% rng(20); %seed random number generator
figure(1); clf;
subplot(1,2,1);
viewFeatures2D(Ytrain,Ctrain);
title('input features');
axis equal
axis tight
%% setup network
T  = 20;   % final time
nTh = 8;     % number of time steps for theta
hTh = T/nTh;  % time steps sizefor theta
tTh = 0:hTh:T;
nY  = 8;
hY  = T/nY;
tY  = 0:hY:T;

nc = 8;   % number of channels (width)


% first block (single layer that opens up)
block1 = NN({singleLayer(dense([nc,2]))});

% second block (ResNN, keeps size fixed)
switch dynamic
    case 'aResNN'
        K      = dense([nc,nc]);
        layer  = singleLayer(K,'Bin',ones(nc,1),'activation',@smoothReluActivation);
        block2 = aResNN(layer,tY,tTh);
    case 'aResNNrk4'
        K      = dense([nc,nc]);
        layer  = singleLayer(K,'Bin',ones(nc,1),'activation',@smoothReluActivation);
        block2 = ResNNrk4(layer,tY,tTh);
    case 'antiSym-ResNN'
        K      = getDenseAntiSym([nc,nc]);
        layer  = singleLayer(K,'Bin',ones(nc,1),'activation',@smoothReluActivation);
        block2 = ResNN(layer,nt,T/nt);
    case 'leapfrog'
         K      = dense([nc,nc]);
       layer  = doubleSymLayer(K,'Bin',ones(nc,1),'activation',@smoothReluActivation);
        block2 = LeapFrogNN(layer,nt,T/nt);
    case 'hamiltonian'
        K      = dense([nc/2,nc/2]);
        layer  = doubleSymLayer(K,'Bin',ones(nc/2,1),'activation',@smoothReluActivation);
        block2 = DoubleHamiltonianNN(layer,layer,nt,T/nt);
    otherwise
        error('Example %s not yet implemented',dynamic);
end
% h      = block2.h;

% combine both blocks
net = Meganet({block1,block2});
%% regularization
alpha  = 1e-3;
reg1 = tikhonovReg(opEye(nTheta(block1)),alpha);
reg2 = tikhonovReg(opTimeDer(nTheta(block2),nTh+1,hTh),alpha);
pRegTh = blockReg({reg1,reg2});
regOpW = opEye((vFeatOut(net)+1)*size(Ctrain,1));
pRegW = tikhonovReg(regOpW,1e-10);


%% setup outer optimization scheme
pLoss = softmaxLoss();
opt      = newton();
opt.out  = 2;
opt.atol = 1e-16;
opt.maxIter=70;
opt.LS.maxIter=20;
opt.linSol.maxIter=20;

%% setup objective function with training and validation data
fctn = dnnObjFctn(net,pRegTh,pLoss,pRegW,Ytrain,Ctrain);
fval = dnnObjFctn(net,[],pLoss,[],Yv,Cv);

%% solve the problem
th0 = split(net,initTheta(net));
th0{2} = 0*th0{2};
th0 = vec(th0);

x0       = [th0; randn(((vFeatOut(net)+1)*size(Ctrain,1)),1)];
xOpt  = solve(opt,fctn,x0,fval);
thetaOpt = xOpt(1:nTheta(net));
WOpt      = reshape(xOpt(nTheta(net)+1:end),[],5);

%% plot results
[Ydata,Yn,tmp] = forwardProp(net,thetaOpt,Yv);
figure(1);
subplot(1,2,2);
viewContour2D([-3 3 -3 3],thetaOpt,WOpt,net,pLoss);
axis equal
hold on
viewFeatures2D(Yv,Cv);
title('classification result');

%% 
[Jc,para,dJ,H] = eval(fctn,xOpt);

H11 = H.blocks{1};
I = eye(size(H11,2));
tic
H11mat = zeros(size(H11));
for k=1:size(H11,2)
    H11mat(:,k) = H11*I(:,k);
end
H11mat =(H11mat+H11mat')/2;
toc

H22 = H.blocks{2};
I = eye(size(H22,2));
tic
H22mat = zeros(size(H22));
for k=1:size(H22,2)
    H22mat(:,k) = H22*I(:,k);
end
H22mat =(H22mat+H22mat')/2;
toc
%%
tic;
[V11,Lam11]= eig(H11mat);
[lam11,id] = sort(diag(Lam11),'descend');
V11 = V11(:,id);
[V22,Lam22]= eig(H22mat);
[lam22,id] = sort(diag(Lam22),'descend');
V22 = V22(:,id);
toc;
%%
figure(2); clf;
subplot(2,2,1);
imagesc(V11);
title('eigenvectors of H(theta)')
colorbar
subplot(2,2,2);
semilogy((lam11));
title('eigenvalues of H(theta)')
subplot(2,2,3);
imagesc(V22);
colorbar
title('eigenvectors of H(W)')
subplot(2,2,4);
semilogy((lam22));
title('eigenvalues of H(W)')

%% th
npx = 31;
idx = [1,2]
dth11 = linspace(-1,1,npx)/(sqrt(lam11(idx(1))));
dth12 = linspace(-1,1,npx)/(sqrt(lam11(idx(2))));
J11 = zeros(numel(dth11),numel(dth12));
tic;
for k1=1:numel(dth11)
    for k2=1:numel(dth12)
        J11(k1,k2) = eval(fctn,[thetaOpt+ dth11(k1)*V11(:,idx(1)) + dth12(k2)*V11(:,idx(2)); WOpt(:)]);
    end
end
toc;
%% W
idx = [1,2];
dth21 = linspace(-1,1,npx)/(sqrt(lam22(idx(1))));
dth22 = linspace(-1,1,npx)/(sqrt(lam22(idx(2))));
J22 = zeros(numel(dth21),numel(dth22));
tic;
for k1=1:numel(dth21)
    for k2=1:numel(dth22)
        J22(k1,k2) = eval(fctn,[thetaOpt; WOpt(:)+dth21(k1)*V22(:,idx(1)) + dth22(k2)*V22(:,idx(2))]);
    end
end
toc;
%%
figure(3); clf;
subplot(1,2,1);
surfc(dth11,dth12,J11')

subplot(1,2,2);
surfc(dth21,dth22,J22')