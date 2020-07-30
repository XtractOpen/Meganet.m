clear; clc;
% Updated January 21, 2019

% rng(20);
%% Create data

rng(20);  % for reproducibility

%some controls
nFeatIn  = 2;
nFeatOut = 4;
nTargets = 3;
nSamples = 20;

% inputs
Y = randn(nFeatIn,nSamples);

% classification
idx = randi([1,nTargets],[1,nSamples]);
idx2 = idx + (0:nTargets:(nTargets*nSamples-1));
C = zeros(nTargets,nSamples);
C(idx2) = 1;

%% Create network

% single layer
% net     = singleLayer(dense([nFeatOut,nFeatIn]),'activation',@tanhActivation);
% regOp   = opEye(nTheta(net));

% resnet
T     = 2;
nt    = 16;
K     = dense([nFeatIn,nFeatIn]);
layer = singleLayer(K,'Bout',ones(nFeatIn,1));
net   = ResNN(layer,nt,T/nt);
nt    = net.nt;
h     = net.h;
regOp = opTimeDer(nTheta(net),nt,h);


%% Choose class solver and loss function
classSolver = trnewton('name','newtonW','out',0,'maxIter',100,'rtol',1e-6,'atol',1e-6);
classSolver.linSol.m = 2;

% regularization
pRegTh  = tikhonovReg(regOp,1e-2,[]);
regOpW  = opEye((prod(sizeFeatOut(net))+1)*size(C,1));
pRegW   = tikhonovReg(regOpW,1e-2);

pLoss = softmaxLoss('addBias',1);

%% Initia lize weights and evaluate

theta = 0.5 * vec(randn(nTheta(net),1));
% W = 0.5 * vec(randn(nTargets * (sizeFeatOut(net)+1),1));

% evaluate
fctn = dnnVarProObjFctn(net,pRegTh,pLoss,pRegW,classSolver,Y,C,'linSol',[]);
% fctn = dnnVarProObjFctn(net,pRegTh,pLoss,pRegW,classSolver,Y,C,'linSol',steihaugPCG('tol',1e-5,'maxIter',100));
fctn.optClass.linSol = GMRES('m',10);
[Jc,para,dJ,H,PC,J,S] = eval(fctn,theta(:));
WOpt = para.W;

%% Check adjoint
[OK,err] = checkAdjoint(J);
fprintf('Adjoint Test\n')
fprintf('Is OK? %d\t err = %0.2e\n',OK,err);

%% Check Jacobian
% 
fprintf('\n');
fprintf('Jacobian of W*Z(theta)\n')

S0 = vec(S);
dth = randn(size(theta));
dS  = vec((J * dth(:)));

for k = 1:15
    h = 2^(-k);
    tht = theta + h * dth;
    [~,~,~,~,~,~,St] = eval(fctn,tht(:));

    E0 = norm(S0 - St(:)) / norm(S0);
    E1 = norm(S0 + h * dS - St(:)) / norm(S0);

    % change to base 2
    d0 = floor(log2(E0));
    c0 = 2^(log2(E0) - d0);

    d1 = floor(log2(E1));
    c1 = 2^(log2(E1) - d1);

    fprintf('h=%0.2f x 2^(%0.2d)\t\tE0=%0.4f x 2^(%0.2d)\t\tE1=%0.4f x 2^(%0.2d)\n',1,log2(h),c0,d0,c1,d1);
end
fprintf('\n');

%% Check gradient

fprintf('\n\n');

dth = randn(size(theta));
dJdth = dth(:)' * dJ(:);

fprintf('Gradient Test\n')
for k = 1:15
    h = 2^(-k);

    tht = theta + h * dth;
    Jt = eval(fctn,tht(:));

    E0 = abs(Jc - Jt) / abs(Jc);
    E1 = abs(Jc + h * dJdth - Jt) / abs(Jc);
    
    % change to base 2
    d0 = floor(log2(E0));
    c0 = 2^(log2(E0) - d0);

    d1 = floor(log2(E1));
    c1 = 2^(log2(E1) - d1);

    fprintf('h=%0.2f x 2^(%0.2d)\t\tE0=%0.4f x 2^(%0.2d)\t\tE1=%0.4f x 2^(%0.2d)\n',1,log2(h),c0,d0,c1,d1);
end
fprintf('\n\n');
            
