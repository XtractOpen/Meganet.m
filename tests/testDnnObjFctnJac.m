% test file for dnnObjFctnJac
close all; clear all;

% rng(42);
[Ytrain,Ctrain] = setupPeaks(100,5);
dynamic = 'ResNN';

%% setup network
T  = 5;   % final time
nt = 8;  % number of time steps
h = T/nt;
nc = 16;   % number of channels (width)


% first block (single layer that opens up)
block1 = NN({singleLayer(dense([nc,2]))},'activation',@tanhActivation);

% second block (ResNN, keeps size fixed)
switch dynamic
    case 'ResNN'
        K      = dense([nc,nc]);
        layer  = singleLayer(K,'Bout',ones(nc,1),'activation',@tanhActivation);
        block2 = ResNN(layer,nt,T/nt);
    case 'antiSym-ResNN'
        K      = getDenseAntiSym([nc,nc]);
        layer  = singleLayer(K,'Bout',ones(nc,1),'activation',@identityActivation);
        tY      = linspace(0,T,nt);
        block2  = ResNNrk4(layer,tY,tY);
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

Kout = dense([size(Ctrain,1),nc]);
block3 = NN({singleLayer(Kout, 'activation',@identityActivation,'Bin', ones(size(Ctrain,1),1))});

% combine both blocks
net = Meganet({block1,block2,block3});
%% regularization
alpha = 0;
pReg = tikhonovReg(opEye(nTheta(net)),alpha);

%% setup classification and Newton solver for this subproblem
pLoss = softmaxLoss();
% pLoss = regressionLoss('Gamma',eye(size(Ctrain,1)));
%% setup outer optimization scheme
fctn1 = dnnObjFctn2(net,pReg,pLoss,Ytrain,Ctrain);
fctn2 = dnnObjFctn2Jac(net,pLoss,Ytrain,Ctrain,'alpha',alpha,'dropTol',0);

%% compare function values
theta = initTheta(net);
[J1,p1,dJ1, d2F1] = eval(fctn1,theta);
[J2,p2,dJ2, d2F2] = eval(fctn2,theta);

%%
assert(norm(J1-J2)/norm(J1) < 1e-10,'function values do not agree');
assert(norm(dJ1-dJ2)/norm(J1) < 1e-10,'gradients do not agree');
v = randn(size(theta),'like',theta);
mv1 = d2F1*v;
mv2 = d2F2*v;
assert(norm(mv1-mv2)/norm(mv1) < 1e-10,'Hessian mv do not agree');

