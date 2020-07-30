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
C = randn(nTargets,nSamples);

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

pLoss = regressionLoss('addBias',1);

%% Initia lize weights and evaluate

theta = 0.5 * vec(randn(nTheta(net),1));
% W = 0.5 * vec(randn(nTargets * (sizeFeatOut(net)+1),1));

% evaluate
fctn = dnnVarProRegressionObjFctn(net,pLoss,Y,C,'alpha1',1e-2,'alpha2',1e-2);
[Jc,para,dJ,H,PC,res,J] = eval(fctn,theta(:));
WOpt = para.W;

%% Check adjoint
[OK,err] = checkAdjoint(J);
fprintf('Adjoint Test\n')
fprintf('Is OK? %d\t err = %0.2e\n',OK,err);

%% Check Jacobian
% 
fprintf('\n');
fprintf('Jacobian of W*Z(theta)\n')

R0 = vec(res);
dth = randn(size(theta));
dR  = vec((J * dth(:)));

for k = 1:15
    h = 2^(-k);
    tht = theta + h * dth;
    [~,~,~,~,~,Rt,~] = eval(fctn,tht(:));

    E0 = norm(R0 - Rt(:)) / norm(R0);
    E1 = norm(R0 + h * dR - Rt(:)) / norm(R0);

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
            
