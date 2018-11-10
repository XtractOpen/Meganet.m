function [Ytrain,Ctrain,Ytest,Ctest] = setupCircle(nTrain,nTest)
%[Y0,C,Ytest,Ctest] = setupPeaks(np)

if nargin==0
    runMinimalExample;
    return
end

if not(exist('nTrain','var')) || isempty(nTrain)
    nTrain = 1000;
end

if not(exist('nTest','var')) || isempty(nTest)
    nTest = 200;
end


% get training data
rb     = rand(nTrain,1);
thetab = rand(nTrain,1)*2*pi; 
xb1 = rb .* cos(thetab);
xb2 = .4*rb .* sin(thetab);
rr     = 1+rand(nTrain,1);
thetar = rand(nTrain,1)*2*pi; 
xr1 = rr .* cos(thetar);
xr2 = .4*rr .* sin(thetar);
Ytrain = [[xb1; xr1],[xb2; xr2]];
Ctrain  = [ [ones(nTrain,1); zeros(nTrain,1)], [zeros(nTrain,1); ones(nTrain,1)]];

% get test data
rb     = rand(nTest,1);
thetab = rand(nTest,1)*2*pi; 
xb1 = rb .* cos(thetab);
xb2 = .4*rb .* sin(thetab);
rr     = 1+rand(nTest,1);
thetar = rand(nTest,1)*2*pi; 
xr1 = rr .* cos(thetar);
xr2 = .4*rr .* sin(thetar);
Ytest = [[xb1; xr1],[xb2; xr2]];
Ctest = [ [ones(nTest,1); zeros(nTest,1)], [zeros(nTest,1); ones(nTest,1)]];

Ytrain = Ytrain';
Ctrain  = Ctrain';
Ytest = Ytest';
Ctest = Ctest';

function runMinimalExample
[Ytrain,Ctrain,Ytest,Ctest] = feval(mfilename,200,100);
figure(1); clf;
subplot(1,2,1);
viewFeatures2D(Ytrain,Ctrain);
title('training data')
axis equal tight
subplot(1,2,2);
viewFeatures2D(Ytest,Ctest);
title('test data');
axis equal tight

