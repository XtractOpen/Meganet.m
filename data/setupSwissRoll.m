function[Ytrain,Ctrain,Ytest,Ctest] = setupSwissRoll(nTrain, nTest)

if nargin==0 && nargout==0
    runMinimalExample
    return
end

if not(exist('nTrain','var')) || isempty(nTrain)
    nTrain = 1000;
end

if not(exist('nTest','var')) || isempty(nTest)
    nTest = 200;
end

thetaBlue = 4*pi*rand((nTrain+nTest)/2,1);
thetaBlue = thetaBlue(:); r = thetaBlue(:)/(4*pi);
xb = r.*cos(thetaBlue); yb = r.*sin(thetaBlue);

thetaRed = 4*pi*rand((nTrain+nTest)/2,1);
r = 0.2+thetaRed(:)/(4*pi); 
r = r(:);
xr = r.*cos(thetaRed); yr = r.*sin(thetaRed);

id      = randperm(nTrain+nTest);
idTrain = id(1:nTrain);
idTest  = id(nTrain+1:end);

Y      = [xb', xr'; yb', yr'];
C      = [1+0*xb',   0*xr'; 0*xb',  1+0*xr'];
Ytrain = Y(:,idTrain);
Ctrain = C(:,idTrain);
Ytest  = Y(:,idTest);
Ctest  = C(:,idTest);


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
