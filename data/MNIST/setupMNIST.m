function [Ytrain,Ctrain,Yval,Cval] = setupMNIST(nTrain,nVal)
% Output:
%     Ytrain - nTrain 28x28 training images in tensor (28,28,nTrain)
%     Yval   - nVal 28x28 training images in tensor (28,28,nVal)
%     Ctrain - corresponding training classes (10, nTrain)
%     Cval   - corresponding validation classes (10, nVal)
%

if nargin==0
    runMinimalExample;
    return;
end

if not(exist('nTrain','var')) || isempty(nTrain)
    nTrain = 50000;
end
if not(exist('nVal','var')) || isempty(nVal)
    nVal = round(nTrain/5);
end

images = reshape ( loadMNISTImages('train-images.idx3-ubyte') , 28,28,1,[] );
labels = loadMNISTLabels('train-labels.idx1-ubyte');

% get class probability matrix
C      = zeros(10,numel(labels));
ind    = sub2ind(size(C),labels+1,(1:numel(labels))');
C(ind) = 1;

idx = randperm(size(C,2));

idTrain = idx(1:nTrain);
idVal   = idx(nTrain+(1:nVal));

% Scale images between [-0.5 0.5]
Ytrain = images(:,:,:,idTrain);
Ctrain = C(:,idTrain);
Ytrain = Ytrain/max(abs(Ytrain(:))) - 0.5;
[~,k] = sort((1:10)*Ctrain);
Ytrain = Ytrain(:,:,:,k);
Ctrain = Ctrain(:,k);

if nargout>2
    Yval = images(:,:,:,idVal);
    Cval = C(:,idVal);
    Yval = Yval/max(abs(Yval(:))) - 0.5;
    [~,k] = sort((1:10)*Cval);
    Yval = Yval(:,:,:,k);
    Cval = Cval(:,k);
end

function runMinimalExample
[Yt,Ct,Yv,Cv] = feval(mfilename,50,10);
figure(1);clf;
subplot(2,1,1);
montageArray(Yt,10);
axis equal tight
colormap(flipud(colormap('gray')))
title('training images');


subplot(2,1,2);
montageArray(Yv,10);
axis equal tight
colormap(flipud(colormap('gray')))
title('validation images');




