function [Y0,C,Ytest,Ctest] = setupMNIST(nex)
%[Y0,C,Ytest,Ctest] = setupPeaks(np)

if not(exist('nex','var')) || isempty(nex)
    nex = 5000;
end

baseDir = fileparts(which('Meganet.m'));
% mnistDir = fullfile(baseDir,'data','MNIST');

I      = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');

I  = I';
idv = (1:round(nex/5));
Iv = I(idv,:);   
labelsv = labels(idv);

idt  = idv(end)+1:min(idv(end)+nex,numel(labels));
I = I(idt,:);
labels = labels(idt);

[Y0,C]   = sortAndScaleMNISTData(I,labels);
[Ytest,Ctest] = sortAndScaleMNISTData(Iv,labelsv);


function[X,Y] = sortAndScaleMNISTData(X,labels)
%[X,Y] = sortAndScaleData(X,labels)
%

% Scale X [-0.5 0.5]
X  = X/max(abs(X(:))) - 0.5;

% Organize labels
[~,k] = sort(labels);
labels = labels(k);
X      = X(k,:);

Y = zeros(size(X,1),max(labels)-min(labels)+1);
for i=1:size(X,1)
    Y(i,labels(i)+1) = 1;
end

