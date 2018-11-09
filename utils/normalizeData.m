function Y = normalizeData(Y,numelFeat)
% Y = normalizeData(Y,numelFeat)
%
% normalize mean and standard deviation as proposed in
% https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization
%
% Input:
%   Y         - features
%   numelFeat - number of features per example.
% 
% Output:
%    Y        - normalized features

if nargin==0
    runMinimalExample
    return
end
szY = size(Y);
Y   = reshape(Y,numelFeat,[]);

Y   = Y - mean(Y,1);

s   = std(Y,[],1);
Y   = Y./max(s,1/sqrt(numelFeat));
Y   = reshape(Y,szY);

function runMinimalExample
Y = setupCIFAR10(50);
figure(1);clf;
subplot(1,2,1);
montageArray(Y,10);
axis equal tight
colormap(flipud(colormap('gray')))
colorbar
title('original images');

[Y] = feval(mfilename,Y,32*32*3);

subplot(1,2,2);
montageArray(Y,10);
axis equal tight
colormap(flipud(colormap('gray')))
colorbar
title('normalized images');
