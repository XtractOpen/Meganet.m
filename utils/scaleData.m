function Y = scaleData(Y,range,numelFeat)
% Y = scaleData(Y,range)
%
% scales data such that range(1) <= Y <= range(2)
%
% Input:
%   Y         - features
%   range     - vector, defines intensity range
%   numelFeat - number of features per example. Optional argument. If
%               provided each example is scales separately
% 
% Output:
%    Y        - scaled features

if nargin==0
    runMinimalExample;
    return
end

if not(exist('numelFeat','var')) || isempty(numelFeat)
    % same shift and scaling for all examples
    Y = (range(2)-range(1))*Y/max(abs(Y(:)));
    Y = Y + range(1);
else
    szY = size(Y);
    Y   = reshape(Y,numelFeat,[]);
    Y   = (range(2)-range(1))*(Y./max(abs(Y),[],1));
    Y   = Y + range(1);
    Y   = reshape(Y,szY);
end   

function runMinimalExample
[Y,C] = setupCIFAR10(50);
figure(1);clf;
subplot(2,1,1);
montageArray(Y,10);
axis equal tight
colormap(flipud(colormap('gray')))
colorbar
title('original images');

[Y] = feval(mfilename,Y,[-0.3 2],32*32*3);

subplot(2,1,2);
montageArray(Y,10);
axis equal tight
colormap(flipud(colormap('gray')))
colorbar
title('scaled images');




