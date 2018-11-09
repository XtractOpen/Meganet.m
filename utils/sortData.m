function[Y,C] = sortData(Y,C)
%[Y,C] = sortData(Y,C)
%
% sorts examples by class
%
% Input:
%    Y  - features
%    C  - labels
% 
% Output:
%    Y  - sorted features
%    C  - sorted labels

if nargin==0
    runMinimalExample;
    return
end
[nc,nex] = size(C);

szY = size(Y);
Y   = reshape(Y,[],nex);

[~,id] = sort((1:nc)*C);
C      = C(:,id);
Y      = reshape(Y(:,id),szY);

function runMinimalExample
[Y,C] = setupMNIST(50);
figure(1);clf;
subplot(2,2,1);
montageArray(Y,10);
axis equal tight
colormap(flipud(colormap('gray')))
colorbar
title('unsorted images');

subplot(2,2,2);
imagesc(C)
axis equal tight
title('unsorted labels');

[Y,C] = feval(mfilename,Y,C);

subplot(2,2,3);
montageArray(Y,10);
axis equal tight
colormap(flipud(colormap('gray')))
colorbar
title('sorted images');

subplot(2,2,4);
imagesc(C)
axis equal tight
title('sorted labels');



