function Y = randomRotate(Y,crop)

if nargin==0
    runMinimalExample
    return
end
nY     = size(Y);
Yp = padarray(Y,[crop/2 crop/2 0 0],0,'both');

shift = round(crop*rand(nY(4),2));
% box   = [shift nY(1:2)+shift];


for k=1:nY(4)
   Y(:,:,:,k) = Yp(shift(k,1)+(1:nY(1)),shift(k,2)+(1:nY(2)),:,k);
end


function runMinimalExample
Y = setupCIFAR10(10);
figure(1);clf;
subplot(2,1,1);
montageArray(Y(:,:,1,:),10);
axis equal tight
colormap(flipud(colormap('gray')))
colorbar
title('original images');

[Y] = feval(mfilename,Y,10);
subplot(2,1,2);
montageArray(Y(:,:,1,:),10);
axis equal tight
colormap(flipud(colormap('gray')))
colorbar
title('flipped images');
