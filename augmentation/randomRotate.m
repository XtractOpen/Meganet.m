function Y = randomRotate(Y,angles)

if nargin==0
    runMinimalExample
    return
end
nY     = size(Y,4);
angles = abs(angles(2)-angles(1))*randn(nY,1)+angles(1);
for k=1:nY
   Y(:,:,:,k) = imrotate(Y(:,:,:,k),angles(k),'nearest','crop'); 
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

[Y] = feval(mfilename,Y,[-15,15]);
subplot(2,1,2);
montageArray(Y(:,:,1,:),10);
axis equal tight
colormap(flipud(colormap('gray')))
colorbar
title('flipped images');
