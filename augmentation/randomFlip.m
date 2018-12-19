function Y = randomFlip(Y,perc)

if nargin==0
    runMinimalExample
    return
end

nY = size(Y,4);
id = randperm(nY,fix(perc*nY));        
Y(:,:,:,id) = flip(Y(:,:,:,id),2);


function runMinimalExample
Y = setupCIFAR10(10);
figure(1);clf;
subplot(2,1,1);
montageArray(Y(:,:,1,:),10);
axis equal tight
colormap(flipud(colormap('gray')))
colorbar
title('original images');

[Y] = feval(mfilename,Y,1);

subplot(2,1,2);
montageArray(Y(:,:,1,:),10);
axis equal tight
colormap(flipud(colormap('gray')))
colorbar
title('flipped images');
