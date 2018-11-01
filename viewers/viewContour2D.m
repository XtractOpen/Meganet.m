function viewContour2D(domain,theta,W,net,pLoss)

if nargin==0
    runMinimalExample;
    return
end

if size(W,2)==1
    cmap = [ .92 .71 .63;.6 .77 .89]; % colors of MATLAB dots with alpha(.4)
else
    cmap = [ .6 .77 .89; .92 .71 .63; 0.969 .875 .651;  0.796 0.671 0.824;.785 .867 .675;]; % colors of MATLAB dots with alpha(.4)
end
N = 200;
xa = linspace(domain(1),domain(2),N);
ya = linspace(domain(3), domain(4), N);
[XX,YY] = meshgrid(xa,ya);

Y = [XX(:) YY(:)]';
if prod(sizeFeatIn(net)) > 2
    Y = [Y; zeros(prod(sizeFeatIn(net))-2,size(Y,2))]; % TODO
end
    
if not(isempty(theta))
    [Y,tmp] = forwardProp(net,theta,Y);
end
[Cp] = getLabels(pLoss,W,Y);
[C,ca] = contourf(xa,ya,reshape((1:size(Cp,1))*Cp,N,N));
cmap
colormap(cmap);
set(ca,'EdgeColor','none');

function runMinimalExample
domain = [-1 1 -1 1];
% theta = 0;
W      = [-1 1 0];
net   = ResNN(doubleSymLayer(dense([2,2])),20,.3);
theta = randn(nTheta(net),1);
pLoss = logRegressionLoss;
figure(1); clf;
subplot(1,2,1);
feval(mfilename,domain,[],W,net,pLoss)
title('no transform')
subplot(1,2,2);
feval(mfilename,domain,theta,W,net,pLoss)
title('with tranform');
