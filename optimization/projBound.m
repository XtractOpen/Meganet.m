% function px = projBound(xc,xlow,xhigh)
%
% projection onto box defined by xlow and xhigh
%
function px = projBound(xc,xlow,xhigh)

if nargin==0
    runMinimalExample;
    return
end

if isscalar(xlow);
    xlow = xlow*ones(size(xc),'like',xc);
end

if isscalar(xhigh)
    xhigh = xhigh*ones(size(xc),'like',xc);
end

px = min(max(xc,xlow),xhigh);


function runMinimalExample
nx = 200;
x = randn(nx,1);

px = feval(mfilename,x,0,Inf)

figure(1); clf;
subplot(2,2,1);
x = reshape(x,[],2);
plot(x(:,1),x(:,2),'or');

subplot(2,2,2);
px = reshape(px,[],2);
plot(px(:,1),px(:,2),'or');

px = feval(mfilename,vec(x),0,1);
subplot(2,2,3);
px = reshape(px,[],2);
plot(px(:,1),px(:,2),'or');


px = feval(mfilename,vec(x),[0*ones(nx/2,1);-Inf*ones(nx/2,1)],[10*ones(nx/2,1);Inf*ones(nx/2,1)]);
subplot(2,2,4);
px = reshape(px,[],2);
plot(px(:,1),px(:,2),'or');
