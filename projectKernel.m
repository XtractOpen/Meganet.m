function x = projectKernel(sz,x,nt,doNorm)

if not(exist('nt','var')) || isempty(nt)
    nt = 1;
end
if not(exist('doNorm','var')) || isempty(doNorm)
    doNorm = false;
end

% split into kernel and rest
x = reshape(x,[],nt);
xk = x(1:prod(sz),:);

xk = reshape(xk,prod(sz(1:2)),[]);
xk = xk - mean(xk,1);
if doNorm
    nx = max(1,sqrt(sum(xk.^2,1)));
    x(1:prod(sz),:) = reshape(xk./nx,[],nt);
else
    x(1:prod(sz),:) = reshape(xk,[],nt);
end
x = vec(x);