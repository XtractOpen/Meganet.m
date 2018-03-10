function x = projectKernel(sz,x,nt)

if not(exist('nt','var')) || isempty(nt)
    nt = 1;
end

% split into kernel and rest
x = reshape(x,[],nt);
xk = x(1:prod(sz),:);

xk = reshape(xk,prod(sz(1:2)),[]);
xk = xk - mean(xk,1);
nx = max(1,sqrt(sum(xk.^2,1)));
x(1:prod(sz),:) = reshape(xk./nx,[],nt);
x = vec(x);