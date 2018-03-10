function xp = proxL2Unconstr(A,p,alpha,tau)
% evaluates proximal operator
%
%  xp =  argmin_x alpha/2*|A*x|^2 + tau/2*|x-p|^2
%     =  argmin_x (alpha/(2*tau))|A*x|^2 + 1/2*|x-p|^2
%
%  The closed-form solution is
%
%  xp = ((alpha/tau) A'*A + I) \ p 
%
if nargin==0
    runMinimalExample;
    return;
end
xp = PCmv(A,p,alpha/tau,1);

function runMinimalExample
n = 10;
d = rand(n,1)+1;
D = diag(d);
p = rand(n,1);
alpha = rand();
tau = rand();
xp = ((alpha/tau)*(D'*D)+eye(n))\(p);

xpt = feval(mfilename,opDiag(d),p,alpha,tau);
norm(xp-xpt)


