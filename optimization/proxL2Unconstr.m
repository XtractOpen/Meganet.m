function xp = proxL2Unconstr(A,p,alpha,tau,xref)
% evaluates proximal operator
%
%  xp =  argmin_x alpha/2*|A*(x-xref)|^2 + tau/2*|x-p|^2
%     =  argmin_x (alpha/(2*tau))|A*(x-xref)|^2 + 1/2*|x-p|^2
%
%  The closed-form solution is
%
%  xp = ((alpha/tau) A'*A + I) \ (p + alpha/tau*A'*xref) 
%
if nargin==0
    runMinimalExample;
    return;
end
if exist('xref','var') && not(isempty(xref))
    p = p + (alpha/tau)*(A'*xref);
end
xp = PCmv(A,p,alpha/tau,1);

function runMinimalExample
n = 10;
d = ones(n,1)+1;
D = diag(d);
p = rand(n,1);
alpha = rand();
tau = rand();
xref = ones(n,1);
xp = ((alpha/tau)*(D'*D)+eye(n))\(p);

xpt = feval(mfilename,opDiag(d),p,alpha,tau,xref);
norm(xp-xpt)


