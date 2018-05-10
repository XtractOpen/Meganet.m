function xp = proxL2Convex(A,p,alpha,tau,P,varargin)
% evaluates proximal operator
%
%  xp =  argmin_{x in C} alpha/2*|A*x|^2 + tau/2*|x-p|^2
%     =  argmin_{x in C} (alpha/(2*tau))|A*x|^2 + 1/2*|x-p|^2
%
%  where C is a convex set and P is a projector onto that set.
%
% Since there is no closed-form solution we use a few steps of projected
% gradient descent.

if nargin==0
    runMinimalExample;
    return;
end
maxIter = 100;

opt = sd();
opt.out = -1;
opt.LS.P = P;
opt.LS.maxIter = 20;
opt.maxIter = maxIter;
opt.atol=1e-8;
opt.rtol=1e-8;

fctn = @(x) LSobjFun(x,A,p,alpha,tau);
xp = solve(opt,fctn,p);
if not(isempty(Reg.xref))
    xt = solve(opt,fctn,Reg.xref);
    if fctn(xt)<fctn(xp)
        xp = xt;
    end
end

function [Jc,para,dJ] = LSobjFun(x,A,p,alpha,tau)
% computes objective function so that SD can handle it
para = [];
resR = A*x;
Rc   = 0.5*(resR'*resR);
dR   = A'*resR;

resD = x-p;
Dc   = 0.5*(resD'*resD);
dD   = resD;

Jc = Dc+ (alpha/tau)*Rc;
dJ = dD+ (alpha/tau)*dR;

function runMinimalExample
n = 10;
p = [2;2];
P =@(x) max(min(x,1),-1);
alpha = rand();
tau = rand();
xpt = feval(mfilename,opEye(n),p,alpha,tau,P)
xp = [1;1];
norm(xp-xpt)


