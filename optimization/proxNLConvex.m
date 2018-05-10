function xp = proxNLConvex(Reg,p,tau,P,varargin)
% evaluates proximal operator
%
%  xp =  argmin_{x in C} R(x) + tau/2*|x-p|^2
%
%  where R is some convex function, C is a convex set and P is a projector onto that set.
%
% Since there is no closed-form solution we use a few steps of projected
% gradient descent.

if nargin==0
    runMinimalExample;
    return;
end
maxIter = 1000;

opt = sd();
opt.out = -1;
opt.LS.P = P;
opt.maxIter = maxIter;
opt.atol=1e-8;
opt.rtol=1e-8;

fctn = @(x) objFun(x,Reg,p,tau);
xp = solve(opt,fctn,p);
if not(isempty(Reg.xref))
    xt = solve(opt,fctn,Reg.xref);
    if fctn(xt)<fctn(xp)
        xp = xt;
    end
end


function [Jc,para,dJ] = objFun(x,Reg,p,tau)
% computes objective function so that SD can handle it
para = [];

[Rc,~,dR] = regularizer(Reg,x);

resD = x-p;
Dc   = 0.5*(resD'*resD);
dD   = resD;

Jc = tau*Dc+ Rc;
dJ = tau*dD+ dR;

function runMinimalExample
n = 2;
D = opDiag(rand(n,1));
wL1 = l1Reg(D,[1;.001]);

P =@(x) max(min(x,1),-1);
tau = rand();
p =[ 3;4];
xpt = feval(mfilename,wL1,p,tau,P)
xp = [1;1]
xp-xpt
norm(xp-xpt)


