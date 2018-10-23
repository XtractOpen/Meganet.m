clear all; clc;
%% double precision CPU
nImg  = [16 18]; h = rand(2,1); nChannels = 2; betaG = rand(1); 

% build grad
D1      = kron(speye(nImg(2)),spdiags(ones(nImg(1),1)*[-1 1],0:1,nImg(1)-1,nImg(1))/h(1));
D2      = kron(spdiags(ones(nImg(2),1)*[-1 1],0:1,nImg(2)-1,nImg(2))/h(2),speye(nImg(1)));
G       = betaG*kron(speye(nChannels),[D1;D2]);
Gop     = opGrad(nImg,nChannels,h);
Gop.beta = betaG;

% build time-der
nTh = 10; nt = 6; betaD = rand(1); ht = .2;
D      = betaD*kron(spdiags(ones(nt,1)*[-1 1],0:1,nt-1,nt)/ht,speye(nTh));
Dop    = opTimeDer(nTh*nt,nt,ht);
Dop.beta = betaD;


L = blkdiag(G,D);
Lop = blkdiag(Gop,Dop);
%
x0 = randn(size(L,2),1);

t1 = L*x0(:);
t2 = Lop*x0;
assert(not(isa(t2,'gpuArray')))
assert(isa(t2,'double'))
assert(norm(t1-t2)/norm(t1)<1e-14,'L*x does not compute time derivative');

ok = checkAdjoint(Lop);
assert(ok,'adjoint test failed');
A   = L'*L;
eA  = eig(full(A));
A   = A;% + .5*min(eA(eA>1e-3))*speye(size(L,2));

rhs = randn(size(A,2),1);
t1  = pinv(full(A))*rhs;
t2  = PCmv(Lop,rhs);
assert(norm(t1-t2(:))/norm(t1)<1e-10,'preconditioner not working');

% solve proximal problem: min_x .5*a|L*x|^2 + .5*|x-y|^2
y = randn(size(L,2),1); y(2:6) = 5;
a = rand(1);

t1 = (a*full(A) + eye(size(L,2))) \ y;
t2 = PCmv(Lop,y,a,1);
assert(norm(t1-t2)/norm(t1)<1e-14,'proximal problem not working')


