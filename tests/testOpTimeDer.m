
clear all;
%% double precision CPU
nTheta = 10; nt = 8; h = rand(1); beta = rand(1);
th0    = randn(nTheta,nt);
L      = beta*kron(spdiags(ones(nt,1)*[-1 1],0:1,nt-1,nt)/h,speye(nTheta));
Lop    = opTimeDer(nTheta*nt,nt,h);
Lop.beta = beta;

t1 = L*th0(:);
t2 = Lop*th0;
assert(norm(t1-t2)/norm(t1)<1e-14,'L*x does not compute time derivative');

ok = checkAdjoint(Lop);
assert(ok,'adjoint test failed');

A   = L'*L;
eA  = eig(full(A));
A   = A;
rhs = randn(size(A,2),1);
t1  = pinv(full(A))*rhs;
t2  = PCmv(Lop,rhs);
assert(norm(t1-t2(:))/norm(t1)<1e-13,'preconditioner not working');
% solve proximal problem: min_x .5*a|L*x|^2 + .5*|x-y|^2
y = randn(size(L,2),1); y(2:6) = 5;
a = rand(1);

t1 = (a*full(A) + eye(size(L,2))) \ y;
t2 = PCmv(Lop,y,a,1);
norm(t1-t2)/norm(t1)
%% single precision CPU
nTheta = 10; nt = 6; h = rand(1); beta = rand(1);
th0    = randn(nTheta,nt);

L      = beta*kron(spdiags(ones(nt,1)*[-1 1],0:1,nt-1,nt)/h,speye(nTheta));
Lop    = opTimeDer(nTheta*nt,nt,h);
Lop.beta = beta;

Lop.precision = 'single';

t1 = L*th0(:);
t2 = Lop*single(th0);
assert(norm(t1-t2)/norm(t1)<1e-7,'L*x does not compute time derivative');

ok = checkAdjoint(Lop);
assert(ok,'adjoint test failed');

A   = L'*L;
rhs = randn(size(A,2),1);
t1  = pinv(full(A))*rhs;
t2  = PCmv(Lop,single(rhs));
assert(norm(t1-t2(:))/norm(t1)<1e-4,'preconditioner not working');

% solve proximal problem: min_x .5*a|L*x|^2 + .5*|x-y|^2
y = randn(size(L,2),1); y(2:6) = 5;
a = rand(1);

t1 = (a*full(A) + eye(size(L,2))) \ y;
t2 = PCmv(Lop,y,a,1);
norm(t1-t2)/norm(t1)
%% single precision GPU
nTheta = 10; nt = 6; h = rand(1);
th0    = randn(nTheta,nt);

L      = kron(spdiags(ones(nt,1)*[-1 1],0:1,nt-1,nt)/h,speye(nTheta));
Lop    = opTimeDer(nTheta*nt,nt,h);

Lop.useGPU    = 1;

t1 = L*th0(:);
t2 = Lop*gpuVar(Lop.useGPU,Lop.precision,th0);
assert(norm(t1-t2)/norm(t1)<1e-7,'L*x does not compute time derivative');

ok = checkAdjoint(Lop);
assert(ok,'adjoint test failed');

A   = L'*L;
eA  = eig(full(A));
A   = A + .5*min(eA(eA>1e-6))*speye(size(L,2));
rhs = randn(size(A,2),1);
t1  = pinv(full(A))*rhs;
t2  = PCmv(Lop,single(rhs));
assert(norm(t1-t2(:))/norm(t1)<1e-4,'preconditioner not working');

%% double precision GPU
nTheta = 10; nt = 6; h = rand(1);
th0    = randn(nTheta,nt);

L      = kron(spdiags(ones(nt,1)*[-1 1],0:1,nt-1,nt)/h,speye(nTheta));
Lop    = opTimeDer(nTheta*nt,nt,h);

Lop.precision = 'double';

t1 = L*th0(:);
t2 = Lop*gpuVar(Lop.useGPU,Lop.precision,th0);
assert(norm(t1-t2)/norm(t1)<1e-14,'L*x does not compute time derivative');

ok = checkAdjoint(Lop);
assert(ok,'adjoint test failed');

A   = L'*L;
eA  = eig(full(A));
A   = A + .5*min(eA(eA>1e-6))*speye(size(L,2));
rhs = randn(size(A,2),1);
t1  = pinv(full(A))*rhs;
t2  = PCmv(Lop,single(rhs));
assert(norm(t1-t2(:))/norm(t1)<1e-7,'preconditioner not working');
