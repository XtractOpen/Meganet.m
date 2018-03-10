clear all; clc;
%% double precision CPU
n  = [16 18]; h = rand(2,1); nChannels = 2; beta = rand(1); 
x0 = randn([n nChannels]);

D1      = kron(speye(n(2)),spdiags(ones(n(1),1)*[-1 1],0:1,n(1)-1,n(1))/h(1));
D2      = kron(spdiags(ones(n(2),1)*[-1 1],0:1,n(2)-1,n(2))/h(2),speye(n(1)));
L       = beta*kron(speye(nChannels),[D1;D2]);
Lop     = opGrad(n,nChannels,h);
Lop.beta = beta;

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

%% solve proximal problem: min_x .5*a|L*x|^2 + .5*|x-y|^2
y = randn(size(L,2),1); y(2:6) = 5;
a = rand(1);

t1 = (a*full(A) + eye(size(L,2))) \ y;
t2 = PCmv(Lop,y,a,1);
norm(t1-t2)/norm(t1)

%% single precision CPU
n  = [16 18]; h = rand(2,1); nChannels = 2;
x0 = randn([n nChannels]);

D1      = kron(speye(n(2)),spdiags(ones(n(1),1)*[-1 1],0:1,n(1)-1,n(1))/h(1));
D2      = kron(spdiags(ones(n(2),1)*[-1 1],0:1,n(2)-1,n(2))/h(2),speye(n(1)));
L       = kron(speye(nChannels),[D1;D2]);
Lop     = opGrad(n,nChannels,h);
Lop.precision = 'single';

t1 = L*x0(:);
t2 = Lop*gpuVar(Lop.useGPU,Lop.precision,x0);
assert(not(isa(t2,'gpuArray')))
assert(isa(t2,'single'))

assert(norm(t1-t2)/norm(t1)<1e-7,'L*x does not compute time derivative');

ok = checkAdjoint(Lop);
assert(ok,'adjoint test failed');
A   = L'*L;
eA  = eig(full(A));
A   = A;% + .5*min(eA(eA>1e-3))*speye(size(L,2));
rhs = randn(size(A,2),1);
t1  = pinv(full(A))*rhs;
t2  = PCmv(Lop,rhs);
assert(norm(t1-t2(:))/norm(t1)<1e-5,'preconditioner not working');


%% single precision GPU
n  = [16 18]; h = rand(2,1); nChannels = 2;
x0 = randn([n nChannels]);

D1      = kron(speye(n(2)),spdiags(ones(n(1),1)*[-1 1],0:1,n(1)-1,n(1))/h(1));
D2      = kron(spdiags(ones(n(2),1)*[-1 1],0:1,n(2)-1,n(2))/h(2),speye(n(1)));
L       = kron(speye(nChannels),[D1;D2]);
Lop     = opGrad(n,nChannels,h);
Lop.precision = 'single';
Lop.useGPU = 1;
t1 = L*x0(:);
t2 = Lop*gpuVar(Lop.useGPU,Lop.precision,x0);
assert(isa(t2,'gpuArray'))
assert(isa(gather(t2),'single'))
assert(norm(t1-t2)/norm(t1)<1e-7,'L*x does not compute time derivative');

ok = checkAdjoint(Lop);
assert(ok,'adjoint test failed');
A   = L'*L;
eA  = eig(full(A));
A   = A;% + .5*min(eA(eA>1e-3))*speye(size(L,2));
rhs = randn(size(A,2),1);
t1  = pinv(full(A))*rhs;
t2  = PCmv(Lop,rhs);
assert(norm(t1-t2(:))/norm(t1)<1e-5,'preconditioner not working');

%% double precision GPU
n  = [16 18]; h = rand(2,1); nChannels = 2;
x0 = randn([n nChannels]);

D1      = kron(speye(n(2)),spdiags(ones(n(1),1)*[-1 1],0:1,n(1)-1,n(1))/h(1));
D2      = kron(spdiags(ones(n(2),1)*[-1 1],0:1,n(2)-1,n(2))/h(2),speye(n(1)));
L       = kron(speye(nChannels),[D1;D2]);
Lop     = opGrad(n,nChannels,h);
Lop.precision = 'double';
Lop.useGPU = 1;

t1 = L*x0(:);
t2 = Lop*gpuVar(Lop.useGPU,Lop.precision,x0);
assert(isa(t2,'gpuArray'))
assert(isa(gather(t2),'double'))
assert(norm(t1-t2)/norm(t1)<1e-7,'L*x does not compute time derivative');

ok = checkAdjoint(Lop);
assert(ok,'adjoint test failed');
A   = L'*L;
eA  = eig(full(A));
A   = A;% + .5*min(eA(eA>1e-3))*speye(size(L,2));
rhs = randn(size(A,2),1);
t1  = pinv(full(A))*rhs;
t2  = PCmv(Lop,rhs);
assert(norm(t1-t2(:))/norm(t1)<1e-5,'preconditioner not working');
