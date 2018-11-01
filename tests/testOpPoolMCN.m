clear all; clc;
nImg = [16 16 3];
np   = 2;


%% double precision CPU
A   = getAveragePooling(nImg,np);
Aop = opPoolMCN(nImg,2^np);
assert(checkAdjoint(Aop),'adjoint test')

v = randn(Aop.n);
t1= A*v(:);
t2 = Aop*v;
assert(norm(t1-t2(:))/norm(t2(:)) < 1e-15)


%% single precision CPU
A = single(full(getAveragePooling(nImg,np)));

v = single(randn(Aop.n));
t1= A*v(:);
t2 = Aop*v;
assert(norm(t1-t2(:))/norm(t2(:)) < 1e-7)


%% single precision GPU
A   = gpuArray(single(full(getAveragePooling(nImg,np))));

v   = gpuArray(single(randn(size(A,2))));
t1= A*v;
t2 = Aop*v;
assert(norm(t1-t2)/norm(t2) < 1e-7)
