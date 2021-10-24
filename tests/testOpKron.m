clear all; clc;
A = rand(6,7);
B = randn(5,3);

AB = kron(A,B);
opAB = opKron(A,B);


%% double precision CPU
v = randn(size(AB,2),1);
t1= AB*v;
t2 = opAB*v;
assert(norm(t1-t2,1)/norm(t2,1) < 1e-15)

v = randn(size(AB,1),1);
t1= AB'*v;
t2 = opAB'*v;
assert(norm(t1-t2,1)/norm(t2,1) < 1e-15)

%% double precision CPU (multiple rhs)
nrhs = 4;
v = randn(size(AB,2),nrhs);
t1= AB*v;
t2 = opAB*v;
assert(norm(t1-t2,1)/norm(t2,1) < 1e-15)

v = randn(size(AB,1),nrhs);
t1= AB'*v;
t2 = opAB'*v;
assert(norm(t1-t2,1)/norm(t2,1) < 1e-15)


