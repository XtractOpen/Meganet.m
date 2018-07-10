clear all; clc;
A = eye(4);
B = randn(5,3);

AB = kron(A,B);
opAB = opKron(4,B);


%% double precision CPU

v = randn(size(AB,2),1);
t1= AB*v;
t2 = opAB*v;
assert(norm(t1-t2)/norm(t2) < 1e-15)

v = randn(size(AB,1),1);
t1= AB'*v;
t2 = opAB'*v;
assert(norm(t1-t2)/norm(t2) < 1e-15)


