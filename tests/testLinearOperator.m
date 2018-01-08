clear all;
A = randn(4,6); B = randn(4,6); C = randn(5,2);

Aop = LinearOperator(A);
Bop = LinearOperator(B);
Cop = LinearOperator(C);
Aop2 = LinearOperator(size(A,1),size(A,2),@(x) A*x, @(x) A'*x);

x = randn(6,3);
t1 = A*x;
t2 = Aop*x;
t3 = Aop2*x;
assert(norm(t1-t2)/norm(t1)<1e-14,'A*x ~= LinearOperator(A)*x');
assert(norm(t1-t3)/norm(t1)<1e-14,'A*x ~= LinearOperator(A)*x');

x = randn(4,2);
t1 = A'*x;
t2 = Aop'*x;
t3 = Aop2'*x;
assert(norm(t1-t2)/norm(t1)<1e-14,'A''*x ~= LinearOperator(A)''*x');
assert(norm(t1-t3)/norm(t1)<1e-14,'A''*x ~= LinearOperator(A)''*x');

x = randn(6,3);
t1 = (A +B)*x;
t2 = (Aop+Bop)*x;
t3 = (Aop+Bop)*x;
assert(norm(t1-t2)/norm(t1)<1e-14,'(A+B)*x ~= (LinearOperator(A)+LineaOperator(B))*x');
assert(norm(t1-t3)/norm(t1)<1e-14,'(A+B)*x ~= (LinearOperator(A)+LineaOperator(B))*x');


x = randn(4,2);
t1 = (A' +B')*x;
t2 = (Aop+Bop)'*x;
t3 = (Aop'+Bop')*x;
assert(norm(t1-t2)/norm(t1)<1e-14,'(A+B)''*x ~= (LinearOperator(A)+LineaOperator(B))''*x');
assert(norm(t1-t3)/norm(t1)<1e-14,'(A+B)''*x ~= (LinearOperator(A)+LineaOperator(B))''*x');


x = randn(6,3);
t1 = (A -B)*x;
t2 = (Aop-Bop)*x;
t3 = (Aop-Bop)*x;
assert(norm(t1-t2)/norm(t1)<1e-14,'(A-B)*x ~= (LinearOperator(A)-LineaOperator(B))*x');
assert(norm(t1-t3)/norm(t1)<1e-14,'(A-B)*x ~= (LinearOperator(A)-LineaOperator(B))*x');



x = randn(4,2);
t1 = (A' -B')*x;
t2 = (Aop-Bop)'*x;
t3 = (Aop'-Bop')*x;
assert(norm(t1-t2)/norm(t1)<1e-14,'(A-B)''*x ~= (LinearOperator(A)-LineaOperator(B))''*x');
assert(norm(t1-t3)/norm(t1)<1e-14,'(A-B)''*x ~= (LinearOperator(A)-LineaOperator(B))''*x');


dAB   = blkdiag(A,B,C);
dABop = blkdiag(Aop,Bop,Cop);
x = randn(size(dAB,2),4);
t1 = dAB*x;
t2 = dABop*x;
assert(norm(t1-t2)/norm(t1)<1e-14,'blkdiag(A,B,C)*x error');


dAB   = blkdiag(A',B',C');
dABop = blkdiag(Aop,Bop,Cop);
x = randn(size(dAB,2),4);
t1 = dAB*x;
t2 = dABop'*x;
assert(norm(t1-t2)/norm(t1)<1e-14,'blkdiag(A,B,C)''*x error');


