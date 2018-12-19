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

%% tensor tests
A = randn(4,6,3); B = randn(4,6,3); C = randn(5,2,7);

Aop = LinearOperator([4,6],3, @(x) reshape( reshape(A,4*6,3)*x, [4 6 size(x,2)]) , @(x) reshape(reshape(A,4*6,3)'*x, [3 size(x,2)]) );
Bop = LinearOperator([4,6],3, @(x) reshape( reshape(B,4*6,3)*x, [4 6 size(x,2)]) , @(x) reshape(reshape(B,4*6,3)'*x, [3 size(x,2)]) );
Cop = LinearOperator([5,2],7, @(x) reshape( reshape(C,5*2,7)*x, [5 2 size(x,2)]) , @(x) reshape(reshape(C,5*2,7)'*x, [7 size(x,2)]) );

x = randn(3,6);
t1 = reshape( reshape(A,[],3)*x , [size(A,1) size(A,2) size(x,2)]);
t2 = Aop*x;
t3 = Bop*x;
t4 = Aop*x + Bop*x;
t5 = reshape( reshape(A+B,[],3)*x , [size(A,1) size(A,2) size(x,2)]); % (A+B)x
assert(norm(vec(t1-t2))/norm(vec(t1))<1e-14,'A*x ~= LinearOperator(A)*x');
assert(norm(vec(t2+t3-t4))/norm(vec(t4))<1e-14,'A*x ~= LinearOperator(A)*x');
assert(norm(vec(t4-t5))/norm(vec(t4))<1e-14,'A*x ~= LinearOperator(A)*x');



