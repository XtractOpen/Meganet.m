clear all; clc;


%% case 1: solve exactly
n = 100;
A = randn(n,n) + eye(n);
b = randn(n,1);

gmr = GMRES('tol',1e-8,'out',2,'m',n);


[x,para,z,V,H,~,flag] = gmr.solve(A,b);
xg = gmres(A,b,[],gmr.tol,gmr.m);
assert(norm(xg-x)/norm(xg) < 1e-12,'solution not equal to MATLAB gmres')
assert(norm(A*x-b)/norm(b) < gmr.tol,'tolerance not reached')

Vk = V(:,1:end-1);
assert(norm(Vk'*Vk-eye(gmr.m),2)/n < 1e-12,'check orthogonality of V')
assert(norm(A*V(:,1:end-1) - V*H,2)/norm(A*V(:,1:end-1),2) < 1e-12,'check A*Vk - Vk1*H')
assert(norm(Vk'*A*Vk-H(1:end-1,:),2)/norm(H(1:end-1,:),2) < 1e-12,'check H = Vk''*A*Vk')

%% case 2: stop early
n = 100;
A = randn(n,n) + eye(n);
b = randn(n,1);

gmr = GMRES('tol',1e-8,'out',2,'m',10);


[x,para,z,V,H,~,flag] = gmr.solve(A,b);
xg = gmres(A,b,[],gmr.tol,gmr.m);
assert(norm(xg-x)/norm(xg) < 1e-12,'solution not equal to MATLAB gmres')
assert(norm(A*x-b)/norm(b) >= gmr.tol,'incorrect flag')

Vk = V(:,1:end-1);
assert(norm(Vk'*Vk-eye(gmr.m),2)/n < 1e-12,'check orthogonality of V')
assert(norm(A*V(:,1:end-1) - V*H,2)/norm(A*V(:,1:end-1),2) < 1e-12,'check A*Vk - Vk1*H')
assert(norm(Vk'*A*Vk-H(1:end-1,:),2)/norm(H(1:end-1,:),2) < 1e-12,'check H = Vk''*A*Vk')

%% case 3: block system, solve exactly
n = 100;
A = blkdiag(randn(n,n) + eye(n),speye(n));
b = [randn(n,1); zeros(n,1)];

gmr = GMRES('tol',1e-8,'out',2,'m',size(A,2));


[x,para,z,V,H,~,flag] = gmr.solve(A,b);
xg = gmres(A,b,[],gmr.tol,gmr.m);
assert(norm(xg-x)/norm(xg) < 1e-12,'solution not equal to MATLAB gmres')
assert(norm(A*x-b)/norm(b) < gmr.tol,'incorrect flag')

Vk = V(:,1:end-1);
assert(norm(Vk'*Vk-eye(size(Vk,2)),2)/n < 1e-12,'check orthogonality of V')
assert(norm(A*V(:,1:end-1) - V*H,2)/norm(A*V(:,1:end-1),2) < 1e-12,'check A*Vk - Vk1*H')
assert(norm(Vk'*A*Vk-H(1:end-1,:),2)/norm(H(1:end-1,:),2) < 1e-12,'check H = Vk''*A*Vk')

%% case 4: block system, solve inexactly
n = 100;
A = blkdiag(randn(n,n) + eye(n),speye(n));
b = [randn(n,1); zeros(n,1)];

gmr = GMRES('tol',1e-8,'out',2,'m',10);


[x,para,z,V,H,~,flag] = gmr.solve(A,b);
xg = gmres(A,b,[],gmr.tol,gmr.m);
assert(norm(xg-x)/norm(xg) < 1e-12,'solution not equal to MATLAB gmres')
assert(norm(A*x-b)/norm(b) >= gmr.tol,'incorrect flag')

Vk = V(:,1:end-1);
assert(norm(Vk'*Vk-eye(size(Vk,2)),2)/n < 1e-12,'check orthogonality of V')
assert(norm(A*V(:,1:end-1) - V*H,2)/norm(A*V(:,1:end-1),2) < 1e-12,'check A*Vk - Vk1*H')
assert(norm(Vk'*A*Vk-H(1:end-1,:),2)/norm(H(1:end-1,:),2) < 1e-12,'check H = Vk''*A*Vk')


%% case 5: rank-deficient system
% n = 100;
% A = randn(n,10) ;
% A = A*A';
% b = randn(n,1);
% 
% gmr = GMRES('tol',1e-8,'out',2,'m',n);
% 
% 
% [x,para,z,V,H,~,flag] = gmr.solve(A,b);
% %%
% % xg = gmres(A,b,[],gmr.tol,gmr.m);
% % assert(norm(xg-x)/norm(xg) < 1e-12,'solution not equal to MATLAB gmres')
% 
% assert(norm(A*x-b)/norm(b) >= gmr.tol,'incorrect flag')
% 
% Vk = V(:,1:end-1);
% assert(norm(Vk'*Vk-eye(size(Vk,2)),2)/n < 1e-12,'check orthogonality of V')
% assert(norm(A*V(:,1:end-1) - V*H,2)/norm(A*V(:,1:end-1),2) < 1e-12,'check A*Vk - Vk1*H')
% assert(norm(Vk'*A*Vk-H(1:end-1,:),2)/norm(H(1:end-1,:),2) < 1e-12,'check H = Vk''*A*Vk')
% 
% 
