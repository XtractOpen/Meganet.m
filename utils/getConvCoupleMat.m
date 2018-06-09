function[A] = getConvCoupleMat(K,n)
%[A] = getConvCoupleMat(K,n)
%
% builds matrix for coupled convolution such that for features y, we have
%
%   A*y = convrCouple(K,y);
%
% Inputs:
%  K  - convolution kernel (4D-Array with size(K) = [m,m,n,n] with m the
%                           kernel size and n being the number of stencils)
%  n  - image size
%
% Output:
%  A  - sparse matrix

if nargin==0
    help(mfilename);
    runMinimalExample;
    return
end

n = n(1:2);
A = [];
for j=1:size(K,4)
    Aj = [];
    for i=1:size(K,3)
        Ki = K(:,:,i,j);
        Aj = [Aj getConvMat(Ki,[n 1])];
    end
    A = [A;Aj];
end

function runMinimalExample

K = randn(5,5,2,3);
n = [16 16 size(K,3)];

A = feval(mfilename,K,n);
fig = figure(99); clf;
fig.Name = sprintf('%s - Minimal Example',mfilename);
subplot(2,2,1);
spy(A)
subplot(2,2,2);
imagesc(full(A));
axis square
colorbar;

Y   = randn(n);
t1  = A*Y(:);
% t2  = convn(Y(:,:,1),K,'same');
Kt = convMCN(n(1:2),size(K));
Aop = getOp(Kt,K);
t2 = Aop*Y(:);
err = norm(t1(:)-t2(:))/norm(t2(:));
subplot(2,2,3);
montageArray(reshape(t1,[n(1:2) size(K,4)]));
subplot(2,2,4);
montageArray(reshape(t2,[n(1:2) size(K,4)]));

fprintf('%s - relative error (A(K)*y - convn(y,K)= %1.2e \t OK? %d\n',mfilename,err,err<1e-15);

