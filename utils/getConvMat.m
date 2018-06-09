function[A] = getConvMat(K,n)
%[A] = getConvMat(K,n)
%
% builds convolution matrix A, such that for an image y it holds that
%
%   A*y = convn(flip(K),y,'same')
%
% Inputs
%   K               - convolution kernel
%   n = [nx,ny,nz]  - image size 
% 
% Outputs
%   A               - convolution matrix


if nargin==0
    help(mfilename);
    runMinimalExample;
    return 
end

vec = @(x)x(:);

G  = reshape(1:prod(n),n);
Gp = zeros(n+4);
Gp(3:end-2,3:end-2,3:end-2) = G;

jj = []; ii = []; vv = [];
[I1,I2,I3] = ndgrid(3:n(1)+2,3:n(2)+2,3:n(3)+2);
for i=1:size(K,1)
    for j=1:size(K,2)
        for k=1:size(K,3)
            ofi = i - (size(K,1)-1)/2 - 1;
            ofj = j - (size(K,2)-1)/2 - 1;
            ofk = k - (size(K,3)-1)/2 - 1;
            
            jj = [jj; G(:)];
            ii = [ii; vec(Gp(sub2ind(n+4,I1+ofi,I2+ofj,I3+ofk)))];
            vv = [vv; ones(prod(n),1)*K(i,j,k)];
        end
    end
end

ind = ii>0;
A = sparse(ii(ind), jj(ind),vv(ind),prod(n),prod(n))';

function runMinimalExample

% K = -[1 2 1;2 4 4;1 2 1]/16;
K = randn(5,5);
n = [16 16 1];

A = feval(mfilename,K(:,:,1),n);
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
Kt = convMCN(n(1:2),[5 5 1 1]);
Aop = getOp(Kt,K);
t2 = Aop*Y(:,:,1);
err = norm(t1(:)-t2(:))/norm(t2);
subplot(2,2,3);
imagesc(reshape(t1,n(1:2)));
subplot(2,2,4);
imagesc(reshape(t2,n(1:2)));

fprintf('%s - relative error (A(K)*y - convn(y,K)= %1.2e \t OK? %d\n',mfilename,err,err<1e-15);

