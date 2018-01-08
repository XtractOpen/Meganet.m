function[A] = getConvMat(K,n)
%[A] = getConvMat(K,n)
%
% builds convolution matrix A, such that for an image y assuming periodic
% boundary conditions (pad images to make up for that)
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

jj = []; ii = []; vv = [];
[I1,I2,I3] = ndgrid(1:n(1),1:n(2),1:n(3));
for i=1:size(K,1)
    for j=1:size(K,2)
        for k=1:size(K,3)
            ofi = i - (size(K,1)-1)/2 - 1;
            ofj = j - (size(K,2)-1)/2 - 1;
            ofk = k - (size(K,3)-1)/2 - 1;
            
            jj = [jj; G(:)];
            t1 = I1+ofi; 
            t1(t1<=0) = n(1)+t1(t1<=0);
            t1(t1>n(1)) = t1(t1>n(1))-n(1);
            t2 = I2+ofj; 
            t2(t2<=0) = n(2)+t2(t2<=0);
            t2(t2>n(2)) = t2(t2>n(2))-n(2);
            t3 = I3+ofk; 
            t3(t3<=0) = n(3)+t3(t3<=0);
            t3(t3>n(3)) = t3(t3>n(3))-n(3);
            
            ii = [ii; vec(G(sub2ind(n,t1,t2,t3)))];
            vv = [vv; ones(prod(n),1)*K(i,j,k)];
        end
    end
end

ind = ii>0;
A = sparse(ii(ind), jj(ind),vv(ind),prod(n),prod(n));

function runMinimalExample

% K = -[1 2 1;2 4 4;1 2 1]/16;
K = randn(3,3);
n = [8 8 1];

A = feval(mfilename,K(:,:,1),[n(1:2)+2 1]);
fig = figure(99); clf;
fig.Name = sprintf('%s - Minimal Example',mfilename);
subplot(1,2,1);
spy(A)
subplot(1,2,2);
imagesc(full(A));
axis square
colorbar;

Y   = randn(n);
Yp  = zeros([n(1:2)+2 1]); Yp(2:end-1,2:end-1) = Y;
t1  = reshape(A*Yp(:),[n(1:2)+2 1]);
t1  = t1(2:end-1,2:end-1);
t2  = convn(Y(:,:,1),K,'same');
err = norm(t1(:)-t2(:))/norm(t2);
fprintf('%s - relative error (A(K)*y - convn(y,K)= %1.2e \t OK? %d\n',mfilename,err,err<1e-15);

