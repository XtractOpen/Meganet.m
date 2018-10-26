function[ival,jval,Qs] = getSparseConvGenMat(sK,nImg,matype)
%[Qs] = getSparseConvGenMat(sK,nImg)
% Generate the matrix Qs such that
% A = sparse(I,J,Qs*theta) is a convolution matrix
%


K0 = reshape(1:prod(sK),sK);
switch matype
    case 'FC' % full convolutions
        A0 = getConvCoupleMat(K0,nImg);
    case 'DC' % diagonal convolution
        A0 = getConvDiagMat(K0,nImg);
    case '1D'
        A0 = get1DConvMat(K0,nImg);
end
    [ival,jval,V] = find(A0);
    
    [JQ,IQ] = sort(V);
    VQ = ones(numel(IQ),1);

    Qs = sparse(IQ,JQ,VQ,nnz(A0),prod(sK)); %%%TODO remove?

end

function[A] = getConvCoupleMat(K,n)
%[A] = getConvCoupleMat(K,n)
%
% builds matrix for coupled convolution such that for features y, we have
%

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
end

function[A] = getConvDiagMat(K,n)
%[A] = getConvDiagMat(K,n)
%
% builds matrix for coupled convolution such that for features y, we have
%

n = n(1:2);
A = [];

for i=1:size(K,3)
    Ki = K(:,:,i);
    A = blkdiag(A, getConvMat(Ki,[n 1]));
end
end

function[A] = get1DConvMat(K,n)
%[A] = get1DConvMat(K,n)
%
% builds matrix for coupled convolution such that for features y, we have
%

n = n(1:2);
A = [];

for j=1:size(K,2)
    Aj = [];
    for i=1:size(K,1)
        Aj = [Aj K(i,j)*speye(prod(n),prod(n))];
    end
    A = [A;Aj];
end
end

function[A] = getConvMat(K,n)
%[A] = getConvMat(K,n)
% builds convolution matrix A, such that for an image y it holds that
%

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
A = sparse(ii(ind), jj(ind),vv(ind),prod(n),prod(n));

end