function[R,P] = cellCenteredInt(n)
% [R,P] = cellCenteredInt(n)
%
% Constructing the restriction operator, R, and prolongation operator, P,
% for 2D images. 
%
% Input:
%
%   n=[n1,n2] - number of pixels in fine images
%
% Outputs:
%
%   R         - restriction operator, size(R)=[prod(n)/4, prod(n)]
%   P         - prolongation operator, size(P)=[prod(n) prod(n)/4]

if nargin == 0
    [R,P] = runMinimalExample;
    return
end

n1 = n(1); n2 = n(2);

% restriction
R1 = spdiags(ones(n1,1)*[1/2 1/2],0:1,n1-1,n1);
R1 = R1(1:2:end,:);
R2 = spdiags(ones(n2,1)*[1/2 1/2],0:1,n2-1,n2);
R2 = R2(1:2:end,:);

R = kron(R2,R1);

% Prolongation
P1 = zeros(n1,n1/2);
for i=2:n1/2-1
    P1(2*i-2:2*i+1,i) = [1;3;3;1];
end
P1(1:3,1) = [4;3;1];
P1(end-2:end,end) = [1;3;4];
P1 = 1/4*sparse(P1);

P2 = zeros(n2,n2/2);
for i=2:n2/2-1
    P2(2*i-2:2*i+1,i) = [1;3;3;1];
end
P2(1:3,1) = [4;3;1];
P2(end-2:end,end) = [1;3;4];
P2 = 1/4*sparse(P2);

P = kron(P2,P1);

end

function[R,P] = runMinimalExample

n = [16,16];
[R,P] = cellCenteredInt(n);

figure(1); clf;
subplot(2,2,1); spy(P)
title('P, prolongation')
subplot(2,2,2); spy(R)
title('R, restriction')
% Check sparsity pattern of coarse scale Laplacian
A  = getConvMat(randn(3,3),[n(1),n(2),1]);
AH = R*A*P;
subplot(2,2,3); spy(A)
title('A, fine mesh convolution')
subplot(2,2,4); spy(AH)
title('AH = R*A*P')
fprintf('nnz per row for A = %3d  nnz per row for AH = %3d\n',nnz(A(34,:)),nnz(AH(19,:)))  

end

