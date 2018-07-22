function[R,P] = avgRestrictionGalerkin(n)
% [R,P] = avgRestrictionGalerkin(n)
%
% Constructing the restriction operator, R, and prolongation operator, P,
% for 2D images. Here, P is chosen as 4*R' as done in Galerkin projection
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

% restriction, 2D averaging matrix
R1 = spdiags(ones(n1,1)*[1/2 1/2],0:1,n1-1,n1);
R1 = R1(1:2:end,:);
R2 = spdiags(ones(n2,1)*[1/2 1/2],0:1,n2-1,n2);
R2 = R2(1:2:end,:);

R = kron(R2,R1);

P = 4*R';

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


