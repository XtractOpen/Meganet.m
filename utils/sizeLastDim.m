function lastDim = sizeLastDim(A)
% equivalent to    size(A, ndims(A))
% if A is a matrix, size(A,2)

lastDim = size(A, ndims(A));

