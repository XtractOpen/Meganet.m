function eigL = eigLap1D(n,h)
% function eigL = eigLap1D(n,h)
%
% computes eigenvalues of 1D laplacian with Neuman bc
%
% Input:
%
%   n - number of discretization point
%   h - discretization size
%
% Output:
%
%   e - eigenvalues of L'*L

if nargin==0
    help(mfilename);
    runMinimalExample;
    return;
end

dx = spdiags(ones(n,1)*[-1 1],[0 1],n-1,n);
d2x = dx'*dx;

% cosine transform preparation
Cx = dct(eye(n,1));
CLx = dct(full(d2x(:,1)));

llx = CLx./Cx;
eigL = llx/h.^2;


function runMinimalExample
n = 8; h = rand(1);
L = spdiags(ones(n,1)*[-1 1],[0 1],n-1,n)/h;
e = eig(L'*L);
et = sort(feval(mfilename,n,h));
norm(e-et)/norm(e)
