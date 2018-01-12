function K = getDenseAntiSym(nK,useGPU,precision)

if nargin==0
    runMinimalExample;
    return
end

if nK(1)~=nK(2)
    error('antisymmetric kernel must be square');
end

if not(exist('precision','var')) || isempty(precision)
    precision = 'double';
end
if not(exist('useGPU','var')) || isempty(useGPU)
    useGPU = 0;
end

A = reshape(1:prod(nK),nK);
QAt = sparse(vec(A'),1:prod(nK),ones(prod(nK),1),prod(nK),prod(nK));
Q = speye(prod(nK)) - QAt;

Q = Q(:,sum(abs(Q),1)>0);
K = dense(nK,'useGPU',useGPU,'precision',precision,'Q',Q);

function runMinimalExample
th = randn(9-3,1);
K = getDenseAntiSym([3,3]);
A = getOp(K,th)
