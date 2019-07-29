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

I = vec(1:prod(nK));
A = reshape(I,nK);
Q = sparse([I;I],[I;vec(A')],[ones(prod(nK),1);-ones(prod(nK),1)],prod(nK),prod(nK));
q = -0.0001*vec(eye(nK(1)));
% find columns that have nonzeros
jj = find(sum(abs(Q),1));
Q = Q(:,jj);

K = dense(nK,'useGPU',useGPU,'precision',precision,'Q',Q,'q',q);

function runMinimalExample
K  = getDenseAntiSym([3,3]);
th = initTheta(K);
A  = getOp(K,th)
