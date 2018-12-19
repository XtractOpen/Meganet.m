function [isOK, err] = checkAdjoint(A,useGPU,precision)
% function [isOK, err] = checkAdjoint(A)
% 
% performs adjoint check for linear operator or matrix (boring here)

if not(isa(A,'LinearOperator')) && not(isnumeric(A))
    error('%s - input must be LinearOperator or matrix',mfilename)
end

if not(exist('useGPU','var')) || isempty(useGPU)
    useGPU = 0;
end
if not(exist('precision','var')) || isempty(precision)
    precision = 'double';
end

if isa(A,'LinearOperator')
    v = randn([A.n,1]);
else
    v = randn(size(A,2),1);
end
[v] = gpuVar(useGPU,precision,v);

Av = A*v;
w  = randn(size(Av),'like',Av);

Aw = vec(A'*w); 
t1 = vec(v)'*Aw;
t2 = vec(w)'*vec(Av);
err = abs(t1-t2)/(abs(t1)+(t1==0));

if isa(t1,'double') && isa(t2,'double')
    isOK = err<1e-10;
else
    isOK = err<1e-5;
end


