function poolOp = getPoolOp()
% function poolOp = getPoolOp()
%
% helper to find the most efficient pooling operator available on your
% system.
%
% No required input.
%  
% Output:
%  poolOp   - either opPoolMCN or opPool depending on whether or not 
%             required binaries are available


binExist = @(fname) exist([fname '.' mexext],'file');

if  binExist('vl_nnpool')
    % use binaries from MatConvNet
    poolOp = @opPoolMCN;
else
    % use plain MATLAB implementation
    poolOp    = @opPool;
end

