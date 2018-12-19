function convOp = getConvOp(useGPU)
% function convOp = getConvOp(useGPU)
%
% helper to find the most efficient convolution operator available on your
% system.
%
% Input:
%  useGPU   - flag for GPU computing
%
% Output:
%  convOp   - either convCuDNN2D, convMCN, convFFT depending on useGPU and
%             whether or not required binaries are available

if not(exist('useGPU','var')) || isempty(useGPU)
    useGPU = 0;
end

binExist = @(fname) exist([fname '.' mexext],'file');

if useGPU && binExist('convCuDNN2DSessionCreate_mex') && ...
             binExist('convCuDNN2DSessionDestroy_mex') &&...
             binExist('convCuDNN2D_mex')
    % use CUDNN wrapper which is most efficient for GPU computing
    cudnnSession = convCuDNN2DSession();
    convOp = @(varargin)convCuDNN2D(cudnnSession,varargin{:});
elseif binExist('vl_nnconv') && binExist('vl_nnconvt')
    % use binaries from MatConvNet
    convOp = @convMCN;
else
    % use plain MATLAB implementation
    convOp    = @convFFT;
end
