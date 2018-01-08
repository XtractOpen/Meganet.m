function varargout = gpuVar(useGPU,precision,varargin)
% function varargout = gpuVar(useGPU,precision,varargin)
%
% brings variables to GPU and/or adjusts precision
%
% Input:
%
%  useGPU    - flag for using GPU
%  precision - flag for precision 'single' or 'double'
%  varargin  - variables to be considered
%
% Output:
%
%  varargout - transformed variables

if nargin==0
    help(mfilename);
    return;
end

nv = numel(varargin);
varargout = varargin;
for k=1:nv
    vark           = varargin{k};
    if not(isempty(precision)) && strcmp(precision,'single')
        if isnumeric(vark)
            vark = single(vark);
        else
            vark.precision = 'single';
        end
    end
    
    if not(isempty(useGPU)) && useGPU && not(isa(vark,'gpuArray'))
        if isnumeric(vark)
            vark = gpuArray(vark);
        else
            vark.useGPU = useGPU;
        end
    end
    varargout{k}   = vark;
end
end