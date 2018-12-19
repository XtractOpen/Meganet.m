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
        elseif (isa(vark,'handle') && isprop(vark,'precision')) || isfield(vark,'precision')
            vark.precision = 'single';
        end
    end
    
    if useGPU  % check if GPU exists, run on CPU if it does not
        try
            gpuDevice();
        catch E
            useGPU = 0;
            disp(['GPU error caught. Changing to using CPU because of '...
             'the following error:']);
            disp(E.identifier);
        end
    end

    
    if not(isempty(useGPU)) && useGPU && not(isa(vark,'gpuArray'))
        if isnumeric(vark) || islogical(vark)
            vark = gpuArray(vark);
        elseif (isa(vark,'handle') && isprop(vark,'useGPU')) || isfield(vark,'useGPU')
            vark.useGPU = useGPU;
        end
    end
    varargout{k}   = vark;
end
end