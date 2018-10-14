classdef tikhonovReg
    % classdef tikhonovReg
    %
    % tikhonov regularizer
    %
    % R(x) = 0.5* alpha* | B*(x-xref)| ^2
    
    properties
        alpha
        B
        xref
        useGPU
        precision
    end
    
    methods
        
        function this = tikhonovReg(B,alpha,xref,varargin)
           useGPU = [];
           precision = [];
           for k=1:2:length(varargin)     % overwrites default parameter
                eval([varargin{k},'=varargin{',int2str(k+1),'};']);
           end
           if not(isempty(useGPU))
                this.useGPU    = useGPU;
           end
           if not(isempty(precision))
                this.precision = precision;
           end
           
%            this.B = B.gpuVar(useGPU,precision);
           this.B = B;
           if not(exist('alpha','var')) || isempty(alpha)
               this.alpha = 1.0;
           else
               this.alpha = alpha;
           end
           if not(exist('xref','var')) || isempty(xref)
               this.xref = 0.0;
           else
               this.xref = gpuVar(useGPU,precision, xref);
           end
           
        end
        
        function [Sc,para,dS,d2S] = regularizer(this,x)
            u = x-this.xref;
            d2S = getA(this);
            
            dS = d2S*u;
          
            Sc = .5*sum(vec(u'*dS));
            para = [Sc this.alpha];
        end
        
        function A = getA(this)
            A = this.alpha*(this.B'*this.B);
        end
        function [str,frmt] = hisNames(this)
            str  = {'R','alpha'};
            frmt = {'%-12.2e','%-12.2e'};
        end
        function str = hisVals(this,para)
            str = para;
        end       
        
        function PC = getPC(this)
            PC = getPCop(this.B);
        end
        
        function nt = nTheta(this)
            nt = sizeLastDim(this.B);
        end
        
        % ------- functions for handling GPU computing and precision ----
        function this = set.useGPU(this,value)
            if isempty(value)
                return
            elseif(value~=0) && (value~=1)
                error('useGPU must be 0 or 1.')
            else
                this.B = gpuVar(value,this.precision,this.B);
                this.xref = gpuVar(value,this.precision,this.xref);
            end
        end
        function this = set.precision(this,value)
            if isempty(value)
                return
            elseif not(strcmp(value,'single') || strcmp(value,'double'))
                error('precision must be single or double.')
            else
                this.B = gpuVar(this.useGPU,value,this.B);
                this.xref = gpuVar(this.useGPU,value,this.xref);
            end
            this.precision = value;
        end
        function useGPU = get.useGPU(this)
            if isnumeric(this.B)
                useGPU = isa(this.B,'gpuArray');
            else
                useGPU = this.B.useGPU;
            end
        end
        function precision = get.precision(this)
            if isnumeric(this.B)
                if isempty(this.B)
                    precision = [];
                elseif isa(this.B(1),'single')
                    precision = 'single';
                else
                    precision = 'double';
                end
            else
                precision = this.B.precision;
            end
        end
    end
    
end

