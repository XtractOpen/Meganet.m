classdef l1Reg
    % classdef l1Reg
    %
    % L1-L2 regularizer
    %
    % R(x) =  a1* psi(B1*(x-xref)) + 0.5*a2*|B2*(x-xref)|^2
    properties
        alpha
        B1
        B2
        xref
        eps
        useGPU
        precision
    end
    
    methods
        
        function this = l1Reg(B1,alpha,xref,varargin)
            if nargin==0
                this.runMinimalExample()
                return
            end
           useGPU = [];
           precision = [];
           B2 = opEye(sizeLastDim(B1));
           eps = 1e-3;
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
           this.B1 = B1;
           this.B2 = B2;
           if not(exist('alpha','var')) || isempty(alpha)
               this.alpha = [1.0; 1.0];
           elseif numel(alpha)==1
               this.alpha = alpha*[1;1];
           else
               this.alpha = alpha;
           end
           if not(exist('xref','var')) || isempty(xref)
               this.xref = 0.0;
           else
               this.xref = gpuVar(useGPU,precision, xref);
           end
           this.eps = eps;
        end
        
        function nth = nTheta(this)
            nth = sizeLastDim(this.B1);
        end
        
        function [Sc,para,dS,d2S] = regularizer(this,x)
            u = x-this.xref;
            
            % l1- part
            wTV  = sqrt((this.B1*u).^2 +this.eps);
            S1   = this.alpha(1)*sum(wTV);
            d2S1  = this.B1'*(opDiag(1./wTV)*this.B1)*this.alpha(1);
            dS1  = d2S1*u;

            % quadratic part
            d2S2 = (this.B2'*this.B2)*this.alpha(2);
            dS2 = d2S2*u;
            S2 = .5*sum(vec(u'*dS2));
            
            % sum up
            Sc = S1+S2;
            dS = dS1 + dS2;
            d2S = d2S1 + d2S2;
            para = [Sc this.alpha'];
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
        
        function runMinimalExample(~)
            nt = 10;
            h = 0.1;
            nTh = 100;
            B1 = opTimeDer(nTh,nt,h);
            reg = l1Reg(B1,[1.0;0.0]);
            
            th = randn(nTh,1);
            [Sc,para,dS,d2S] = regularizer(reg,th);
            fctn = @(th) regularizer(reg,th);
            checkDerivative(fctn,th,'out',2)
        end
        % ------- functions for handling GPU computing and precision ----
        function this = set.useGPU(this,value)
            if isempty(value)
                return
            elseif(value~=0) && (value~=1)
                error('useGPU must be 0 or 1.')
            else
                this.B1.useGPU = value;
                this.B2.useGPU = value;
                this.xref = gpuVar(value,this.precision,this.xref);
            end
        end
        function this = set.precision(this,value)
            if isempty(value)
                return
            elseif not(strcmp(value,'single') || strcmp(value,'double'))
                error('precision must be single or double.')
            else
                this.B1.precision = value;
                this.B2.precision = value;
                this.xref = gpuVar(this.useGPU,value,this.xref);
            end
            this.precision = value;
        end
        function useGPU = get.useGPU(this)
            if isnumeric(this.B1)
                useGPU = isa(this.B1,'gpuArray');
            else
                useGPU = this.B1.useGPU;
            end
        end
        function precision = get.precision(this)
            if isnumeric(this.B1)
                if isempty(this.B1)
                    precision = [];
                elseif isa(this.B1(1),'single')
                    precision = 'single';
                else
                    precision = 'double';
                end
            else
                precision = this.B1.precision;
            end
        end
    end
    
end

