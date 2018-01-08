classdef lsObjFctn < objFctn
    % classdef lsObjFctn < objFctn
    %
    % Linear Least-Squares objective function
    %
    %  min | A*x - vec(Y)|
    %
    
    properties
        A           % linear operator
        Y           % Data
        pReg        % regularizer
        useGPU      % flag for GPU computing
        precision   % flag for precision
    end
    
    methods
        function this = lsObjFctn(A,Y,pReg,varargin)
            
            if nargout==0 && nargin==0
                this.runMinimalExample;
                return;
            end
            useGPU    = [];
            precision = [];
            for k=1:2:length(varargin)     % overwrites default parameter
                eval([varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            
            if exist('pReg','var') && not(isempty(pReg))
                this.pReg      = pReg;
            end
            
            if not(isempty(useGPU))
                this.useGPU = useGPU;
            end
            
            if not(isempty(precision))
                this.precision=precision;
            end
%             [A,Y] = gpuVar(this.useGPU,this.precision,A,Y);
            this.A         = A;
            this.Y         = reshape(Y,1,[]);
        end
        function x = split(this,x)
        end
        
        
        function [Jc,para,dJ,H,PC] = eval(this,x,idx)
            
            compGrad = nargout>2;
            compHess = nargout>3;
            PC = [];
           
            % compute loss
            if not(exist('idx','var')) || isempty(idx)
                Ak = this.A;
                Yk = this.Y';
            else
                Ak = this.A(idx,:);
                Yk = this.Y(idx)';
            end
            nex = numel(Yk);
            
            res = Ak*x-Yk;
            Jc   = 0.5*(res'*res)/nex;

            if not(isempty(this.pReg))
                [Rx,hisRx,dRx,d2Rx]      = regularizer(this.pReg,x);
            else
                Rx = 0; dRx = 0; d2Rx = 0; hisRx = [];
            end
            Jc = Jc + Rx;
            
                if compGrad
                    dJ = (Ak'*res)/nex + dRx;
                end
                
                hisLoss  = [Jc];
                    if compHess
                        H   = (Ak'*Ak)/nex+ d2Rx;
                    end
            para = struct('F',Jc,'R',Rx);

            
            if nargout>4
                PCth = H;
            end
            
        end
        

        
        function [str,frmt] = hisNames(this)
            str = {'F(x)'};
            frmt = {'%1.2e'};
            if not(isempty(this.pReg))
                [s,f] = hisNames(this.pReg);
                s{1} = [s{1} '(x)'];
                str  = [str, s{:}];
                frmt = [frmt, f{:}];
            end
        end
        
        function his = hisVals(this,para)
            his = para.F;
            if not(isempty(this.pReg))
                his = [his, hisVals(this.pReg,para.R) this.pReg.alpha];
            end
        end
        
        function str = objName(this)
            str = 'lsObjFctn';
        end
        % ------- functions for handling GPU computing and precision ----
        function this = set.useGPU(this,value)
            if isempty(value)
                return
            elseif(value~=0) && (value~=1)
                error('useGPU must be 0 or 1.')
            else
                if not(isempty(this.net)); this.net.useGPU       = value; end
                if not(isempty(this.pRegTheta)); this.pRegTheta.useGPU       = value; end
                if not(isempty(this.pRegW)); this.pRegW.useGPU       = value; end
                
                [this.A,this.Y] = gpuVar(value,this.precision,...
                                                         this.A,this.Y);
            end
        end
        function this = set.precision(this,value)
            if isempty(value)
                return
            elseif not(strcmp(value,'single') || strcmp(value,'double'))
                error('precision must be single or double.')
            else
                if not(isempty(this.net)); this.net.precision       = value; end
                if not(isempty(this.pRegTheta)); this.pRegTheta.precision       = value; end
                if not(isempty(this.pRegW)); this.pRegW.precision       = value; end
                
                [this.A,this.Y] = gpuVar(this.useGPU,value,...
                                                         this.A,this.Y);
            end
        end
        function useGPU = get.useGPU(this)
                if not(isempty(this.pReg)) && not(isempty(this.pReg.useGPU))
                    useGPU = this.pReg.useGPU;
                else
                    useGPU = 0;
                end
        end
        function precision = get.precision(this)
            isSingle = 1;
            if not(isempty(this.pReg)) && not(isempty(this.pReg.precision))
                isSingle = strcmp(this.pReg.precision,'single');
            end
            if all(isSingle==1)
                precision = 'single';
            elseif all(isSingle==0)
                precision = 'double';
            else
                error('precision flag must agree');
            end

        end

        function runMinimalExample(~)
            
            nex    = 400; nf =20;
            A      = sprandn(nex,20,.1);
            x      = randn(nf,1);
            Y      = A*x;
            
            pReg  = tikhonovReg(.01*speye(numel(x)));
            
            f1 = lsObjFctn(A,Y,pReg);
            
            % [Jc,para,dJ,H,PC] = fctn([Kb(:);W(:)]);
            % checkDerivative(fctn,[Kb(:);W(:)])
            x0 = randn(size(x));
            opt =sd('out',1,'maxIter',20);
            [xn] = solve(opt,f1,x0);
            norm(xn(:)-x(:))
            
        end
    end
end










