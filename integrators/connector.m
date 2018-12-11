classdef connector < abstractMeganetElement
    % connector block that applies Y = K*Y0 + b, where K and b are fixed
    
    properties
        K   % numeric K is unsupported, use LinearOperator(K)
        b
        useGPU
        precision
    end
    
    
    methods
        function this = connector(K,b,varargin)
            if nargout==0 && nargin==0
                this.runMinimalExample;
                return;
            end
            for k=1:2:length(varargin)     % overwrites default parameter
                eval([varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            
            if nargin==1 || isempty(b)
                b = 0.0;
            end
            
            if isnumeric(K) && ndims(K)==2
               K = LinearOperator(K); 
            end
            this.K = K;
            this.b = b;
        end
        
        % -------- counting ---------
        function np = nTheta(this)
            np = 0;
        end
        function n = sizeFeatIn(this)
                n = this.K.n;
        end
        function n = sizeFeatOut(this)
                n = this.K.m;
        end
        
        function theta = initTheta(this)
            theta = [];
        end
        
        % -------- forwardProp forward problem -------
        function [Y,tmp] = forwardProp(this,~,Y0,varargin)
            Y = this.K*Y0 + this.b;
            tmp = [];
        end
        
        function dY = Jmv(this,~,dY,~,~,~)
            if (isempty(dY)); dY = 0.0; return; end
            dY = this.K*dY;
        end
            
        function [dtheta,W] = JTmv(this,W,~,Y,~,doDerivative)
            if isempty(W)
                W = 0;
            end
            
            dtheta = [];
            W   = this.K'*W;
            if nargout==1 && all(doDerivative==1)
                dtheta=[dtheta(:); W(:)];
            end
        end
        
        function this = set.useGPU(this,value)
            if (value~=0) && (value~=1)
                error('useGPU must be 0 or 1.')
            else
                this.useGPU = value;
                [this.K, this.b] = gpuVar(value,this.precision,this.K,this.b);
            end
        end
        function this = set.precision(this,value)
            if not(strcmp(value,'single') || strcmp(value,'double'))
                error('precision must be single or double.')
            else
                this.precision = value;
                [this.K, this.b] = gpuVar(this.useGPU,value,this.K,this.b);
            end
        end
        
        function runMinimalExample(~)
            nex = 10;
            
            net = connector(LinearOperator(randn(4,2)),.3);
            theta  = randn(nTheta(net),1);
            
            Y0  = randn(2,nex); 
            [Y,tmp]   = net.forwardProp(theta,Y0);
            dmb = randn(size(theta));
            dY0 = randn(size(Y0));
            
            dY = net.Jmv(dmb(:),dY0,theta,Y0,tmp);
            for k=1:14
                hh = 2^(-k);
                
                Yt = net.forwardProp(theta+hh*dmb(:),Y0+hh*dY0);
                
                E0 = norm(Yt(:)-Y(:));
                E1 = norm(Yt(:)-Y(:)-hh*dY(:));
                
                fprintf('h=%1.2e\tE0=%1.2e\tE1=%1.2e\n',hh,E0,E1);
            end
            
            W = randn(size(Y));
            t1  = W(:)'*dY(:);
            
            [dWdmb,dWY] = net.JTmv(W,[],theta,Y0,tmp);
            t2 = dmb(:)'*dWdmb(:) + dY0(:)'*dWY(:);
            
            fprintf('adjoint test: t1=%1.2e\tt2=%1.2e\terr=%1.2e\n',t1,t2,abs(t1-t2));
        end
    end
    
end

