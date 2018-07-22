classdef connector < abstractMeganetElement
    % connector block that applies Y = K*Y0 + b, where K and b are fixed
    
    properties
        K
        b
        outTimes 
        Q
        useGPU
        precision
    end
    
    
    methods
        function this = connector(K,b,varargin)
            if nargout==0 && nargin==0
                this.runMinimalExample;
                return;
            end
            outTimes = 0;
            Q = 1.0;
            for k=1:2:length(varargin)     % overwrites default parameter
                eval([varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            
            if nargin==1 || isempty(b)
                b = 0.0;
            end
            
            this.K = K;
            this.b = b;
            this.outTimes = outTimes;
            this.Q = Q;
        end
        
        % -------- counting ---------
        function np = nTheta(this)
            np = 0;
        end
        function n = nFeatIn(this)
            n = size(this.K,2);
        end
        function n = nFeatOut(this)
            n = size(this.K,1);
        end
        function n = nDataOut(this)
            if numel(this.Q)==1
                n = nnz(this.outTimes)*nFeatOut(this);
            else
                n = nnz(this.outTimes)*size(this.Q,1);
            end
        end
        
        function theta = initTheta(this)
            theta = [];
        end
        
        % -------- apply forward problem -------
        function [Ydata,Y,tmp] = apply(this,~,Y0)
            nex = numel(Y0)/nFeatIn(this);
            Y0  = reshape(Y0,[],nex);
            Y = this.K*Y0 + this.b;
            if this.outTimes==1
                Ydata = this.Q*Y;
            else
                Ydata = [];
            end
            tmp = {Y0};
            tmp{2} = Y;
        end
        
        function [dYdata,dY] = Jmv(this,~,dY,~,~,~)
            if (isempty(dY)); dY = 0.0; return; end
            nex = numel(dY)/nFeatIn(this);
            dY  = reshape(dY,[],nex);
            dY = this.K*dY;
            if this.outTimes==1
                dYdata = this.Q*dY;
            else
                dYdata = [];
            end
        end
            
        function [dtheta,W] = JTmv(this,Wdata,W,~,Y,~,doDerivative)
            nex = numel(Y)/nFeatIn(this);
            if isempty(W)
                W = 0;
            elseif not(isscalar(W))
                W     = reshape(W,[],nex);
            end
            if ~isempty(Wdata)
                Wdata = reshape(Wdata,[],nex);
                W     = W+ this.Q'*Wdata;
            end
            
            dtheta = [];
            W   = this.K'*W;
            if nargout==1 && all(doDerivative==1)
                dtheta=[dtheta(:); W(:)];
            end
        end
        function runMinimalExample(~)
            nex = 10;
            
            net = connector(randn(4,2),.3,'outTimes',1);
            theta  = randn(nTheta(net),1);
            
            Y0  = randn(2,nex); 
            [Ydata,~,tmp]   = net.apply(theta,Y0);
            dmb = randn(size(theta));
            dY0 = randn(size(Y0));
            
            dY = net.Jmv(dmb(:),dY0,theta,Y0,tmp);
            for k=1:14
                hh = 2^(-k);
                
                Yt = net.apply(theta+hh*dmb(:),Y0+hh*dY0);
                
                E0 = norm(Yt(:)-Ydata(:));
                E1 = norm(Yt(:)-Ydata(:)-hh*dY(:));
                
                fprintf('h=%1.2e\tE0=%1.2e\tE1=%1.2e\n',hh,E0,E1);
            end
            
            W = randn(size(Ydata));
            t1  = W(:)'*dY(:);
            
            [dWdmb,dWY] = net.JTmv(W,[],theta,Y0,tmp);
            t2 = dmb(:)'*dWdmb(:) + dY0(:)'*dWY(:);
            
            fprintf('adjoint test: t1=%1.2e\tt2=%1.2e\terr=%1.2e\n',t1,t2,abs(t1-t2));
        end
    end
    
end

