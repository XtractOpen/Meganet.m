classdef LeapFrogNN < abstractMeganetElement
    % Residual Neural Network block
    %
    % Y_k+1 = 2*Y_{k} - Y_{k-1} - h^2*K'*sigma(K*Y_k))
    %
    % References:
    %
    % Haber E, Ruthotto L: Stable Architectures for Deep Neural Networks, 
    %      Inverse Problems, 2017
    %
    % Chang B, Meng L, Haber E, Ruthotto L, Begert D, Holtham E: 
    %      Reversible Architectures for Arbitrarily Deep Residual Neural Networks, 
    %      AAAI Conference on Artificial Intelligence 2018
    %
    % Haber E, Ruthotto L, Holtham E, Jun SH: 
    %      Learning across scales - A multiscale method for Convolution Neural Networks
    %      AAAI Conference on Artificial Intelligence 2018
    properties
        layer
        nt
        h
        outTimes
        Q
        useGPU
        precision
    end
    
    methods
        function this = LeapFrogNN(layer,nt,h,varargin)
            if nargin==0
                this.runMinimalExample;
                return;
            end
            useGPU = [];
            precision = [];
            outTimes = zeros(nt,1); outTimes(end)=1;
            Q = 1.0;
            for k=1:2:length(varargin)     % overwrites default parameter
                eval([varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            if not(isempty(useGPU))
                layer.useGPU = useGPU;
            end
            if not(isempty(precision))
                layer.precision = precision;
            end
            this.layer = layer;
            if nFeatOut(layer)~=nFeatIn(layer)
                error('%s - dim. of input and output features must agree for ResNet layers',mfilename);
            end
            this.nt    = nt;
            this.h     = h;
            this.outTimes = outTimes;
            this.Q = Q;
        end
        
        function n = nTheta(this)
            n = this.nt*nTheta(this.layer);
        end
        function n = nFeatIn(this)
            n = nFeatIn(this.layer);
        end
        function n = nFeatOut(this)
            n = nFeatOut(this.layer);
        end
        
        function n = nDataOut(this)
            if numel(this.Q)==1
                n = nnz(this.outTimes)*nFeatOut(this.layer);
            else
                n = nnz(this.outTimes)*size(this.Q,1);
            end
        end
        
        function theta = initTheta(this)
            theta = [];
            for k=1:this.nt
                theta = [theta; vec(initTheta(this.layer))];
            end
        end
        
        % ------- apply forward problems -----------
        function [Ydata,Y,tmp] = apply(this,theta,Y0)
            nex = numel(Y0)/nFeatOut(this);
            Y   = reshape(Y0,[],nex);
            if nargout>1;    tmp = cell(this.nt+1,2); tmp{1,1} = Y0; end
            
            theta = reshape(theta,[],this.nt);
            
            Ydata = [];
            
            Yold = 0;
            for i=1:this.nt
                [Z,~,tmp{i,2}] = apply(this.layer,theta(:,i),Y);
                Ytemp = Y;
                Y =  2*Y - Yold + this.h^2 * Z;
                Yold = Ytemp;
                if this.outTimes(i)==1
                    Ydata = [Ydata;this.Q*Y];
                end
                if nargout>1, tmp{i+1,1} = Y; end
            end
        end
        
        % -------- Jacobian matvecs ---------------
        function [dYdata,dY] = JYmv(this,dY,theta,~,tmp)
            if isempty(dY)
                dY = 0.0;
            elseif numel(dY)>1
                nex = numel(dY)/nFeatOut(this);
                dY   = reshape(dY,[],nex);
            end
            
            dYdata=[];
            
            dYold = 0;
            theta  = reshape(theta,[],this.nt);
            for i=1:this.nt
                dYtemp = dY;
                dY     = 2*dY - dYold + this.h^2* JYmv(this.layer,dY,theta(:,i),tmp{i,1},tmp{i,2});
                if this.outTimes(i)==1
                    dYdata = [dYdata;this.Q*dY];
                end
                dYold  = dYtemp;
            end
        end
        
        
        function [dYdata,dY] = Jmv(this,dtheta,dY,theta,~,tmp)
            if isempty(dY)
                dY = 0.0;
            elseif numel(dY)>1
                nex = numel(dY)/nFeatOut(this);
                dY   = reshape(dY,[],nex);
            end
            
            dYold = 0;
            dYdata = [];
            theta  = reshape(theta,[],this.nt);
            dtheta = reshape(dtheta,[],this.nt);
            for i=1:this.nt
                dYtemp = dY;
                dY = 2*dY - dYold + this.h^2* Jmv(this.layer,dtheta(:,i),dY,theta(:,i),tmp{i,1},tmp{i,2});
                if this.outTimes(i)==1
                    dYdata = [dYdata;this.Q*dY];
                end
                dYold = dYtemp;
            end
        end
        
        % -------- Jacobian' matvecs ----------------
        
        function W = JYTmv(this,Wdata,W,theta,Y,tmp)
            % call JYTmv (saving computations of the derivatives w.r.t.
            % theta)
            nex = numel(Y)/nFeatIn(this);
            if ~isempty(Wdata)
                Wdata = reshape(Wdata,[],nnz(this.outTimes),nex);
            end
            if isempty(W)
                W = 0;
            else
                W     = reshape(W,[],nex);
            end
            
            theta  = reshape(theta,[],this.nt);
            Wold = 0;
            
            cnt = nnz(this.outTimes); 
            for i=this.nt:-1:1
                if this.outTimes(i)==1
                    W = W + this.Q'*squeeze(Wdata(:,cnt,:));
                    cnt = cnt-1;
                end
                
                dW = JYTmv(this.layer,W,[],theta(:,i),tmp{i,1},tmp{i,2});
                Wtemp = W;
                W     = 2*W - Wold + this.h^2*dW;
                Wold  = Wtemp;
            end
        end
        
        function [dtheta,W] = JTmv(this,Wdata,W,theta,Y,tmp,doDerivative)
            if not(exist('doDerivative','var')) || isempty(doDerivative)
               doDerivative =[1;0]; 
            end
            nex = numel(Y)/nFeatIn(this);
            
            if ~isempty(Wdata)
                Wdata = reshape(Wdata,[],nnz(this.outTimes),nex);
            end
            if isempty(W) 
                W = 0;
            elseif numel(W)>1
                W     = reshape(W,[],nex);
            end
            
            theta  = reshape(theta,[],this.nt);
            dtheta = 0*theta;
            Wold   = 0;
            cnt = nnz(this.outTimes);
            for i=this.nt:-1:1
                if this.outTimes(i)==1
                    W = W + this.Q'*squeeze(Wdata(:,cnt,:));
                    cnt = cnt-1;
                end
                
                [dmbi,dW] = JTmv(this.layer,W,[],theta(:,i),tmp{i,1},tmp{i,2});
                dtheta(:,i)  = this.h^2*dmbi;
                Wtemp = W;
                W     = 2*W - Wold + this.h^2*dW;
                Wold  = Wtemp;
            end
            dtheta = vec(dtheta);
            if nargout==1 && all(doDerivative==1)
                dtheta=[dtheta(:); W(:)];
            end
        end
        % ------- functions for handling GPU computing and precision ----
        function this = set.useGPU(this,value)
            if (value~=0) && (value~=1)
                error('useGPU must be 0 or 1.')
            else
                this.layer.useGPU  = value;
            end
        end
        function this = set.precision(this,value)
            if not(strcmp(value,'single') || strcmp(value,'double'))
                error('precision must be single or double.')
            else
                this.layer.precision = value;
            end
        end
        function useGPU = get.useGPU(this)
            useGPU = this.layer.useGPU;
        end
        function precision = get.precision(this)
            precision = this.layer.precision;
        end
        
        function runMinimalExample(~)
            nex = 10;
            nK  = [4 4];
            
            D   = dense(nK);
            S   = singleLayer(D);
            net = LeapFrogNN(S,2,.01);
            mb  = randn(nTheta(net),1);
            
            Y0  = randn(nK(2),nex);
            [Ydata,~,tmp]   = net.apply(mb,Y0);
            dmb = reshape(randn(size(mb)),[],net.nt);
            dY0  = randn(size(Y0));
            
            dY = net.Jmv(dmb(:),dY0,mb,[],tmp);
            for k=1:14
                hh = 2^(-k);
                
                Yt = net.apply(mb+hh*dmb(:),Y0+hh*dY0);
                
                E0 = norm(Yt(:)-Ydata(:));
                E1 = norm(Yt(:)-Ydata(:)-hh*dY(:));
                
                fprintf('h=%1.2e\tE0=%1.2e\tE1=%1.2e\n',hh,E0,E1);
            end
            
            W = randn(size(Ydata));
            t1  = W(:)'*dY(:);
            
            [dWdmb,dWY] = net.JTmv(W,[],mb,Y0,tmp);
            t2 = dmb(:)'*dWdmb(:) + dY0(:)'*dWY(:);
            
            fprintf('adjoint test: t1=%1.2e\tt2=%1.2e\terr=%1.2e\n',t1,t2,abs(t1-t2));
            
            
            
            
            
        end
    end
    
end

