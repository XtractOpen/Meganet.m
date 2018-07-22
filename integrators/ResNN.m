classdef ResNN < abstractMeganetElement
    % Residual Neural Network block
    %
    % Y_k+1 = Y_k + h*layer{k}(trafo(theta{k},Y_k))
    
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
        function this = ResNN(layer,nt,h,varargin)
            if nargin==0
                this.runMinimalExample;
                return;
            end
            useGPU = [];
            precision = [];
            outTimes  = zeros(nt,1); outTimes(end)=1;
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
            theta = repmat(vec(initTheta(this.layer)),this.nt,1);
        end
        
        function [net2,theta2] = prolongateWeights(this,theta)
            % piecewise linear interpolation of network weights 
            t1 = 0:this.h:(this.nt-1)*this.h;
            
            net2 = ResNN(this.layer,2*this.nt,this.h/2,'useGPU',this.useGPU,'Q',this.Q,'precision',this.precision);
            net2.outTimes = (sum(this.outTimes)>0)*net2.outTimes;
          
            t2 = 0:net2.h:(net2.nt-1)*net2.h;
            
            theta2 = inter1D(theta,t1,t2);
        end
        
        
        % ------- apply forward problems -----------
        function [Ydata,Y,tmp] = apply(this,theta,Y0)
            nex = numel(Y0)/nFeatIn(this);
            Y   = reshape(Y0,[],nex);
            if nargout>1;    tmp = cell(this.nt,2); end
            
            theta = reshape(theta,[],this.nt);
            
            Ydata = [];
            for i=1:this.nt
                if (nargout>1), tmp{i,1} = Y; end
                [Z,~,tmp{i,2}] = apply(this.layer,theta(:,i),Y);
                Y =  Y + this.h * Z;
                if this.outTimes(i)==1
                    Ydata = [Ydata;this.Q*Y];
                end
            end
        end
        
        % -------- Jacobian matvecs ---------------
        function [dYdata,dY] = JYmv(this,dY,theta,~,tmp)
            if isempty(dY)
                dY = 0.0;
            elseif numel(dY)>1
                nex = numel(dY)/nFeatIn(this);
                dY   = reshape(dY,[],nex);
            end
            dYdata = [];
            theta  = reshape(theta,[],this.nt);
            for i=1:this.nt
                dY = dY + this.h* JYmv(this.layer,dY,theta(:,i),tmp{i,1},tmp{i,2});
                if this.outTimes(i)==1
                    dYdata = [dYdata; this.Q*dY];
                end
            end
        end
        
        
        function [dYdata,dY] = Jmv(this,dtheta,dY,theta,~,tmp)
            if isempty(dY)
                dY = 0.0;
            elseif numel(dY)>1
                nex = numel(dY)/nFeatIn(this);
                dY   = reshape(dY,[],nex);
            end
            
            dYdata = [];
            theta  = reshape(theta,[],this.nt);
            dtheta = reshape(dtheta,[],this.nt);
            for i=1:this.nt
                  dY = dY + this.h* Jmv(this.layer,dtheta(:,i),dY,theta(:,i),tmp{i,1},tmp{i,2});
                  if this.outTimes(i)==1
                      dYdata = [dYdata;this.Q*dY];
                  end
            end
        end
        
        % -------- Jacobian' matvecs ----------------
        
        function W = JYTmv(this,Wdata,W,theta,Y,tmp)
            nex = numel(Y)/nFeatIn(this);
            if ~isempty(Wdata)
                Wdata = reshape(Wdata,[],nnz(this.outTimes),nex);
            end
            if isempty(W)
                W = 0;
            elseif not(isscalar(W))
                W     = reshape(W,[],nex);
            end
            theta  = reshape(theta,[],this.nt);
            
            cnt = nnz(this.outTimes);
            for i=this.nt:-1:1
                Yi = tmp{i,1};
                if  this.outTimes(i)==1
                    W = W + this.Q'*squeeze(Wdata(:,cnt,:));
                    cnt = cnt-1;
                end
                dW = JYTmv(this.layer,W,[],theta(:,i),Yi,tmp{i,2});
                W  = W + this.h*dW;
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
            cnt = nnz(this.outTimes);
            dtheta = 0*theta;
            for i=this.nt:-1:1
                Yi = tmp{i,1};
                if  this.outTimes(i)==1
                    W = W + this.Q'* squeeze(Wdata(:,cnt,:));
                    cnt = cnt-1;
                end
                [dmbi,dW] = JTmv(this.layer,W,[],theta(:,i),Yi,tmp{i,2});
                dtheta(:,i)  = this.h*dmbi;
                W = W + this.h*dW;
            end
            dtheta = vec(dtheta);
            if nargout==1 && all(doDerivative==1)
                dtheta=[dtheta(:); W(:)];
            end
        end
        
        function [thFine] = prolongateConvStencils(this,theta,getRP)
            % prolongate convolution stencils, doubling image resolution
            %
            % Inputs:
            %
            %   theta - weights
            %   getRP - function for computing restriction operator, R, and
            %           prolongation operator, P. Default @avgRestrictionGalerkin
            %
            % Output
            %  
            %   thFine - prolongated stencils
            
            if not(exist('getRP','var')) || isempty(getRP)
                getRP = @avgRestrictionGalerkin;
            end
            
            thFine = reshape(theta,[],this.nt);
            for k=1:this.nt
                 thFine(:,k) = vec(prolongateConvStencils(this.layer,thFine(:,k),getRP));
            end
            thFine = vec(thFine);
        end
        function [thCoarse] = restrictConvStencils(this,theta,getRP)
            % restrict convolution stencils, dividing image resolution by two
            %
            % Inputs:
            %
            %   theta - weights
            %   getRP - function for computing restriction operator, R, and
            %           prolongation operator, P. Default @avgRestrictionGalerkin
            %
            % Output
            %  
            %   thCoarse - restricted stencils
            
            if not(exist('getRP','var')) || isempty(getRP)
                getRP = @avgRestrictionGalerkin;
            end
            
            thCoarse = reshape(theta,[],this.nt);
            for k=1:this.nt
                thCoarse(:,k) = vec(restrictConvStencils(this.layer,thCoarse(:,k),getRP));
            end
            thCoarse = vec(thCoarse);
        end
        
        % ------- functions for handling GPU computing and precision ---- 
        function this = set.useGPU(this,value)
            if (value~=0) && (value~=1)
                error('useGPU must be 0 or 1.')
            else
                this.layer.useGPU  = value;
            end
            this.Q = gpuVar(value,this.precision,this.Q);
        end
        function this = set.precision(this,value)
            if not(strcmp(value,'single') || strcmp(value,'double'))
                error('precision must be single or double.')
            else
                this.layer.precision = value;
            end
            this.Q = gpuVar(this.useGPU,value,this.Q);
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
%              S   = doubleLayer(D,D);
%             S = singleLayer(D)
            S = doubleSymLayer(D);
            nt = 20;
            outTimes = zeros(nt,1);
            outTimes([1;10;nt])= 1;
            net = ResNN(S,nt,.1,'outTimes',outTimes);
            mb  = randn(nTheta(net),1);
            
            Y0  = randn(nK(2),nex);
            [Ydata,~,dA]   = net.apply(mb,Y0);
            dmb = reshape(randn(size(mb)),[],net.nt);
            dY0  = randn(size(Y0));
            
            dY = net.Jmv(dmb(:),dY0,mb,Y0,dA);
            for k=1:14
                hh = 2^(-k);
                
                Yt = net.apply(mb+hh*dmb(:),Y0+hh*dY0);
                
                E0 = norm(Yt(:)-Ydata(:));
                E1 = norm(Yt(:)-Ydata(:)-hh*dY(:));
                
                fprintf('h=%1.2e\tE0=%1.2e\tE1=%1.2e\n',hh,E0,E1);
            end
            
            W = randn(size(Ydata));
            t1  = W(:)'*dY(:);
            
            [dWdmb,dWY] = net.JTmv(W,[],mb,Y0,dA);
            t2 = dmb(:)'*dWdmb(:) + dY0(:)'*dWY(:);
            
            fprintf('adjoint test: t1=%1.2e\tt2=%1.2e\terr=%1.2e\n',t1,t2,abs(t1-t2));
        end
    end
    
end

