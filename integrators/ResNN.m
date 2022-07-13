classdef ResNN < abstractMeganetElement
    % Residual Neural Network block
    %
    % Y_k+1 = Y_k + h*layer{k}(Theta(t_k),Y_k)
    % 
    % Theta(t) = sum_{i=1}^n p_i(t) * theta_i
    % 
    % with 
    %  - theta_i = coefficients of the weight function Theta,
    %  length(theta_i) = nTheta(layer) 
    %  - p_i = basis functions
    % 
    
    properties
        layer
        nt
        h
        useGPU
        precision 
        A       % A_ij = p_i(t_j), t_j = j x h 
    end
    
    methods
        function this = ResNN(layer,nt,h,varargin)
            if nargin==0
                this.runMinimalExample;
                return;
            end
            % This is the default parameters but can be overwritten
            useGPU = [];
            precision = [];
            A = eye(nt); 
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
            if any(sizeFeatOut(layer)~=sizeFeatIn(layer))
                error('%s - dim. of input and output features must agree for ResNet layers',mfilename);
            end
            this.A     = A;
            this.nt    = nt;
            this.h     = h;
        end
        
        function n = nTheta(this)
            n = size(this.A,1)*nTheta(this.layer);
        end
        function n = sizeFeatIn(this)
            n = sizeFeatIn(this.layer);
        end
        function n = sizeFeatOut(this)
            n = sizeFeatOut(this.layer);
        end
        
        function theta = initTheta(this)
            theta = repmat(vec(initTheta(this.layer)),size(this.A,1),1);
        end
        
        function [net2,theta2] = prolongateWeights(this,theta)
            if norm(this.A - eye(this.nt),'fro')/this.nt < 1e-5

                % piecewise linear interpolation of network weights
                t1 = 0:this.h:(this.nt-1)*this.h;
                

                net2 = ResNN(this.layer,2*this.nt,this.h/2,'useGPU',this.useGPU,'precision',this.precision,'A', speye(this.nt*2));
                t2 = 0:net2.h:(net2.nt-1)*net2.h;

                theta2 = inter1D(theta,t1,t2);
            else
                error('prolongation not compatible with weight parameterization')
            end
        end
        
        
        %% ------- forwardProp forward problems -----------
        function [Y,tmp] = forwardProp(this,theta,Y,varargin) % interpret theta as coeff of polynomial (function of t), vector of coefficients
            doDerivative = (nargout>1);
            for k=1:2:length(varargin)     % overwrites default parameter
                eval([varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            
            if nargout>1;    tmp = cell(this.nt,2); end
            
            % evaluate polynomial at points 0,h,2h,etc. (time points)
            theta = reshape(theta,nTheta(this.layer),[]); % number of rows is number of weights layer needs, number of cols = how many coeffs
            Theta = theta*this.A;  % can be A*coeff, we will replace eye with A, # coeffs x # time pts
            
            for i=1:this.nt
                if doDerivative, tmp{i,1} = Y; end
                [Z,tmp{i,2}] = forwardProp(this.layer,Theta(:,i),Y,'doDerivative',doDerivative);
                Y =  Y + this.h * Z;
            end
        end
        
        % -------- Jacobian matvecs ---------------
        function dY = JYmv(this,dY,theta,~,tmp)
            if isempty(dY)
                dY = 0.0;
            elseif numel(dY)>1
                nex = numel(dY)/numelFeatIn(this);
                dY   = reshape(dY,[],nex);
            end
            theta = reshape(theta,nTheta(this.layer),[]); % number of rows is number of weights layer needs, number of cols = how many coeffs
            Theta = theta*this.A;  % can be A*coeff, we will replace eye with A, # coeffs x # time pts
            for i=1:this.nt
                dY = dY + this.h* JYmv(this.layer,dY,Theta(:,i),tmp{i,1},tmp{i,2});
            end
        end
        
        
        function dY = Jmv(this,dtheta,dY,theta,~,tmp)
            if isempty(dY)
                dY = 0.0;
            elseif numel(dY)>1
                nex = numel(dY)/numelFeatIn(this);
%                 dY   = reshape(dY,[],nex);
            end
            
            theta = reshape(theta,nTheta(this.layer),[]); % number of rows is number of weights layer needs, number of cols = how many coeffs
            Theta = theta*this.A;  % can be A*coeff, we will replace eye with A, # coeffs x # time pts
            dtheta = reshape(dtheta,nTheta(this.layer),[]); % number of rows is number of weights layer needs, number of cols = how many coeffs
            dTheta = dtheta*this.A;  % can be A*coeff, we will replace eye with A, # coeffs x # time pts
            
            
            for i=1:this.nt
                  dY = dY + this.h* Jmv(this.layer,dTheta(:,i),dY,Theta(:,i),tmp{i,1},tmp{i,2});
            end
        end
        
        % -------- Jacobian' matvecs ----------------
        
        function W = JYTmv(this,W,theta,Y,tmp)
            if isempty(W)
                W = 0;
            end
            theta = reshape(theta,nTheta(this.layer),[]); 
            Theta = theta*this.A;  
            
            for i=this.nt:-1:1
                Yi = tmp{i,1};
                dW = JYTmv(this.layer,W,Theta(:,i),Yi,tmp{i,2});
                W  = W + this.h*dW;
            end
        end
        
        function [dtheta,W] = JTmv(this,W,theta,Y,tmp,doDerivative,varargin)
            reduceDim=true;
            for k=1:2:length(varargin)     % overwrites default parameter
                eval([varargin{k},'=varargin{',int2str(k+1),'};']);
            end

            if not(exist('doDerivative','var')) || isempty(doDerivative)
               doDerivative =[1;0]; 
            end
            
            if isempty(W) 
                W = 0;
            end
            
            theta = reshape(theta,nTheta(this.layer),[]); 
            Theta = theta*this.A;  
            if reduceDim
                dTheta = 0*Theta;
            else
                dTheta = zeros(size(Theta,1),size(Theta,2),sizeLastDim(Y),'like',Theta);
            end
            
            
            for i=this.nt:-1:1
                Yi = tmp{i,1};
                [dThetai,dW] = JTmv(this.layer,W,Theta(:,i),Yi,tmp{i,2},[],varargin{:});
                if reduceDim
                    dTheta(:,i)  = this.h*dThetai;
                else
                    dTheta(:,i,:) = this.h*dThetai;
                end
                W = W + this.h*dW;
            end
            if reduceDim
                dtheta = vec(dTheta*this.A');
            else
%                 dtheta = sum(dTheta .* reshape(this.A',size(this.A,2),size(this.A,1),1),2);
                  dtheta = pagemtimes(dTheta,'none',full(this.A),'transpose');
                  dtheta = reshape(dtheta,[],sizeLastDim(Y));
            end

            if nargout==1 && all(doDerivative==1)
                dtheta=[dtheta(:); W(:)];
            end
        end
        
      %%  
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
        
        %% ------- functions for handling GPU computing and precision ---- 
        % the A matrix needs to be moved to GPU if we use this
        
        function this = set.useGPU(this,value)
            if (value~=0) && (value~=1)
                error('useGPU must be 0 or 1.')
            else
                this.A = gpuVar(value, this.layer.precision, this.A);
                this.layer.useGPU  = value;
            end
        end
        function this = set.precision(this,value)
            if not(strcmp(value,'single') || strcmp(value,'double'))
                error('precision must be single or double.')
            else
                this.A = gpuVar(this.layer.useGPU, value, this.A);
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
%              S   = doubleLayer(D,D);
%             S = singleLayer(D)
            S = doubleSymLayer(D);
            nt = 20;
            net = ResNN(S,nt,.1);
            mb  = randn(nTheta(net),1);
            
            Y0  = randn(nK(2),nex);
            [Y,dA]   = net.forwardProp(mb,Y0);
            dmb = reshape(randn(size(mb)),[],net.nt);
            dY0  = randn(size(Y0));
            
            dY = net.Jmv(dmb(:),dY0,mb,Y0,dA);
            for k=1:14
                hh = 2^(-k);
                
                Yt = net.forwardProp(mb+hh*dmb(:),Y0+hh*dY0);
                
                E0 = norm(Yt(:)-Y(:));
                E1 = norm(Yt(:)-Y(:)-hh*dY(:));
                
                fprintf('h=%1.2e\tE0=%1.2e\tE1=%1.2e\n',hh,E0,E1);
            end
            
            W = randn(size(Y));
            t1  = W(:)'*dY(:);
            
            [dWdmb,dWY] = net.JTmv(W,mb,Y0,dA);
            t2 = dmb(:)'*dWdmb(:) + dY0(:)'*dWY(:);
            
            fprintf('adjoint test: t1=%1.2e\tt2=%1.2e\terr=%1.2e\n',t1,t2,abs(t1-t2));
        end
    end
    
end

