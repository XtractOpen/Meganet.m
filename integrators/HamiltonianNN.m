classdef HamiltonianNN < abstractMeganetElement
    % Double Layer Hamiltonian block
    %
    % Z_k+1 = Z_k - h*act(K(theta_k)'*Y_k + b),  
    % Y_k+1 = Y_k + h*act(K(theta)* Z_k+1 + b) 
    %
    % The input features are divided into Y and Z here based on the sizes 
    % of K.
    %
    % References:
    %
    % Chang B, Meng L, Haber E, Ruthotto L, Begert D, Holtham E: 
    %      Reversible Architectures for Arbitrarily Deep Residual Neural Networks, 
    %      AAAI Conference on Artificial Intelligence 2018
    
    properties
        activation
        K
        B
        nt
        h
        useGPU
        precision
    end
    
    methods
        function this = HamiltonianNN(activation,K,B,nt,h,varargin)
            if nargin==0
                this.runMinimalExample;
                return;
            end
            useGPU    = [];
            precision = [];
            for k=1:2:length(varargin)     % overwrites default parameter
                eval([varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            
            this.activation = activation;
            this.K = K;
            this.B = B;
            this.nt       = nt;
            this.h        = h;
        end
        
        function n = nTheta(this)
            n = this.nt*(nTheta(this.K)+ sizeLastDim(this.B));
        end
        function n = sizeFeatIn(this)
            n = sizeFeatOut(this.K);
%             n2 = sizeFeatIn(this.K);
%             if (numel(n1) > 2) && (numel(n2) > 2)
%                 % convolution layer. add chanels together
%                 n = n1;
%                 n(3) = n1(3) + n2(3);
%             else
%                 n = n1+n2;            
%             end
        end
        function n = sizeFeatOut(this)
            n = sizeFeatIn(this.K);
        end

        
        function theta = initTheta(this)
            theta = repmat([vec(initTheta(this.K));...
                            zeros(sizeLastDim(this.B),1)],this.nt,1);
        end
        
        function [net2,theta2] = prolongateWeights(this,theta)
            % piecewise linear interpolation of network weights 
            t1 = 0:this.h:(this.nt-1)*this.h;
            
            net2 = HamiltonianNN(this.activation,this.K,this.B,2*this.nt,this.h/2,'useGPU',this.useGPU,'precision',this.precision);
          
            t2 = 0:net2.h:(net2.nt-1)*net2.h;
            
            theta2 = inter1D(theta,t1,t2);
        end
        
        function [thetaK,thetaB] = split(this,x)
           x   = reshape(x,[],this.nt);
           thetaK = x(1:nTheta(this.K),:);
           thetaB = x(nTheta(this.K)+1:end,:);
        end

               % ------- forwardProp forward problems -----------
        function [Y,tmp] = forwardProp(this,theta,Y0,varargin)
            for k=1:2:length(varargin)     % overwrites default parameter
                eval([varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            
            Y = Y0;
            Z = 0*Y;
            [thetaK,thetaB] = split(this,theta);
            
            for i=1:this.nt
                Ki = getOp(this.K,thetaK(:,i)); 
                bi = this.B*thetaB(:,i);
                
                fY = this.activation(Ki'*Y + bi);
                Z  = Z - this.h*fY;
                
                fZ = this.activation(Ki*Z + bi);
                Y  = Y + this.h*fZ;
            end
            tmp = {Y,Z,Y0};
        end
        
        % -------- Jacobian matvecs ---------------
%         function dX = JYmv(this,dX,theta,~,tmp)
%             if isempty(dX) || (numel(dX)==0 && dX==0.0)
%                 dX     = zeros(size(Y),'like',Y);
%                 return
%             end
%             X0 = tmp{3};
%             [Y,Z] = splitData(this,X0);
%             [dY,dZ]   = splitData(this,dX);
%             [th1,th2] = split(this,theta);
%             for i=1:this.nt
%                 [fZ,tmp] = forwardProp(this.layer1,th1(:,i),Z,'storeInterm',1);
%                 dY = dY + this.h*JYmv(this.layer1,dZ,th1(:,i),Z,tmp);
%                 Y  = Y + this.h*fZ;
%                 
%                 [fY,tmp] = forwardProp(this.layer2,th2(:,i),Y);
%                 dZ = dZ - this.h*JYmv(this.layer2,dY,th2(:,i),Y,tmp);
%                 Z = Z - this.h*fY;
%             end
%             dX = unsplitData(this,dY,dZ);
%         end
        
        function dY = Jmv(this,dtheta,dY,theta,~,tmp)
            if isempty(dY)
                dY = 0*tmp{1};
            end
            Y = tmp{3};
            Z = 0*Y;
            dZ = 0*Z;
            
            [thK,thB]   = split(this,theta);
            [dthK,dthB] = split(this,dtheta);
            for i=1:this.nt
                 Ki  = getOp(this.K,thK(:,i));
                 dKi = getOp(this.K,dthK(:,i));
                 bi  = this.B*thB(:,i);
                 dbi = this.B*dthB(:,i);
                 
                 [fY,dfY]  = this.activation(Ki'*Y + bi);
                 JY = dfY.*(dKi'*Y+Ki'*dY+dbi);
                 dZ = dZ - this.h*JY;
                 Z  = Z  - this.h*fY;
                 
                 [fZ,dfZ] = this.activation(Ki*Z + bi);
                 JZ = dfZ.*(dKi*Z+Ki*dZ+dbi);
                 dY = dY + this.h*JZ;
                 Y = Y + this.h*fZ;
            end
        end
        
        % -------- Jacobian' matvecs ----------------
%         function W = JYTmv(this,W,theta,X0,tmp)
%             % nex = sizeLastDim(Y);
%             if isempty(W)
%                 WY = 0;
%                 WZ = 0;
%             elseif not(isscalar(W))
%                 [WY,WZ] = splitData(this,W);
%             end
%             
%             Y = tmp{1};
%             Z = tmp{2};
%             [th1,th2]  = split(this,theta);
%             
%             for i=this.nt:-1:1
%                 [fY,tmp] = forwardProp(this.layer2,th2(:,i),Y);
%                 dWY = JYTmv(this.layer2,WZ,th2(:,i),Y,tmp);
%                 WY  = WY - this.h*dWY;
%                 Z = Z + this.h*fY;
%                 
%                 [fZ,tmp] = forwardProp(this.layer1,th1(:,i),Z,'storeInterm',1);
%                 dWZ = JYTmv(this.layer1,WY,th1(:,i),Z,tmp);
%                 WY  = WZ + this.h*dWZ;
%                 Y = Y - this.h*fZ;
%             end
%             W = unsplitData(this,WY,WZ);
%         end
        
        function [dtheta,W] = JTmv(this,W,theta,X0,tmp,doDerivative)
            if not(exist('doDerivative','var')) || isempty(doDerivative)
               doDerivative =[1;0]; 
            end
            
            if isempty(W) || numel(W)==1
                WY = 0;
                WZ = 0;
            else
                WY = W;
                WZ = 0*WY;
            end

            Y = tmp{1};
            nd = ndims(Y);
            Z = tmp{2};
            [thK,thB]   = split(this,theta);
            [dthK,dthB] = split(this,0*theta);
            
            for i=this.nt:-1:1
                Ki = getOp(this.K,thK(:,i)); 
                bi = this.B*thB(:,i);
                
                [fZ,dfZ] = this.activation(Ki*Z + bi);
                dWZ = Ki'*(dfZ.*WY); 
                dthK(:,i) = dthK(:,i)+ this.h*JthetaTmv(this.K,dfZ.*WY,[],Z);
                dthB(:,i) = dthB(:,i) + this.h*vec(sum(this.B'*(dfZ.*WY),nd));
                WZ = WZ + this.h*dWZ;
                Y  = Y - this.h*fZ;
                
                [fY,dfY] = this.activation(Ki'*Y + bi);
                dWY = Ki*(dfY.*WZ); 
                dJK = reshape(JthetaTmv(this.K,dfY.*WZ,[],Y),size(Ki));
                dJK = vec(dJK');
                dthK(:,i) = dthK(:,i) - this.h*dJK;
                dthB(:,i) = dthB(:,i) - vec(sum(this.h*this.B'*(dfY.*WZ),nd));
                WY = WY - this.h*dWY;
                Z  = Z + this.h*fY;
            end
            dtheta = vec([dthK;dthB]);
            
%             W = unsplitData(this,WY,WZ);
            W = WY;
            if nargout==1 && all(doDerivative==1)
                dtheta=[dtheta; W(:)];
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
            
            [th1Fine,th2Fine] = split(this,theta);
            for k=1:this.nt
                th1Fine(:,k) = vec(prolongateConvStencils(this.layer1,th1Fine(:,k),getRP));
                th2Fine(:,k) = vec(prolongateConvStencils(this.layer2,th2Fine(:,k),getRP));
            end
            thFine = vec([th1Fine;th2Fine]);
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
            
            
           [th1Coarse,th2Coarse] = split(this,theta);
            for k=1:this.nt
                th1Coarse(:,k) = vec(restrictConvStencils(this.layer1,th1Coarse(:,k),getRP));
                th2Coarse(:,k) = vec(restrictConvStencils(this.layer2,th2Coarse(:,k),getRP));
            end
            thCoarse = vec([th1Coarse;th2Coarse]);
        end
        % ------- functions for handling GPU computing and precision ----
        function this = set.useGPU(this,value)
            if (value~=0) && (value~=1)
                error('useGPU must be 0 or 1.')
            else
                this.K.useGPU  = value;
%                 this.K.useGPU  = value;
            end
        end
        function this = set.precision(this,value)
            if not(strcmp(value,'single') || strcmp(value,'double'))
                error('precision must be single or double.')
            else
                this.K.precision = value;
%                 this.K.precision = value;
            end
        end
        function useGPU = get.useGPU(this)
            useGPU = this.K.useGPU;
        end
        function precision = get.precision(this)
            precision = this.K.precision;
        end
        
        function runMinimalExample(~)
            act = @tanhActivation;
            K    = dense([2,2]);
            B     = ones(2,1);
            net   = HamiltonianNN(act,K,B,10,.1);
            Y = [1;1];
            theta =  [vec([2 1;1 2]);0];
            theta = vec(repmat(theta,1,net.nt));

            
            [YN,tmp] = forwardProp(net,theta,Y); % Yd was deleted
            Ys = reshape(cell2mat(tmp(:,1)),2,[]);
            
            nex = 100;
             mb  = randn(nTheta(net),1);
            Y0  = randn(numelFeatIn(net),nex);
            
            [Y,tmp]   = net.forwardProp(mb,Y0);
            dmb = reshape(randn(size(mb)),[],net.nt);
            dY0  = randn(size(Y0));
            
            dY = net.Jmv(dmb(:),dY0,mb,Y0,tmp);
            for k=1:20
                hh = 2^(-k);
                
                Yt = net.forwardProp(mb+hh*dmb(:),Y0+hh*dY0);
                
                E0 = norm(Yt(:)-Y(:));
                E1 = norm(Yt(:)-Y(:)-hh*dY(:));
                
                fprintf('h=%1.2e\tE0=%1.2e\tE1=%1.2e\n',hh,E0,E1);
            end
            
            W = randn(size(Y));
            t1  = W(:)'*dY(:);
            
            [dWdmb,dWY] = net.JTmv(W,mb,Y0,tmp);
            t2 = dmb(:)'*dWdmb(:) + dY0(:)'*dWY(:);
            
            fprintf('adjoint test: t1=%1.2e\tt2=%1.2e\terr=%1.2e\n',t1,t2,abs(t1-t2)/abs(t1));
        end
    end
    
end

