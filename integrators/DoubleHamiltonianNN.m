classdef DoubleHamiltonianNN < abstractMeganetElement
    % Double Layer Hamiltonian block
    %
    % Z_k+1 = Z_k - h*layer1(Y_k, theta_1),  
    % Y_k+1 = Y_k + h*layer2(Z_k+1,theta_2) 
    %
    % The input features are divided into Y and Z here based on the sizes 
    % of the layers. The layers do not have to have the same size.
    %
    % References:
    %
    % Chang B, Meng L, Haber E, Ruthotto L, Begert D, Holtham E: 
    %      Reversible Architectures for Arbitrarily Deep Residual Neural Networks, 
    %      AAAI Conference on Artificial Intelligence 2018
    
    properties
        layer1
        layer2
        nt
        h
        useGPU
        precision
    end
    
    methods
        function this = DoubleHamiltonianNN(layer1,layer2,nt,h,varargin)
            if nargin==0
                this.runMinimalExample;
                return;
            end
            useGPU    = [];
            precision = [];
            for k=1:2:length(varargin)     % overwrites default parameter
                eval([varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            if not(isempty(useGPU))
                layer1.useGPU = useGPU;
                layer2.useGPU = useGPU;
            end
            if not(isempty(precision))
                layer1.precision = precision;
                layer2.precision = precision;
            end
            if any(sizeFeatOut(layer1)~=sizeFeatIn(layer2)) || any(sizeFeatIn(layer1)~=sizeFeatOut(layer2))
                error('number of input and output features must agree');
            end
            this.layer1   = layer1;
            this.layer2   = layer2;
            this.nt       = nt;
            this.h        = h;
        end
        
        function n = nTheta(this)
            n = this.nt*(nTheta(this.layer1)+ nTheta(this.layer2));
        end
        function n = sizeFeatIn(this)
            n1 = sizeFeatIn(this.layer1);
            n2 = sizeFeatIn(this.layer2);
            if (numel(n1) > 2) && (numel(n2) > 2)
                % convolution layer. add chanels together
                n = n1;
                n(3) = n1(3) + n2(3);
            else
                n = n1+n2;            
            end
        end
        function n = sizeFeatOut(this)
            n = sizeFeatIn(this);
        end

        
        function theta = initTheta(this)
            theta = repmat([vec(initTheta(this.layer1));...
                            vec(initTheta(this.layer2))],this.nt,1);
        end
        
        function [net2,theta2] = prolongateWeights(this,theta)
            % piecewise linear interpolation of network weights 
            t1 = 0:this.h:(this.nt-1)*this.h;
            
            net2 = DoubleHamiltonianNN(this.layer1,this.layer2,2*this.nt,this.h/2,'useGPU',this.useGPU,'precision',this.precision);
          
            t2 = 0:net2.h:(net2.nt-1)*net2.h;
            
            theta2 = inter1D(theta,t1,t2);
        end
        
        function [th1,th2] = split(this,x)
           x   = reshape(x,[],this.nt);
           th1 = x(1:nTheta(this.layer1),:);
           th2 = x(nTheta(this.layer1)+1:end,:);
        end
        function [Y,Z] = splitData(this,X)
           n1 = sizeFeatIn(this.layer1);
           
           if (numel(n1) > 2)
                % convolution layers
                Y = X(:,:,1:n1(3),:);
                Z = X(:,:,n1(3)+1:end,:);
           else
               Y = X(1:n1,:);
               Z = X(n1+1:end,:);
           end
        end
        function X = unsplitData(this,Y,Z)
            n1 = sizeFeatIn(this.layer1);
            if (numel(n1) > 2)
                X = cat(3,Y,Z);
            else
               X = [Y;Z]; 
            end
        end

               % ------- forwardProp forward problems -----------
        function [X,tmp] = forwardProp(this,theta,X0,varargin)
            for k=1:2:length(varargin)     % overwrites default parameter
                eval([varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            
            [Y,Z] = splitData(this,X0);
            [th1,th2] = split(this,theta);
            
            for i=1:this.nt
                
                fZ = forwardProp(this.layer1,th1(:,i),Z);
                Y  = Y + this.h*fZ;
                
                fY = forwardProp(this.layer2,th2(:,i),Y);
                Z  = Z - this.h*fY;
            end
            tmp = {Y,Z,X0};
            X = unsplitData(this,Y,Z);
        end
        
        % -------- Jacobian matvecs ---------------
        function dX = JYmv(this,dX,theta,~,tmp)
            if isempty(dX) || (numel(dX)==0 && dX==0.0)
                dX     = zeros(size(Y),'like',Y);
                return
            end
            X0 = tmp{3};
            [Y,Z] = splitData(this,X0);
            [dY,dZ]   = splitData(this,dX);
            [th1,th2] = split(this,theta);
            for i=1:this.nt
                [fZ,tmp] = forwardProp(this.layer1,th1(:,i),Z,'storeInterm',1);
                dY = dY + this.h*JYmv(this.layer1,dZ,th1(:,i),Z,tmp);
                Y  = Y + this.h*fZ;
                
                [fY,tmp] = forwardProp(this.layer2,th2(:,i),Y);
                dZ = dZ - this.h*JYmv(this.layer2,dY,th2(:,i),Y,tmp);
                Z = Z - this.h*fY;
            end
            dX = unsplitData(this,dY,dZ);
        end
        
        function dX = Jmv(this,dtheta,dX,theta,~,tmp)
            if isempty(dX)
                dY = 0.0;
                dZ = 0.0;
            elseif numel(dX)>1
                [dY,dZ] = splitData(this,dX);
            end
            X0 = tmp{3};
            
            [Y,Z] = splitData(this,X0);
            [th1,th2]   = split(this,theta);
            [dth1,dth2] = split(this,dtheta);
            for i=1:this.nt
                 [fZ,tmp] = forwardProp(this.layer1,th1(:,i),Z,'storeInterm',1);
                 dY = dY + this.h*Jmv(this.layer1,dth1(:,i),dZ,th1(:,i),Z,tmp);
                 Y = Y + this.h*fZ;
                 
                 [fY,tmp] = forwardProp(this.layer2,th2(:,i),Y);
                 dZ = dZ - this.h*Jmv(this.layer2,dth2(:,i),dY,th2(:,i),Y,tmp);
                 Z = Z - this.h*fY;
            end
            dX = unsplitData(this,dY,dZ);
        end
        
        % -------- Jacobian' matvecs ----------------
        function W = JYTmv(this,W,theta,X0,tmp)
            % nex = sizeLastDim(Y);
            if isempty(W)
                WY = 0;
                WZ = 0;
            elseif not(isscalar(W))
                [WY,WZ] = splitData(this,W);
            end
            
            Y = tmp{1};
            Z = tmp{2};
            [th1,th2]  = split(this,theta);
            
            for i=this.nt:-1:1
                [fY,tmp] = forwardProp(this.layer2,th2(:,i),Y);
                dWY = JYTmv(this.layer2,WZ,th2(:,i),Y,tmp);
                WY  = WY - this.h*dWY;
                Z = Z + this.h*fY;
                
                [fZ,tmp] = forwardProp(this.layer1,th1(:,i),Z,'storeInterm',1);
                dWZ = JYTmv(this.layer1,WY,th1(:,i),Z,tmp);
                WY  = WZ + this.h*dWZ;
                Y = Y - this.h*fZ;
            end
            W = unsplitData(this,WY,WZ);
        end
        
        function [dtheta,W] = JTmv(this,W,theta,X0,tmp,doDerivative)
            if not(exist('doDerivative','var')) || isempty(doDerivative)
               doDerivative =[1;0]; 
            end
            
            if isempty(W) || numel(W)==1
                WY = 0;
                WZ = 0;
            elseif not(isscalar(W))
                [WY,WZ] = splitData(this,W);
            end
            
            Y = tmp{1};
            Z = tmp{2};
            [th1,th2]   = split(this,theta);
            [dth1,dth2] = split(this,0*theta);
            
            for i=this.nt:-1:1
                [fY,tmp] = forwardProp(this.layer2,th2(:,i),Y);
                [dt2,dWY] = JTmv(this.layer2,WZ,th2(:,i),Y,tmp);
                WY  = WY - this.h*dWY;
                dth2(:,i) = -this.h*dt2;
                Z = Z + this.h*fY;
                
                [fZ,tmp] = forwardProp(this.layer1,th1(:,i),Z,'storeInterm',1);
                [dt1,dWZ] = JTmv(this.layer1,WY,th1(:,i),Z,tmp);
                WZ  = WZ + this.h*dWZ;
                dth1(:,i) = this.h*dt1;
                Y = Y - this.h*fZ;
            end
            dtheta = vec([dth1;dth2]);
            
            W = unsplitData(this,WY,WZ);
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
                this.layer1.useGPU  = value;
                this.layer2.useGPU  = value;
            end
        end
        function this = set.precision(this,value)
            if not(strcmp(value,'single') || strcmp(value,'double'))
                error('precision must be single or double.')
            else
                this.layer1.precision = value;
                this.layer2.precision = value;
            end
        end
        function useGPU = get.useGPU(this)
            useGPU = this.layer1.useGPU;
        end
        function precision = get.precision(this)
            precision = this.layer1.precision;
        end
        
        function runMinimalExample(~)
            
            layer = doubleSymLayer(dense([2,2]));
%             layer = singleLayer(affineTrafo(dense([2,2])));
            net   = DoubleHamiltonianNN(layer,layer,10,.1);
            Y = [1;1;1;1];
            theta =  [vec([2 1;-1 2]);vec([2 -1;1 2])];
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

