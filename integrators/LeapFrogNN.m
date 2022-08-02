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
        useGPU
        precision
        A       
    end
    
    methods
        function this = LeapFrogNN(layer,nt,h,varargin)
            if nargin==0
                this.runMinimalExample;
                return;
            end
            useGPU = [];
            precision = [];
            A = speye(nt); 
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
            this.nt    = nt;
            this.h     = h;
            this.A     = A;
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
%             theta = [];
%             for k=1:this.nt
%                 theta = [theta; vec(initTheta(this.layer))];
%             end
           % theta = repmat(vec(initTheta(this.layer)),this.nt,1);
            theta = repmat(vec(initTheta(this.layer)),size(this.A,1),1);
        end
        
        function [net2,theta2] = prolongateWeights(this,theta) % need to go back to this
            % piecewise linear interpolation of network weights 
            t1 = 0:this.h:(this.nt-1)*this.h;
            
            net2 = LeapFrogNN(this.layer,2*this.nt,this.h/2,'useGPU',this.useGPU,'precision',this.precision);
          
            t2 = 0:net2.h:(net2.nt-1)*net2.h;
            
            theta2 = inter1D(theta,t1,t2);
        end
        
        %% ------- forwardProp forward problems -----------
        function [Y,tmp] = forwardProp(this,theta,Y,varargin)
            for k=1:2:length(varargin)     % overwrites default parameter
                eval([varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            
           
            theta = reshape(theta,nTheta(this.layer),[]); 
            Theta = theta*this.A;  
            Yin = Y;
            Yold = Y;
            
            for i=1:this.nt
                Z     = forwardProp(this.layer,Theta(:,i),Y); 
                Ytemp = Y;
                Y     =  2*Y - Yold + this.h^2 * Z;
                Yold  = Ytemp;
            end
            tmp = {Yin, Y,Yold};
        end
        
        %% -------- Jacobian matvecs ---------------
        function dY = JYmv(this,dY,theta,~,tmp)
            if isempty(dY)
                dY = 0.0;
            end
            Y = tmp{1};
            dYold = dY;
            Yold  = Y;

            theta = reshape(theta,nTheta(this.layer),[]); 
            Theta = theta*this.A; 
            
            for i=1:this.nt
                % evaluate layer
                [Z,tmp]     = forwardProp(this.layer,Theta(:,i),Y,'storeInterm',1);
                
                % update dY
                dYtemp = dY;
                dY     = 2*dY - dYold + this.h^2* JYmv(this.layer,dY,Theta(:,i),Y,tmp);
                dYold  = dYtemp;

                % update Y
                Ytemp = Y;
                Y     =  2*Y - Yold + this.h^2 * Z;
                Yold  = Ytemp;
            end
        end
              

        function dY = Jmv(this,dtheta,dY,theta,~,tmp)
            if isempty(dY)
                dY = 0.0;
            end
            
            Y = tmp{1};
            dYold = dY;
            Yold  = Y;
            
            theta = reshape(theta,nTheta(this.layer),[]); 
            Theta = theta*this.A;  
            dtheta = reshape(dtheta,nTheta(this.layer),[]); 
            dTheta = dtheta*this.A;  
            
            for i=1:this.nt
                % evaluate layer
                [Z,tmp]     = forwardProp(this.layer,Theta(:,i),Y,'storeInterm',1);
                
                % update dY
                dYtemp = dY;
                dY = 2*dY - dYold + this.h^2* Jmv(this.layer,dTheta(:,i),dY,Theta(:,i),Y,tmp);
                dYold = dYtemp;
                
                % update Y
                Ytemp = Y;
                Y     =  2*Y - Yold + this.h^2 * Z;
                Yold  = Ytemp;
            end
        end
        
        %% -------- Jacobian' matvecs ----------------
        
        function W = JYTmv(this,W,theta,~,tmp)
            % call JYTmv (saving computations of the derivatives w.r.t.
            % theta)
            if isempty(W)
                W = 0;
            end
            
            % get last two time points
            Yold = tmp{2};
            Y    = tmp{3};
            
            theta = reshape(theta,nTheta(this.layer),[]); 
            Theta = theta*this.A;
            
            Wold = 0*W;
            for i=this.nt:-1:1
                % evaluate layer
                [Z,tmp]     = forwardProp(this.layer,Theta(:,i),Y,'storeInterm',1);
                
                % compute Jacobian matvecs
                dW = JYTmv(this.layer,W,Theta(:,i),Y,tmp);
                Wtemp = W;
                if i>1
                    W     = 2*W - Wold + this.h^2*dW;
                else
                    % note that we use homogeneous Neumann boundary
                    % conditions
                    W     =(W-Wold)+this.h^2*dW;
                end
                Wold  = Wtemp;
                
                % update Y
                Ytemp = Y;
                Y     =  2*Y - Yold + this.h^2 * Z;
                Yold  = Ytemp;
            end
        end
        
        function [dtheta,W] = JTmv(this,W,theta,~,tmp,doDerivative,varargin)
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
            
            % get last two time points
            Yold = tmp{2};
            Y    = tmp{3};
            
            theta = reshape(theta,nTheta(this.layer),[]); % number of rows is number of weights layer needs, number of cols = how many coeffs
            Theta = theta*this.A;            

            if reduceDim
                dTheta = 0*Theta;
            else
                dTheta = zeros(size(Theta,1),size(Theta,2),sizeLastDim(Y),'like',Theta);
            end
            Wold   = 0*W;
            for i=this.nt:-1:1
                
                % evaluate layer
                [Z,tmp]     = forwardProp(this.layer,Theta(:,i),Y,'storeInterm',1);
                
                % compute Jacobian matvecs
                [dThetai,dW] = JTmv(this.layer,W,Theta(:,i),Y,tmp,[],'reduceDim',reduceDim);
                if reduceDim
                    dTheta(:,i)  = this.h^2*dThetai;
                else
                    dTheta(:,i,:) = this.h^2*dThetai;
                end
                
                Wtemp = W;
                if i>1
                    W     = 2*W - Wold + this.h^2*dW;
                else
                    % note that we use homogeneous Neumann boundary
                    % conditions
                    W     =(W-Wold)+this.h^2*dW;
                end
                Wold  = Wtemp;
                
                % update Y
                Ytemp = Y;
                Y     =  2*Y - Yold + this.h^2 * Z;
                Yold  = Ytemp;
            end
            if reduceDim
                dtheta = vec(dTheta*this.A');
            else
                dtheta = pagemtimes(dTheta,'none',full(this.A),'transpose');
                dtheta = reshape(dtheta,[],sizeLastDim(Y));
            end

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
        
        %% ------- functions for handling GPU computing and precision ----
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
            nex = 500;
            nK  = [2 2];
            
            D   = dense(nK);
            S   = doubleSymLayer(D);
            S.activation = @reluActivation;
            net = LeapFrogNNrev(S,3,1);
            mb  = randn(nTheta(net),1);
            
            Y0  = randn(nK(2),nex);
            [Ydata,tmp]   = net.forwardProp(mb,Y0);
            dmb = reshape(randn(size(mb)),[],net.nt);
            dY0  = randn(size(Y0));
            
            dY = net.Jmv(dmb(:),dY0,mb,Y0,tmp);
            for k=1:30
                hh = 2^(-k);
                
                Yt = net.forwardProp(mb+hh*dmb(:),Y0+hh*dY0);
                
                E0 = norm(Yt(:)-Ydata(:));
                E1 = norm(Yt(:)-Ydata(:)-hh*dY(:));
                
                fprintf('h=%1.2e\tE0=%1.2e\tE1=%1.2e\n',hh,E0,E1);
            end
            
            W = randn(size(Ydata));
            t1  = W(:)'*dY(:);
            
            [dWdmb,dWY] = net.JTmv(W,mb,Y0,tmp);
            t2 = dmb(:)'*dWdmb(:) + dY0(:)'*dWY(:);
            
            fprintf('adjoint test: t1=%1.2e\tt2=%1.2e\terr=%1.2e\n',t1,t2,abs(t1-t2));
        end
    end
    
end

