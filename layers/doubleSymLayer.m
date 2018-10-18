classdef doubleSymLayer < abstractMeganetElement
    % classdef doubleSymLayer < abstractMeganetElement
    %
    % Implementation of symmetric double layer model
    %
    % Y(theta,Y0) = K(th1)'(activation( K(th1)*Y0 + trafo.Bin*th2))) + trafo.Bout*th3
    %
    % Chang B, Meng L, Haber E, Ruthotto L, Begert D, Holtham E: 
    %      Reversible Architectures for Arbitrarily Deep Residual Neural Networks, 
    %      AAAI Conference on Artificial Intelligence 2018
    
    
    properties
        activation     % activation function
        K              % Kernel model, e.g., convMod
        nLayer1        % inner normalization layer
        nLayer2        % outer normalization layer
        Bin            % Bias inside the nonlinearity
        Bout           % bias outside the nonlinearity
        useGPU
        precision
        storeInterm   % flag for storing intermediates
    end
    methods
        function this = doubleSymLayer(K,varargin)
            if nargin==0
                help(mfilename)
                return;
            end
            useGPU    = [];
            precision = [];
            Bout      = [];
            Bin       = [];
            nLayer1    = [];
            nLayer2    = [];
            activation = @tanhActivation;
            storeInterm=0;
            for k=1:2:length(varargin)     % overwrites default parameter
                eval([varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            if not(isempty(useGPU))
                K.useGPU = useGPU;
            end
            if not(isempty(precision))
                K.precision = precision;
            end
            
            
            this.activation = activation;
            this.K = K;
            if not(exist('Bin','var')) || isempty(Bin)
                Bin = zeros([sizeFeatOut(K),0]);
            end
            if not(exist('Bout','var')) || isempty(Bout)
                Bout = zeros([sizeFeatIn(K),0]);
            end
            if not(isempty(nLayer1)) && any(sizeFeatIn(nLayer1) ~= sizeFeatOut(this.K))
                error('input dimension of normalization layer must match output dimension of K')
            end
            this.nLayer1 = nLayer1;
            
            if not(isempty(nLayer2)) && any(sizeFeatIn(nLayer2) ~= sizeFeatIn(this.K))
                error('input dimension of normalization layer must match output dimension of K')
            end
            this.nLayer2 = nLayer2;
            
            this.storeInterm=storeInterm;
            
            [this.Bin,this.Bout] = gpuVar(this.K.useGPU,this.K.precision,Bin,Bout);
        end
        
        function [th1,th2,th3,th4,th5] = split(this,theta)
            th1 = theta(1:nTheta(this.K));
            cnt = numel(th1);
            th2 = theta(cnt+(1:sizeLastDim(this.Bin)));
            cnt = cnt + numel(th2);
            th3 = theta(cnt+sizeLastDim(this.Bin)+(1:sizeLastDim(this.Bout)));
            cnt = cnt + numel(th3);
            th4 = [];
            if not(isempty(this.nLayer1))
                th4 = theta(cnt+(1:nTheta(this.nLayer1)));
                cnt = cnt + numel(th4);
            end
            th5 = [];
            if not(isempty(this.nLayer2))
                th5 = theta(cnt+1:end);
            end
        end
        
        function [Z,QZ,tmp] = forwardProp(this,theta,Y,varargin)
            QZ =[]; tmp =  cell(1,2);
            nex        = numel(Y)/numelFeatIn(this);
            Y          = reshape(Y,[],nex);
            storedAct  = (nargout>1);
            
            [th1,th2,th3,th4,th5] = split(this,theta);
            Kop    = getOp(this.K,th1);
            Y     = Kop*Y;
            if this.storeInterm
                tmp{1}    = Y;
            end
            if not(isempty(this.nLayer1))
                Y = forwardProp(this.nLayer1,th4,Y);
            end
            if not(isempty(th2))
                Y     = Y + this.Bin*th2;
            end
            Z      = this.activation(Y,'doDerivative',storedAct);
            Z      = -(Kop'*Z);
            if not(isempty(this.nLayer2))
                if this.storeInterm
                    tmp{2} = Z;
                end
                Z = forwardProp(this.nLayer2,th5,Z);
            end
            if not(isempty(th3))
                Z      = Z + this.Bout*th3;
            end
        end

        
        function [A,dA,KY,KZ,tmpNL1,tmpNL2] = getTempsForSens(this,theta,Y,tmp)
            % re-computes temp variables needed for sensitivity computations
            %
            % Input:
            %   theta - current weights
            %   Y     - input features
            %   tmp    - either {K*Y,-K'*Z} stored during apply or empty
            %
            % Output:
            %   dA    - derivative of activation
            %   KY    - K(theta)*Y
            %   tmpNL - temp results of norm Layer
            
            nex = numel(Y)/numelFeatIn(this);
            tmpNL1 =[]; tmpNL2 = []; KZ = [];
            [th1, th2,~,th4,th5]  = split(this,theta);
            
            if not(this.storeInterm)
                Y = reshape(Y,[],nex);
                KY = getOp(this.K,th1)*Y;
            else
                KY = tmp{1};
            end
            
            if not(isempty(this.nLayer1))
                [KYn,~,tmpNL1] = forwardProp(this.nLayer1,th4,KY);
            else
                KYn = KY;
            end
            if not(isempty(th2))
                KYn = KYn + this.Bin*th2;
            end
            [A,dA] = this.activation( KYn );
            if not(isempty(this.nLayer2))
                if not(this.storeInterm)
                    KZ = - (getOp(this.K,th1)*A);
                else
                    KZ = tmp{2};
                end
                [~,~,tmpNL2] = forwardProp(this.nLayer2,th5,KZ);
            end
                
        end
        
        function n = nTheta(this)
            n = nTheta(this.K) + sizeLastDim(this.Bin)+ sizeLastDim(this.Bout); 
            if not(isempty(this.nLayer1))
                n = n + nTheta(this.nLayer1);
            end
            if not(isempty(this.nLayer2))
                n = n + nTheta(this.nLayer2);
            end
        end
        
        function n = sizeFeatIn(this)
            n = sizeFeatIn(this.K);
        end
        
        function n = sizeFeatOut(this)
            n = sizeFeatIn(this.K);
        end
        
        function n = nDataOut(this)
            n = sizeFeatIn(this);
        end
        
        function theta = initTheta(this)
            theta = [vec(initTheta(this.K)); ...
                     0.0*ones( sizeLastDim(this.Bin)  , 1) ;...
                     0.0*ones( sizeLastDim(this.Bout) , 1) ];
           if not(isempty(this.nLayer1))
               theta = [theta; initTheta(this.nLayer1)];
           end
           if not(isempty(this.nLayer2))
               theta = [theta; initTheta(this.nLayer2)];
           end
          
        end
        
        function dY = Jthetamv(this,dtheta,theta,Y,KY)
            
            [th1, ~,~,th4,th5]  = split(this,theta);
            [dth1,dth2,dth3,dth4,dth5] = split(this,dtheta);
            
            [A,dA,KY,KZ,tmpNL1,tmpNL2] = getTempsForSens(this,theta,Y,KY);
            
            Kop    = getOp(this.K,th1);
            dKop   = getOp(this.K,dth1);
            dY     = dKop*Y;
            if not(isempty(this.nLayer1))
                dY = Jmv(this.nLayer1,dth4,dY,th4,KY,tmpNL1);
            end
            dY     = dY + this.Bin*dth2;
            
            dY = -(Kop'*(dA.*dY) + dKop'*A);
            if not(isempty(this.nLayer2))
                dY = Jmv(this.nLayer2,dth5,dY,th5,KZ,tmpNL2);
            end
            dY = dY + this.Bout*dth3;
        end
        
        function dZ = JYmv(this,dY,theta,Y,KY)
            nex       = numel(Y)/numelFeatIn(this);
            Y   = reshape(Y,[],nex);
            [th1, ~,~,th4,th5]  = split(this,theta);
            
            [A,dA,KY,KZ,tmpNL1,tmpNL2] = getTempsForSens(this,theta,Y,KY);
            
            
            nex = numel(dY)/numelFeatIn(this);
            dY  = reshape(dY,[],nex);
            
            Kop = getOp(this.K,th1);
            dY = Kop*dY;
            if not(isempty(this.nLayer1))
                dY = JYmv(this.nLayer1,dY,th4,KY,tmpNL1);
            end
            dZ = -(Kop'*(dA.*dY));
            if not(isempty(this.nLayer2))
                dZ = Jmv(this.nLayer2,[],dZ,th5,KZ,tmpNL2);
            end
        end
        
        function dY = Jmv(this,dtheta,dY,theta,Y,KY)
            [th1, ~,~,th4,th5]  = split(this,theta);
            [dth1,dth2,dth3,dth4,dth5] = split(this,dtheta);
            
            [A,dA,KY,KZ,tmpNL1,tmpNL2] = getTempsForSens(this,theta,Y,KY);
            
            nex = numel(Y)/numelFeatIn(this);
            Kop    = getOp(this.K,th1);
            dKop   = getOp(this.K,dth1);
            if numel(dY)>1
                dY  = reshape(dY,[],nex);
                KdY = Kop*dY;
            else
                KdY = 0;
            end
            dY = dKop*Y+KdY;
            if not(isempty(this.nLayer1))
                dY = Jmv(this.nLayer1,dth4,dY,th4,KY,tmpNL1);
            end
            dY     = dY + this.Bin*dth2;
            
            dY = -(Kop'*(dA.*dY) + dKop'*A);
            if not(isempty(this.nLayer2))
                dY = Jmv(this.nLayer2,dth5,dY,th5,KZ,tmpNL2);
            end
            dY = dY + this.Bout*dth3;
        end
        
        
        function dtheta = JthetaTmv(this,Z,~,theta,Y,KY)
            [th1, ~,~,th4,th5]  = split(this,theta);
            [A,dA,KY,KZ,tmpNL1,tmpNL2] = getTempsForSens(this,theta,Y,KY);
            
            nex       = numel(Y)/numelFeatIn(this);
            Z         = reshape(Z,[],nex);
            Kop       = getOp(this.K,th1);
            
            dth3      = vec(sum(this.Bout'*Z,2));
            if not(isempty(this.nLayer2))
                [dth5,Z] = JTmv(this.nLayer2,Z,[],th5,KZ,tmpNL2);
            else
                dth5 = [];
            end
            dAZ       = dA.*(Kop*Z);
            dth2      = vec(sum(this.Bin'*dAZ,2));
            if not(isempty(this.nLayer1))
                [dth4,dAZ] = JTmv(this.nLayer1,dAZ,[],th4,KY,tmpNL1);
            else
                dth4 = [];
            end
            dth1      = JthetaTmv(this.K,dAZ,[],Y);
            dth1      = dth1 + JthetaTmv(this.K,A,[],Z);
            dtheta    = [-dth1(:); -dth2(:); dth3(:); -dth4(:);dth5(:)];
        end
        
        function dY = JYTmv(this,Z,~,theta,Y,KY)
            [th1, ~,~,th4,th5]  = split(this,theta);
            [A,dA,KY,KZ,tmpNL1,tmpNL2] = getTempsForSens(this,theta,Y,KY);
            
            nex       = numel(Y)/numelFeatIn(this);
            Z         = reshape(Z,[],nex);
            Kop       = getOp(this.K,th1);
            
            if not(isempty(this.nLayer2))
                Z = JYTmv(this.nLayer2,Z,[],th5,KZ,tmpNL2);
            end
            dAZ       = dA.*(Kop*Z);
            if not(isempty(this.nLayer1))
                dAZ = JYTmv(this.nLayer1,dAZ,[],th4,KY,tmpNL1);
            end
            dY  = -(Kop'*dAZ);
        end
        
        function [dtheta,dY] = JTmv(this,Z,~,theta,Y,KY,doDerivative)
            if not(exist('doDerivative','var')) || isempty(doDerivative)
                doDerivative =[1;0];
            end
            [th1, ~,~,th4,th5]  = split(this,theta);
            [A,dA,KY,KZ,tmpNL1,tmpNL2] = getTempsForSens(this,theta,Y,KY);
            
            dY = [];
            nex       = numel(Y)/numelFeatIn(this);
            Z         = reshape(Z,[],nex);
            Kop       = getOp(this.K,th1);
            
            dth3      = vec(sum(this.Bout'*Z,2));
            if not(isempty(this.nLayer2))
                [dth5,Z] = JTmv(this.nLayer2,Z,[],th5,KZ,tmpNL2);
            else
                dth5 = [];
            end
            
            dAZ       = dA.*(Kop*Z);
            dth2      = vec(sum(this.Bin'*dAZ,2));
            if not(isempty(this.nLayer1))
                [dth4,dAZ] = JTmv(this.nLayer1,dAZ,[],th4,KY,tmpNL1);
            else
                dth4 = [];
            end
            dth1      = JthetaTmv(this.K,dAZ,[],Y);
            
            dth1      = dth1 + JthetaTmv(this.K,A,[],Z);
            dtheta    = [-dth1(:); -dth2(:); dth3(:);-dth4(:);dth5(:)];
            
            if nargout==2 || doDerivative(2)==1
                dY  = -(Kop'*dAZ);
            end
            if nargout==1 && all(doDerivative==1)
                dtheta = [dtheta(:);dY(:)];
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
            
            thFine = theta;
            thFine(1:nTheta(this.K)) = prolongateConvStencils(this.K,theta(1:nTheta(this.K)),getRP);
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
            thCoarse = theta;
            thCoarse(1:nTheta(this.K)) = restrictConvStencils(this.K,theta(1:nTheta(this.K)),getRP);
        end
        
        % ------- functions for handling GPU computing and precision ----
        function this = set.useGPU(this,value)
            if (value~=0) && (value~=1)
                error('useGPU must be 0 or 1.')
            else
                this.K.useGPU  = value;
                [this.Bin,this.Bout] = gpuVar(value,this.precision,this.Bin,this.Bout);
            end
        end
        function this = set.precision(this,value)
            if not(strcmp(value,'single') || strcmp(value,'double'))
                error('precision must be single or double.')
            else
                this.K.precision = value;
                [this.Bin,this.Bout] = gpuVar(this.useGPU,value,this.Bin,this.Bout);
            end
        end
        function useGPU = get.useGPU(this)
            useGPU = this.K.useGPU;
            
        end
        function precision = get.precision(this)
            precision = this.K.precision;
        end

    end
end


