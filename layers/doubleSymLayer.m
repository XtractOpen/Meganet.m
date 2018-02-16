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
        nLayer         % normalization layer
        Bin            % Bias inside the nonlinearity
        Bout           % bias outside the nonlinearity
        useGPU
        precision
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
            nLayer    = [];
            activation = @tanhActivation;
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
                Bin = zeros(nFeatOut(K),0);
            end
            if not(exist('Bout','var')) || isempty(Bout)
                Bout = zeros(nFeatIn(K),0);
            end
            if not(isempty(nLayer)) && nFeatIn(nLayer) ~= nFeatOut(this.K)
                error('input dimension of normalization layer must match output dimension of K')
            end
            this.nLayer = nLayer;
            
            [this.Bin,this.Bout] = gpuVar(this.K.useGPU,this.K.precision,Bin,Bout);
        end
        
        function [th1,th2,th3,th4] = split(this,theta)
            th1 = theta(1:nTheta(this.K));
            cnt = numel(th1);
            th2 = theta(cnt+(1:size(this.Bin,2)));
            cnt = cnt + numel(th2);
            th3 = theta(cnt+size(this.Bin,2)+(1:size(this.Bout,2)));
            cnt = cnt + numel(th3);
            th4 = [];
            if not(isempty(this.nLayer))
                th4 = theta(cnt+1:end);
            end
        end
        
        function [Z,QZ,tmp] = apply(this,theta,Y,varargin)
            QZ =[];
            nex        = numel(Y)/nFeatIn(this);
            Y          = reshape(Y,[],nex);
            storedAct  = (nargout>1);
            
            [theta1,theta2,theta3,th4] = split(this,theta);
            Kop    = getOp(this.K,theta1);
            KY     = Kop*Y;
            tmp    = KY;
            if not(isempty(this.nLayer))
                KY = apply(this.nLayer,th4,KY);
            end
            Yt     = KY + this.Bin*theta2;
            Z      = this.activation(Yt,'doDerivative',storedAct);
            Z      = -(Kop'*Z) + this.Bout*theta3;
        end
        
        function n = nTheta(this)
            n = nTheta(this.K) + size(this.Bin,2)+ size(this.Bout,2); 
            if not(isempty(this.nLayer))
                n = n + nTheta(this.nLayer);
            end
        end
        
        function n = nFeatIn(this)
            n = nFeatIn(this.K);
        end
        
        function n = nFeatOut(this)
            n = nFeatIn(this.K);
        end
        
        function n = nDataOut(this)
            n = nFeatIn(this);
        end
        
        function theta = initTheta(this)
            theta = [vec(initTheta(this.K)); ...
                     0.1*ones(size(this.Bin,2),1);...
                     0.1*ones(size(this.Bout,2),1)];
           if not(isempty(this.nLayer))
               theta = [theta; initTheta(this.nLayer)];
           end
          
        end
        
        function dY = Jthetamv(this,dtheta,theta,Y,tmp)
            
            [th1, th2,~,th4]  = split(this,theta);
            [dth1,dth2,dth3,dth4] = split(this,dtheta);
            KY = tmp;
            if not(isempty(this.nLayer))
                [KY,~,tmpNL] = apply(this.nLayer,th4,KY);
            end
            [A,dA] = this.activation( KY + this.Bin*th2);
            
            Kop    = getOp(this.K,th1);
            dKop   = getOp(this.K,dth1);
            dY     = dKop*Y;
            if not(isempty(this.nLayer))
                dY = Jmv(this.nLayer,dth4,dY,th4,tmp,tmpNL);
            end
            dY     = dY + this.Bin*dth2;
            
            dY = -(Kop'*(dA.*dY) + dKop'*A) + this.Bout*dth3;
        end
        
        function dZ = JYmv(this,dY,theta,~,tmp)
            
            [th1, th2,~,th4]  = split(this,theta);
            KY = tmp;
            if not(isempty(this.nLayer))
                [KY,~,tmpNL] = apply(this.nLayer,th4,KY);
            end
            [~,dA] = this.activation( KY + this.Bin*th2);
            
            nex = numel(dY)/nFeatIn(this);
            dY  = reshape(dY,[],nex);
            
            Kop = getOp(this.K,th1);
            dY = Kop*dY;
            if not(isempty(this.nLayer))
                dY = JYmv(this.nLayer,dY,th4,tmp,tmpNL);
            end
            dZ = -(Kop'*(dA.*dY));
            
        end
        
        function dY = Jmv(this,dtheta,dY,theta,Y,tmp)
            [th1, th2,~,th4]  = split(this,theta);
            [dth1,dth2,dth3,dth4] = split(this,dtheta);
            KY = tmp;
            if not(isempty(this.nLayer))
                [KY,~,tmpNL] = apply(this.nLayer,th4,KY);
            end
            [A,dA] = this.activation( KY + this.Bin*th2);
            nex = numel(Y)/nFeatIn(this);
            

            Kop    = getOp(this.K,th1);
            dKop   = getOp(this.K,dth1);
            if numel(dY)>1
                dY  = reshape(dY,[],nex);
                KdY = Kop*dY;
            else
                KdY = 0;
            end
            dY = dKop*Y+KdY;
            if not(isempty(this.nLayer))
                dY = Jmv(this.nLayer,dth4,dY,th4,tmp,tmpNL);
            end
            dY     = dY + this.Bin*dth2;
            
            dY = -(Kop'*(dA.*dY) + dKop'*A) + this.Bout*dth3;
        end
        
        
        function dtheta = JthetaTmv(this,Z,~,theta,Y,tmp)
            [th1, th2,~,th4]  = split(this,theta);
            KY = tmp;
            if not(isempty(this.nLayer))
                [KY,~,tmpNL] = apply(this.nLayer,th4,KY);
            end
            [A,dA] = this.activation( KY + this.Bin*th2);
            
            
            nex       = numel(Y)/nFeatIn(this);
            Z         = reshape(Z,[],nex);
            Kop       = getOp(this.K,th1);
            
            dth3      = vec(sum(this.Bout'*Z,2));
            dAZ       = dA.*(Kop*Z);
            dth2      = vec(sum(this.Bin'*dAZ,2));
            if not(isempty(this.nLayer))
               [dth4,dAZ] = JTmv(this.nLayer,dAZ,[],th4,tmp,tmpNL); 
            else
               dth4 = [];
            end
            dth1      = JthetaTmv(this.K,dAZ,[],Y);
            dth1      = dth1 + JthetaTmv(this.K,A,[],Z);
            dtheta    = [-dth1(:); -dth2(:); dth3(:); -dth4(:)];
        end
        
        function dY = JYTmv(this,Z,~,theta,Y,tmp)
            [th1, th2,~,th4]  = split(this,theta);
            KY = tmp;
            if not(isempty(this.nLayer))
                [KY,~,tmpNL] = apply(this.nLayer,th4,KY);
            end
            [~,dA] = this.activation( KY + this.Bin*th2);
            
            nex       = numel(Y)/nFeatIn(this);
            Z         = reshape(Z,[],nex);
            Kop       = getOp(this.K,th1);
            
            dAZ       = dA.*(Kop*Z);
            %dAZ       = dA.*Z;
            if not(isempty(this.nLayer))
                dAZ = JYTmv(this.nLayer,dAZ,[],th4,tmp,tmpNL);
            end
            dY  = -(Kop'*dAZ);
        end
        
        function [dtheta,dY] = JTmv(this,Z,~,theta,Y,tmp,doDerivative)
            if not(exist('doDerivative','var')) || isempty(doDerivative)
                doDerivative =[1;0];
            end
            [th1, th2,~,th4]  = split(this,theta);
            KY = tmp;
            if not(isempty(this.nLayer))
                [KY,~,tmpNL] = apply(this.nLayer,th4,KY);
            end
            [A,dA] = this.activation( KY + this.Bin*th2);
            
            
            
            dY = [];
            nex       = numel(Y)/nFeatIn(this);
            Z         = reshape(Z,[],nex);
            Kop       = getOp(this.K,th1);
            
            dth3      = vec(sum(this.Bout'*Z,2));
            dAZ       = dA.*(Kop*Z);
            dth2      = vec(sum(this.Bin'*dAZ,2));
            if not(isempty(this.nLayer))
               [dth4,dAZ] = JTmv(this.nLayer,dAZ,[],th4,tmp,tmpNL); 
            else
               dth4 = [];
            end
            dth1      = JthetaTmv(this.K,dAZ,[],Y);
            
            dth1      = dth1 + JthetaTmv(this.K,A,[],Z);
            dtheta    = [-dth1(:); -dth2(:); dth3(:);-dth4(:)];
            
            if nargout==2 || doDerivative(2)==1
                dY  = -(Kop'*dAZ);
            end
            if nargout==1 && all(doDerivative==1)
                dtheta = [dtheta(:);dY(:)];
            end
            
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


