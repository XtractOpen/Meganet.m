classdef doubleLayer < abstractMeganetElement
    % classdef doubleLayer < abstractMeganetElement
    %
    % implementation of double layer
    %
    % Y(theta) = act2 (K2(th2)*act1(K1(th1)+Bin1*th3)) + Bin2*th4) + Bout*th5
    
    properties
        K1      % inner kernel
        K2      % outer kernel
        activation1  % inner activation
        activation2  % outer activation
        Bin1         % bias inside first activation
        Bin2         % bias inside second activation
        Bout         % outer bias
        useGPU       
        precision
    end
    methods
        function this = doubleLayer(K1,K2,varargin)
            if nargin==0
%                 this.runMinimalExample;
                help(mfilename);
                return;
            end
            useGPU = [];
            precision = [];
            Bin1 = [];
            Bin2 = [];
            Bout = [];
            activation1 = [];
            activation2 = [];
            for k=1:2:length(varargin)     % overwrites default parameter
               eval([varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            if not(isempty(useGPU))
                K1.useGPU = useGPU;
                K2.useGPU = useGPU;
            end
            if not(isempty(precision))
                K1.precision = precision;
                K2.precision = precision;
            end
            if isempty(Bin1)
                Bin1 = zeros(nFeatOut(K1),0);
            end
            this.Bin1 = gpuVar(K1.useGPU, K1.precision, Bin1);
                
            if isempty(Bin2)
                Bin2 = zeros(nFeatOut(K2),0);
            end
            this.Bin2 = gpuVar(K2.useGPU, K2.precision, Bin2);
            
            if isempty(Bout)
                Bout = zeros(nFeatOut(K2),0);
            end
            this.Bout = gpuVar(K2.useGPU, K2.precision, Bout);
            
            if isempty(activation1)
                activation1  = @tanhActivation;
            end
            this.activation1 = activation1;
            
            if isempty(activation2)
                activation2  = @tanhActivation;
            end
            this.activation2 = activation2;
            if nFeatOut(K1) ~= nFeatIn(K2)
                error('%s - number of output features of first trafo (%d) must match input of second trafo (%d)',...
                    mfilename,nFeatOut(K1), nFeatIn(K2));
            end
            this.K1      = K1;
            this.K2      = K2;
        end
        
        % ------------ counting ----------
        function [th1,th2,th3,th4,th5] = split(this,x)
            th1 = []; th2 = []; th3 = []; th4 = []; th5 = [];
            if not(isempty(x))
                cnt = 1;
                nk1  = nTheta(this.K1);
                th1 = x(cnt:nk1);
                cnt = cnt + nk1;
                
                nk2 = nTheta(this.K2);
                th2 = x(cnt:cnt+nk2-1);
                cnt = cnt + nk2;
                
                nk3 = size(this.Bin1,2);
                th3 = x(cnt:cnt+nk3-1);
                cnt = cnt+nk3;
                
                nk4 = size(this.Bin2,2);
                th4 = x(cnt:cnt+nk4-1);
                cnt = cnt+nk4;
                
                nk5 = size(this.Bout,2);
                th5 = x(cnt:cnt+nk5-1);
            end
        end
        function n = nTheta(this)
            n = nTheta(this.K1) + nTheta(this.K2)...
                 + size(this.Bin1,2) + size(this.Bin2,2) + size(this.Bout,2);
        end
        function n = nFeatIn(this)
            n = nFeatIn(this.K1);
        end
        
        function n = nFeatOut(this)
            n = nFeatOut(this.K2);
        end
        function n = nDataOut(this)
            n = nFeatOut(this);
        end
        
        function theta = initTheta(this)
            theta = [vec(initTheta(this.K1)); vec(initTheta(this.K2)); ...
                     0.1*ones(size(this.Bin1,2),1); 0.1*ones(size(this.Bin2,2),1); 0.1*ones(size(this.Bout,2),1)];
        end
        
        % ------- apply forward model ----------
        function [Ydata,Y,tmp] = apply(this,theta,Y,varargin)
            doDerivative  = (nargout>1);
            for k=1:2:length(varargin)     % overwrites default parameter
                eval([varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            nex = numel(Y)/nFeatIn(this);
            Y   = reshape(Y,[],nex);
            
            tmp        = cell(1,3);
            [th1,th2,th3,th4,th5] = this.split(theta);
            
            T          = getOp(this.K1,th1)* Y + this.Bin1*th3;
            [Z1,tmp{2}] = this.activation1(T,'doDerivative',doDerivative);
            T         = getOp(this.K2, th2)* Z1 + this.Bin2*th4; 
            [Y,tmp{3}] = this.activation2(T,'doDerivative',doDerivative);
            Y      = Y+  this.Bout*th5;
            tmp{1} = Z1; 
            Ydata = Y;
        end
        
        % ----------- Jacobian matvecs -----------
        function [dZ] = Jthetamv(this,dtheta,theta,Y,dA)
            if isempty(dtheta); dZ = []; return; end
            nex = numel(Y)/nFeatIn(this);
            Y  = reshape(Y,[],nex);
            
            [dth1,dth2,dth3,dth4,dth5] = this.split(dtheta);
            [~, th2, th3, th4, th5]  = this.split(theta);
            
            Z = dA{1}; dA1 = dA{2}; dA2 = dA{3};
            
            dZ = 0.0;
            T  = dA1.*(getOp(this.K1,dth1)*Y + this.Bin1*dth3);
            dZ = dZ + dA2 .* (getOp(this.K2,th2)*T  + this.Bin2*dth4);
            dZ = dZ + dA2 .* (getOp(this.K2, dth2)*Z);
            dZ = dZ + this.Bout*dth5;
        end
        
        function [dZ] = JYmv(this,dY,theta,~,dA)
            nex = numel(dY)/nFeatIn(this);
            dY  = reshape(dY,[],nex);
            if not(isempty(dY)) && (not(isscalar(dY) && dY==0))
                dA1 = dA{2}; dA2 = dA{3};
                [th1, th2] = this.split(theta);
                dZ = dA2.*(getOp(this.K2,th2)* (dA1.*(getOp(this.K1,th1)* dY)));
            end
        end
        
        function [dZ] = Jmv(this,dtheta,dY,theta,Y,tmp)
            if not(isempty(dtheta))
                dZ = Jthetamv(this,dtheta,theta,Y,tmp);
            else
                dZ = 0.0;
            end
            if not(isempty(dY)) && (not(isscalar(dY) && dY==0))
                dZ = dZ + JYmv(this,dY,theta,Y,tmp);
            end
        end
        
        % ----------- Jacobian' matvecs ----------
        
        function dth = JthetaTmv(this,W,~,theta,Y,dA)
            nex        = numel(W)/nFeatOut(this);
            W          = reshape(W,[],nex);
            Z = dA{1}; dA1 = dA{2}; dA2 = dA{3};
            
            dth5 = vec(sum(this.Bout'*W,2));
            [th1, th2] = this.split(theta);
            
            dAZ1 = dA1.*(getOp(this.K2,th2)'*(dA2.*W));
            dth1 = JthetaTmv(this.K1,dAZ1,th1,Y);
            dth3      = vec(sum(this.Bin1'*reshape(dAZ1,[],nex),2));
     
            dAZ2 = dA2.*W;
            dth2 = JthetaTmv(this.K2,dAZ2,th2,Z);
            dth4 = vec(sum(this.Bin2'*reshape(dAZ2,[],nex),2));
            dth = [dth1(:); dth2(:); dth3(:); dth4(:); dth5(:)];
        end
        
        function dY = JYTmv(this,Z,~,theta,~,dA)
            [th1, th2] = this.split(theta);
            nex = numel(Z)/nFeatOut(this);
            Z   = reshape(Z,[],nex);
            dA1 = dA{2}; dA2 = dA{3};
            dY  = getOp(this.K1,th1)'*(dA1.*(getOp(this.K2,th2)'*(dA2.*Z)));
        end
        
        function [dth,dY] = JTmv(this,Z,~,theta,Y,tmp,doDerivative)
            if not(exist('doDerivative','var')) || isempty(doDerivative)
               doDerivative =[1;0]; 
            end
            nex       = numel(Y)/nFeatIn(this);
            Z         = reshape(Z,[],nex);
            
            dY = [];
            
            [th1, th2] = this.split(theta);
            
            dA1 = tmp{2}; dA2 = tmp{3};
            
            dth  = JthetaTmv(this,Z,[],theta,Y,tmp);
            
            if nargout==2 || doDerivative(2)==1
                dY  = getOp(this.K1,th1)'* (dA1.*(getOp(this.K2,th2)'*(dA2.*Z)));
            end
            if nargout==1 && all(doDerivative==1)
                dth = [dth(:);dY(:)];
            end

        end
        
        % ------- functions for handling GPU computing and precision ---- 
        function this = set.useGPU(this,value)
            if (value~=0) && (value~=1)
                error('useGPU must be 0 or 1.')
            else
                this.K1.useGPU  = value;
                this.K2.useGPU  = value;
                this.Bin1  = gpuVar(this.useGPU, this.precision, this.Bin1);
                this.Bin2  = gpuVar(this.useGPU, this.precision, this.Bin2);
                this.Bout  = gpuVar(this.useGPU, this.precision, this.Bout);
            end
        end
        function this = set.precision(this,value)
            if not(strcmp(value,'single') || strcmp(value,'double'))
                error('precision must be single or double.')
            else
                this.K1.precision = value;
                this.K2.precision = value;
                this.Bin1  = gpuVar(this.useGPU, value, this.Bin1);
                this.Bin2  = gpuVar(this.useGPU, value, this.Bin2);
                this.Bout  = gpuVar(this.useGPU, value, this.Bout);
            end
        end
        function useGPU = get.useGPU(this)
            useGPU = this.K1.useGPU;
            useGPU2 = this.K2.useGPU;
            if useGPU~=useGPU2
                error('both transformations need to be on GPU or CPU')
            end
        end
        function precision = get.precision(this)
            precision = this.K1.precision;
            precision2 = this.K2.precision;
            if not(strcmp(precision,precision2))
                error('precision of transformations must match')
            end
        end


    end
end


