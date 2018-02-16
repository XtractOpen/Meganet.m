classdef doubleLayer < abstractMeganetElement
    % classdef doubleLayer < abstractMeganetElement
    %
    % implementation of double layer
    %
    % Y(theta) = act2 (K2(th2)*act1(K1(th1)+Bin1*th3)) + Bin2*th4) + Bout*th5
    
    properties
        K1      % inner kernel
        K2      % outer kernel
        nLayer1 % inner normalization
        nLayer2 % outer normalization
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
                help(mfilename);
                return;
            end
            useGPU = [];
            precision = [];
            Bin1 = [];
            Bin2 = [];
            Bout = [];
            nLayer1 = [];
            nLayer2 = [];
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
            this.nLayer1 = nLayer1;
            this.nLayer2 = nLayer2;
            
            if not(isempty(nLayer1)) && nFeatIn(nLayer1)~=nFeatOut(this.K1)
                error('input dimension of first normalization layer does not match output dimension of K1')
            end
            if not(isempty(nLayer2)) && nFeatIn(nLayer2)~=nFeatOut(this.K2)
                error('input dimension of second normalization layer does not match output dimension of K2')
            end
            
        end
        
        % ------------ counting ----------
        function [th1,th2,th3,th4,th5,th6,th7] = split(this,x)
            th1 = []; th2 = []; th3 = []; th4 = []; th5 = []; th6=[];th7=[];
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
                cnt = cnt+nk5;
                if not(isempty(this.nLayer1))
                    nk6 = nTheta(this.nLayer1);
                    th6 = x(cnt:cnt+nk6-1);
                    cnt = cnt + nk6;
                end
                
                if not(isempty(this.nLayer2))
                   nk7 = nTheta(this.nLayer2);
                   th7 = x(cnt:cnt+nk7-1);
                end
            end
        end
        function n = nTheta(this)
            n = nTheta(this.K1) + nTheta(this.K2)...
                 + size(this.Bin1,2) + size(this.Bin2,2) + size(this.Bout,2);
            if not(isempty(this.nLayer1))
                n = n+nTheta(this.nLayer1);
            end
            if not(isempty(this.nLayer2))
                n = n + nTheta(this.nLayer2);
            end
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
                 
            if not(isempty(this.nLayer1))
                theta = [theta; initTheta(this.nLayer1)];
            end
            if not(isempty(this.nLayer2))
                theta = [theta; initTheta(this.nLayer2)];
            end
        end
        
        % ------- apply forward model ----------
        function [Ydata,Y,tmp] = apply(this,theta,Y,varargin)
            
            nex = numel(Y)/nFeatIn(this);
            Y   = reshape(Y,[],nex);
            
            tmp        = cell(1,2);
            [th1,th2,th3,th4,th5,th6,th7] = this.split(theta);
            K1Y         = getOp(this.K1,th1)* Y;
            tmp{1} = K1Y;
            if not(isempty(this.nLayer1))
                K1Y = apply(this.nLayer1,th6,K1Y);
            end
            T          =  K1Y + this.Bin1*th3;
            Z1 = this.activation1(T);
            K2Z        = getOp(this.K2, th2)* Z1;
            tmp{2} = K2Z;
            if not(isempty(this.nLayer2))
                K2Z = apply(this.nLayer2,th7,K2Z);
            end
            T     = K2Z + this.Bin2*th4;
            Y     = this.activation2(T);
            Y     = Y+  this.Bout*th5;
            Ydata = Y;
        end
        
        % ----------- Jacobian matvecs -----------
        function [dZ] = Jthetamv(this,dtheta,theta,Y,dA)
            dZ = Jmv(this,dtheta,[],theta,Y,dA);
        end
        
        function [dZ] = JYmv(this,dY,theta,~,dA)
            nex = numel(dY)/nFeatIn(this);
            if not(isempty(dY)) && (not(isscalar(dY) && dY==0))
                % load temps and recompute activations
                dY  = reshape(dY,[],nex);
                [th1, th2,th3,th4,~,th6,th7] = this.split(theta);
                K1Y = dA{1};
                if not(isempty(this.nLayer1))
                    [K1Y,~,tmpNL1] = apply(this.nLayer1,th6,K1Y);
                end
                [~,dA1] = this.activation1(K1Y + this.Bin1*th3);
                K2Z = dA{2};
                if not(isempty(this.nLayer2))
                    [K2Z,~,tmpNL2] = apply(this.nLayer2,th7,K2Z);
                end
                [~,dA2] = this.activation2(K2Z + this.Bin2*th4);

                K1dY  = getOp(this.K1,th1)* dY;
                if not(isempty(this.nLayer1))
                    K1dY = JYmv(this.nLayer1,K1dY,th6,dA{1},tmpNL1);
                end
                K2dZ = getOp(this.K2,th2)* (dA1.*(K1dY));
                if not(isempty(this.nLayer2))
                    K2dZ = JYmv(this.nLayer2,K2dZ,th7,dA{2},tmpNL2);
                end
                dZ = dA2.*(K2dZ);
            end
        end
        
        function [dZ] = Jmv(this,dtheta,dY,theta,Y,dA)
            nex = numel(Y)/nFeatIn(this);
            Y  = reshape(Y,[],nex);
            
            [dth1,dth2,dth3,dth4,dth5,dth6,dth7] = this.split(dtheta);
            [th1, th2,th3,th4,~,th6,th7] = this.split(theta);
            
            % load temps and recompute activations
            K1Y = dA{1};
            if not(isempty(this.nLayer1))
                [K1Y,~,tmpNL1] = apply(this.nLayer1,th6,K1Y);
            end
            [A1,dA1] = this.activation1(K1Y + this.Bin1*th3);
            K2Z = dA{2};
            if not(isempty(this.nLayer2))
                [K2Z,~,tmpNL2] = apply(this.nLayer2,th7,K2Z);
            end
            [~,dA2] = this.activation2(K2Z + this.Bin2*th4);
            
            % now compute sensitivities
            dZ = 0.0;
            dK1Op = getOp(this.K1,dth1);
            
            dK2Op = getOp(this.K2,dth2);
            K2Op  = getOp(this.K2,th2);
            
            dK1dY = dK1Op*Y; 
            if not(isempty(dY)) || (numel(dY>1) && dY~=0)
                dY  = reshape(dY,[],nex);
                dK1dY = dK1dY + getOp(this.K1,th1)* dY;
            end
            if not(isempty(this.nLayer1))
                dK1dY = Jmv(this.nLayer1,dth6,dK1dY,th6,dA{1},tmpNL1);
            end
            d1  = dA1.*(dK1dY + this.Bin1*dth3);
            dK2Z = dK2Op*A1 + K2Op*d1;
            if not(isempty(this.nLayer2))
                dK2Z = Jmv(this.nLayer2,dth7,dK2Z,th7,dA{2},tmpNL2);
            end
            dZ = dZ + dA2 .* ( dK2Z + this.Bin2*dth4);
            dZ = dZ + this.Bout*dth5;
        end
        
        % ----------- Jacobian' matvecs ----------
        
        function [dth,dAZ1] = JthetaTmv(this,W,~,theta,Y,dA)
            nex        = numel(W)/nFeatOut(this);
            W          = reshape(W,[],nex);
            dth6 = []; dth7=[];
            dth5 = vec(sum(this.Bout'*W,2));
            [th1, th2,th3,th4,~,th6,th7] = this.split(theta);
            
            % load temps and recompute activations
            K1Y = dA{1};
            if not(isempty(this.nLayer1))
                [K1Y,~,tmpNL1] = apply(this.nLayer1,th6,K1Y);
            end
            [A1,dA1] = this.activation1(K1Y + this.Bin1*th3);
            K2Z = dA{2};
            if not(isempty(this.nLayer2))
                [K2Z,~,tmpNL2] = apply(this.nLayer2,th7,K2Z);
            end
            [~,dA2] = this.activation2(K2Z + this.Bin2*th4);
            
            % the actual sensitivity computation
            dAZ2 = dA2.*W;
            dth4 = vec(sum(this.Bin2'*reshape(dAZ2,[],nex),2));
            if not(isempty(this.nLayer2))
                [dth7,dAZ2] = JTmv(this.nLayer2,dAZ2,[],th7,dA{2},tmpNL2);
            end
            dth2 = JthetaTmv(this.K2,dAZ2,th2,A1);
            
            
            dAZ1 = dA1.*(getOp(this.K2,th2)'*dAZ2);
            dth3      = vec(sum(this.Bin1'*reshape(dAZ1,[],nex),2));
            if not(isempty(this.nLayer1))
                [dth6,dAZ1] = JTmv(this.nLayer1,dAZ1,[],th6,dA{1},tmpNL1);
            end
            dth1 = JthetaTmv(this.K1,dAZ1,th1,Y);
            
            dth = [dth1(:); dth2(:); dth3(:); dth4(:); dth5(:);dth6(:);dth7(:)];
        end
        
        function dY = JYTmv(this,Z,~,theta,Y,dA)
            [th1, th2,th3,th4,~,th6,th7] = this.split(theta);
            nex = numel(Z)/nFeatOut(this);
            Z   = reshape(Z,[],nex);
            
            % load temps and recompute activations
            K1Y = dA{1};
            if not(isempty(this.nLayer1))
                [K1Y,~,tmpNL1] = apply(this.nLayer1,th6,K1Y);
            end
            [~,dA1] = this.activation1(K1Y + this.Bin1*th3);
            K2Z = dA{2};
            if not(isempty(this.nLayer2))
                [K2Z,~,tmpNL2] = apply(this.nLayer2,th7,K2Z);
            end
            [~,dA2] = this.activation2(K2Z + this.Bin2*th4);
            
            K1Op = getOp(this.K1,th1);
            K2Op = getOp(this.K2,th2);
            
            dA2Z = dA2.*Z;
            if not(isempty(this.nLayer2))
               dA2Z = JYTmv(this.nLayer2,dA2Z,[],th7,dA{2},tmpNL2);
            end
            dA1Z = (dA1.*(K2Op'*dA2Z));
            if not(isempty(this.nLayer1))
                dA1Z = JYTmv(this.nLayer1,dA1Z,[],th6,dA{1},tmpNL1);
            end
            dY  = K1Op'*dA1Z;
        end
        
        function [dth,dY] = JTmv(this,Z,~,theta,Y,tmp,doDerivative)
            if not(exist('doDerivative','var')) || isempty(doDerivative)
               doDerivative =[1;0]; 
            end
            nex       = numel(Y)/nFeatIn(this);
            Z         = reshape(Z,[],nex);
            
            dY = [];
            
            [th1, ~] = this.split(theta);
            [dth,dA1Z]  = JthetaTmv(this,Z,[],theta,Y,tmp);
            if nargout==2 || doDerivative(2)==1
                dY  = getOp(this.K1,th1)'*dA1Z;
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


