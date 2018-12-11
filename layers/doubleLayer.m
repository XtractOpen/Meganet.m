classdef doubleLayer < abstractMeganetElement
    % classdef doubleLayer < abstractMeganetElement
    %
    % implementation of double layer
    %
    % Y(theta) = act2 (K2(th2)*act1(K1(th1)*Y0 + Bin1*th3) + Bin2*th4) + Bout*th5
    
    properties
        K1      % inner kernel
        K2      % outer kernel
        normLayer1 % inner normalization
        normLayer2 % outer normalization
        activation1  % inner activation
        activation2  % outer activation
        Bin1         % bias inside first activation
        Bin2         % bias inside second activation
        Bout         % outer bias
        useGPU       
        precision
        storeInterm   % flag for storing intermediates

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
            normLayer1 = [];
            normLayer2 = [];
            activation1 = [];
            activation2 = [];
            storeInterm = 0;
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
                Bin1 = zeros([sizeFeatOut(K1),0]);
            end
            this.Bin1 = gpuVar(K1.useGPU, K1.precision, Bin1);
                
            if isempty(Bin2)
                Bin2 = zeros([sizeFeatOut(K2),0]);
            end
            this.Bin2 = gpuVar(K2.useGPU, K2.precision, Bin2);
            
            if isempty(Bout)
                Bout = zeros([sizeFeatOut(K2),0]);
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
            if sizeFeatOut(K1) ~= sizeFeatIn(K2)
                error('%s - number of output features of first trafo (%d) must match input of second trafo (%d)',...
                    mfilename,sizeFeatOut(K1), sizeFeatIn(K2));
            end
            this.K1      = K1;
            this.K2      = K2;
            this.normLayer1 = normLayer1;
            this.normLayer2 = normLayer2;
            this.storeInterm=storeInterm;
            
            if not(isempty(normLayer1)) && any(sizeFeatIn(normLayer1)~=sizeFeatOut(this.K1))
                error('input dimension of first normalization layer does not match output dimension of K1')
            end
            if not(isempty(normLayer2)) && any(sizeFeatIn(normLayer2)~=sizeFeatOut(this.K2))
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
                
                nk3 = sizeLastDim(this.Bin1);
                th3 = x(cnt:cnt+nk3-1);
                cnt = cnt+nk3;
                
                nk4 = sizeLastDim(this.Bin2);
                th4 = x(cnt:cnt+nk4-1);
                cnt = cnt+nk4;
                
                nk5 = sizeLastDim(this.Bout);
                th5 = x(cnt:cnt+nk5-1);
                cnt = cnt+nk5;
                if not(isempty(this.normLayer1))
                    nk6 = nTheta(this.normLayer1);
                    th6 = x(cnt:cnt+nk6-1);
                    cnt = cnt + nk6;
                end
                
                if not(isempty(this.normLayer2))
                   nk7 = nTheta(this.normLayer2);
                   th7 = x(cnt:cnt+nk7-1);
                end
            end
        end
        function n = nTheta(this)
            n = nTheta(this.K1) + nTheta(this.K2)...
                 + sizeLastDim(this.Bin1) + sizeLastDim(this.Bin2) ...
                 + sizeLastDim(this.Bout);
            if not(isempty(this.normLayer1))
                n = n+nTheta(this.normLayer1);
            end
            if not(isempty(this.normLayer2))
                n = n + nTheta(this.normLayer2);
            end
        end
        function n = sizeFeatIn(this)
            n = sizeFeatIn(this.K1);
        end
        
        function n = sizeFeatOut(this)
            n = sizeFeatOut(this.K2);
        end
        
        function theta = initTheta(this)
            theta = [vec(initTheta(this.K1)); vec(initTheta(this.K2)); ...
                     0.1*ones(sizeLastDim(this.Bin1),1); ...
                     0.1*ones(sizeLastDim(this.Bin2),1); ...
                     0.1*ones(sizeLastDim(this.Bout),1)];
                 
            if not(isempty(this.normLayer1))
                theta = [theta; initTheta(this.normLayer1)];
            end
            if not(isempty(this.normLayer2))
                theta = [theta; initTheta(this.normLayer2)];
            end
        end
        
        % ------- apply forward model ----------
        function [Y,tmp] = forwardProp(this,theta,Y,varargin)
            doDerivative = (nargout>1);
            for k=1:2:length(varargin)     % overwrites default parameter
                eval([varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            
            tmp        = cell(1,2);
            [th1,th2,th3,th4,th5,th6,th7] = this.split(theta);
            K1Y         = getOp(this.K1,th1)* Y;
            if doDerivative, tmp{1} = K1Y; end
            
            if not(isempty(this.normLayer1))
                K1Y = forwardProp(this.normLayer1,th6,K1Y);
            end
            if not(isempty(th3))
                K1Y          =  K1Y + this.Bin1*th3;
            end
            Z1 = this.activation1(K1Y);
            K2Z        = getOp(this.K2, th2)* Z1;
            if doDerivative, tmp{2} = K2Z; end
            if not(isempty(this.normLayer2))
                K2Z = forwardProp(this.normLayer2,th7,K2Z);
            end
            if not(isempty(th4))
                K2Z = K2Z + this.Bin2*th4;
            end
            Y     = this.activation2(K2Z);
            if not(isempty(th5))
                Y     = Y+  this.Bout*th5;
            end
        end
        
        function [A1,dA1,A2,dA2,K1Y,K2Z,tmpNL1,tmpNL2] = getTempsForSens(this,theta,Y,tmp)
            % re-computes temp variables needed for sensitivity computations
            %
            % Input:
            %   theta - current weights
            %   Y     - input features
            %   KY    - either K*Y stored during forwardProp or empty
            %
            % Output:
            %   dA    - derivative of activation
            %   KY    - K(theta)*Y
            %   tmpNL - temp results of norm Layer
            
           
            tmpNL1 =[]; tmpNL2 =[];
            [th1,th2,th3,th4,~,th6,th7] = this.split(theta);
            
            if not(this.storeInterm) || isempty(tmp{1})
                K1Y         = getOp(this.K1,th1) * Y;
            else
                K1Y = tmp{1};
            end
            if not(isempty(this.normLayer1))
                [K1Yn,tmpNL1] = forwardProp(this.normLayer1,th6,K1Y);
            else
                K1Yn = K1Y;
            end
            if not(isempty(th3))
                K1Yn = K1Yn + this.Bin1*th3;
            end
            [A1,dA1] = this.activation1(K1Yn);
            if not(this.storeInterm) || isempty(tmp{2})
                K2Z        = getOp(this.K2,th2) * A1;
            else
                K2Z = tmp{2};
            end
            
            if not(isempty(this.normLayer2))
                [K2Zn,tmpNL2] = forwardProp(this.normLayer2,th7,K2Z);
            else
                K2Zn= K2Z;
            end
            if not(isempty(th4))
                K2Zn = K2Zn + this.Bin2*th4;
            end
            [A2,dA2] = this.activation2(K2Zn);
        end
        
        % ----------- Jacobian matvecs -----------
        function dZ = Jthetamv(this,dtheta,theta,Y,dA)
            dZ = Jmv(this,dtheta,[],theta,Y,dA);
        end
        
        function dZ = JYmv(this,dY,theta,Y,dA)
            
            if not(isempty(dY)) && (not(isscalar(dY) && dY==0))
                % load temps and recompute activations
                [th1, th2,~,~,~,th6,th7] = this.split(theta);
                [~,dA1,~,dA2,~,~,tmpNL1,tmpNL2] = getTempsForSens(this,theta,Y,dA);

                K1dY  = getOp(this.K1,th1)* dY;
                if not(isempty(this.normLayer1))
                    K1dY = JYmv(this.normLayer1,K1dY,th6,dA{1},tmpNL1);
                end
                K2dZ = getOp(this.K2,th2)* (dA1.*(K1dY));
                if not(isempty(this.normLayer2))
                    K2dZ = JYmv(this.normLayer2,K2dZ,th7,dA{2},tmpNL2);
                end
                dZ = dA2.*(K2dZ);
            end
        end
        
        function dZ = Jmv(this,dtheta,dY,theta,Y,dA)
            
            [dth1,dth2,dth3,dth4,dth5,dth6,dth7] = this.split(dtheta);
            [th1, th2,~,~,~,th6,th7] = this.split(theta);
            
            [A1,dA1,A2,dA2,K1Y,K2Z,tmpNL1,tmpNL2] = getTempsForSens(this,theta,Y,dA);

            dZ = 0.0;
            dK1Op = getOp(this.K1,dth1);
            dK2Op = getOp(this.K2,dth2);
            K2Op  = getOp(this.K2,th2);
            
            dK1dY = dK1Op*Y; 
            if not(isempty(dY)) && numel(dY)>1 && any(vec(dY)~=0)
                dK1dY = dK1dY + getOp(this.K1,th1)* dY;
            end
            if not(isempty(this.normLayer1))
                dK1dY = Jmv(this.normLayer1,dth6,dK1dY,th6,K1Y,tmpNL1);
            end
            if not(isempty(this.Bin1))
                d1 = dA1.*(dK1dY + this.Bin1*dth3);
            else
                d1 = dA1 .* dK1dY;
            end
            dK2Z = dK2Op*A1 + K2Op*d1;
            
            if not(isempty(this.normLayer2))
                dK2Z = Jmv(this.normLayer2,dth7,dK2Z,th7,K2Z,tmpNL2);
            end
            if not(isempty(this.Bin2))
                dZ = dZ + dA2 .* ( dK2Z + this.Bin2*dth4);
            else
                dZ = dZ + dA2 .* dK2Z;
            end
            if not(isempty(this.Bout))
                dZ = dZ + this.Bout*dth5;
            end
        end
        
        % ----------- Jacobian' matvecs ----------
        
        function [dth,dAZ1] = JthetaTmv(this,W,theta,Y,dA)
            nd = ndims(Y);
            dth3 = []; dth4=[]; dth5 = []; dth6 = []; dth7=[];
            if not(isempty(this.Bout))
                dth5 = vec(sum(this.Bout'*W,nd));
            end
            [th1,th2,~,~,~,th6,th7] = this.split(theta);
            
            [A1,dA1,A2,dA2,K1Y,K2Z,tmpNL1,tmpNL2] = getTempsForSens(this,theta,Y,dA);

            dAZ2 = dA2.*W;
            if not(isempty(this.Bin2))
                dth4 = vec(sum(this.Bin2'*dAZ2,nd));
            end
            if not(isempty(this.normLayer2))
                [dth7,dAZ2] = JTmv(this.normLayer2,dAZ2,th7,K2Z,tmpNL2);
            end
            dth2 = JthetaTmv(this.K2,dAZ2,th2,A1);
            
            
            dAZ1 = dA1.*(getOp(this.K2,th2)'*dAZ2);
            if not(isempty(this.Bin1))
                dth3      = vec(sum(this.Bin1'*dAZ1,nd));
            end
            if not(isempty(this.normLayer1))
                [dth6,dAZ1] = JTmv(this.normLayer1,dAZ1,th6,K1Y,tmpNL1);
            end
            dth1 = JthetaTmv(this.K1,dAZ1,th1,Y);
            
            dth = [dth1(:); dth2(:); dth3(:); dth4(:); dth5(:);dth6(:);dth7(:)];
        end
        
        function dY = JYTmv(this,Z,theta,Y,dA)
            [th1, th2,th3,th4,~,th6,th7] = this.split(theta);
            
            [A1,dA1,A2,dA2,K1Y,K2Z,tmpNL1,tmpNL2] = getTempsForSens(this,theta,Y,dA);

            K1Op = getOp(this.K1,th1);
            K2Op = getOp(this.K2,th2);
            
            dA2Z = dA2.*Z;
            if not(isempty(this.normLayer2))
               dA2Z = JYTmv(this.normLayer2,dA2Z,th7,K2Z,tmpNL2);
            end
            dA1Z = (dA1.*(K2Op'*dA2Z));
            if not(isempty(this.normLayer1))
                dA1Z = JYTmv(this.normLayer1,dA1Z,th6,K1Y,tmpNL1);
            end
            dY  = K1Op'*dA1Z;
        end
        
        function [dth,dY] = JTmv(this,Z,theta,Y,tmp,doDerivative)
            if not(exist('doDerivative','var')) || isempty(doDerivative)
               doDerivative =[1;0]; 
            end
            
            dY = [];
            
            [th1, ~] = this.split(theta);
            [dth,dA1Z]  = JthetaTmv(this,Z,theta,Y,tmp);
            if nargout==2 || doDerivative(2)==1
                dY  = getOp(this.K1,th1)'*dA1Z;
            end
            if nargout==1 && all(doDerivative==1)
                dth = [dth(:);dY(:)];
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
            
            [th1,th2,th3,th4,th5,th6,th7] = this.split(theta);
            th1 = prolongateConvStencils(this.K1,th1,getRP);
            th2 = prolongateConvStencils(this.K2,th2,getRP);
            thFine = [vec(th1); vec(th2); th3; th4; th5; th6; th7];
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
            
            [th1,th2,th3,th4,th5,th6,th7] = this.split(theta);
            th1 = restrictConvStencils(this.K1,th1,getRP);
            th2 = restrictConvStencils(this.K2,th2,getRP);
            thCoarse = [vec(th1); vec(th2); th3; th4; th5; th6; th7];
        end
        
        % ------- functions for handling GPU computing and precision ---- 
        function set.useGPU(this,value)
            if (value~=0) && (value~=1)
                error('useGPU must be 0 or 1.')
            else
                this.K1.useGPU  = value;
                this.K2.useGPU  = value;
                this.Bin1  = gpuVar(this.useGPU, this.precision, this.Bin1);
                this.Bin2  = gpuVar(this.useGPU, this.precision, this.Bin2);
                this.Bout  = gpuVar(this.useGPU, this.precision, this.Bout);
                if not(isempty(this.normLayer1))
                    this.normLayer1.useGPU = value;
                end
                if not(isempty(this.normLayer2))
                    this.normLayer2.useGPU = value;
                end
                
            end
        end
        function set.precision(this,value)
            if not(strcmp(value,'single') || strcmp(value,'double'))
                error('precision must be single or double.')
            else
                this.K1.precision = value;
                this.K2.precision = value;
                this.Bin1  = gpuVar(this.useGPU, value, this.Bin1);
                this.Bin2  = gpuVar(this.useGPU, value, this.Bin2);
                this.Bout  = gpuVar(this.useGPU, value, this.Bout);
                if not(isempty(this.normLayer1))
                    this.normLayer1.precision = value;
                end
                if not(isempty(this.normLayer2))
                    this.normLayer2.precision = value;
                end

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


