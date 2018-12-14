classdef singleLayer < abstractMeganetElement
    % classdef singleLayer < abstractMeganetElement
    %
    % Implementation of single layer model
    %
    % Y(th,Y0) = activation( K(th_1)*Y0 + Bin*th_2)+Bout*th_3
    %
    properties
        activation  % activation function
        K           % transformation type
        normLayer      % normalization layer
        Bin         % bias inside nonlinearity
        Bout        % bias outside nonlinearity
        useGPU      % flag for GPU computing (derived from trafo)
        precision   % flag for precision (derived from trafo)
        storeInterm % flag for storing intermediate K*Y
    end
    methods
        function this = singleLayer(K,varargin)
            if nargin==0
                help(mfilename)
                return;
            end
            activation = @tanhActivation;
            useGPU     = [];
            precision  = [];
            Bin        = [];
            Bout       = [];
            normLayer     = [];
            storeInterm =0;
            for k=1:2:length(varargin)     % overwrites default parameter
                    eval([varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            if not(isempty(useGPU))
                K.useGPU = useGPU;
            end
            if not(isempty(precision))
                K.precision=precision;
            end
            if isempty(Bin)
                Bin = zeros( [sizeFeatOut(K),0] );
            end
            this.Bin = gpuVar(K.useGPU, K.precision, Bin);
                
            if isempty(Bout)
                Bout = zeros([sizeFeatOut(K),0]);
            end
            this.Bout = gpuVar(K.useGPU, K.precision, Bout);
            this.normLayer = normLayer;
            this.K      = K;
            this.activation = activation;
            this.storeInterm=storeInterm;
            
        end
        function [th1,th2,th3,th4] = split(this,theta)
            th1 = theta(1:nTheta(this.K));
            cnt = numel(th1);
            th2 = theta(cnt+(1:sizeLastDim(this.Bin)));
            cnt = cnt + numel(th2);
            th3 = theta(cnt+(1:sizeLastDim(this.Bout)));
            cnt = cnt + numel(th3);
            th4 = theta(cnt+1:end);
        end
        
        function [Y,KY] = forwardProp(this,theta,Y,varargin)
            doDerivative  = (nargout>1); KY = [];
            for k=1:2:length(varargin)     % overwrites default parameter
                eval([varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            [th1,th2,th3,th4] = split(this,theta);
            
            Y      =  getOp(this.K,th1)*Y;
            if this.storeInterm
                KY    = Y;
            end
            if not(isempty(this.normLayer))
                Y = forwardProp(this.normLayer,th4,Y,'doDerivative',doDerivative);
            end
            
            if not(isempty(th2))
                Y = Y + this.Bin * th2;
            end
            Y = this.activation(Y);
            if not(isempty(th3))
                Y = Y +this.Bout*th3;
            end
        end
        
        function n = nTheta(this)
            n = nTheta(this.K) + sizeLastDim(this.Bin) + sizeLastDim(this.Bout) ;
            if not(isempty(this.normLayer))
                n = n + nTheta(this.normLayer);
            end
        end
        
        function n = sizeFeatIn(this)
            n = sizeFeatIn(this.K);
        end
        
        function n = sizeFeatOut(this)
            n = sizeFeatOut(this.K);
        end
        
        function theta = initTheta(this)
           theta = [vec(initTheta(this.K)); 0.0*ones(sizeLastDim(this.Bin),1) ;...
                    0.0*ones(sizeLastDim(this.Bout),1) ];
           if not(isempty(this.normLayer))
               theta = [theta; initTheta(this.normLayer)];
           end
        end
        
        function [dA,KY] = getTempsForSens(this,theta,Y,KY)
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
            
            [th1, th2,~,th4]  = split(this,theta);
            
            if not(this.storeInterm)
                KY = getOp(this.K,th1)*Y;
            end
            if not(isempty(this.normLayer))
                KYn = forwardProp(this.normLayer,th4,KY,'doDerivative',1);
            else
                KYn = KY;
            end
            if not(isempty(th2))
                KYn = KYn + this.Bin*th2;
            end
            [~,dA] = this.activation( KYn);
        end
        
        function dZ = Jthetamv(this,dtheta,theta,Y,KY)
            [th1, ~,~,th4]  = split(this,theta);
            
            [dA,KY] = getTempsForSens(this,theta,Y,KY);
            [dth1,dth2,dth3,dth4] = split(this,dtheta);
            
            dZ = Jthetamv(this.K,dth1,th1,Y);
            if not(isempty(this.normLayer))
                dZ  = Jmv(this.normLayer,dth4,dZ,th4,KY);
            end
            
            if not(isempty(this.Bin))
                dZ = dZ +  this.Bin*dth2;
            end
            
            dZ = dA.*dZ;
            if not(isempty(this.Bout))
                dZ = dZ +this.Bout*dth3;
            end
        end
        
        function dZ = JYmv(this,dY,theta,Y,KY)
            [th1,~,~,th4]  = split(this,theta);
            [dA,KY] = getTempsForSens(this,theta,Y,KY);

            Kop = getOp(this.K,th1);
            dZ = Kop*dY;
            if not(isempty(this.normLayer))
                dZ = JYmv(this.normLayer,dZ,th4,KY);
            end
            dZ = dA.*dZ;
        end
        
        function dZ = Jmv(this,dtheta,dY,theta,Y,KY)
            [th1, ~,~,th4]  = split(this,theta);
                        
            [dA,KY] = getTempsForSens(this,theta,Y,KY);

            
            [dth1,dth2,dth3,dth4]= split(this,dtheta);
            Kop = getOp(this.K,th1);
            if isempty(dY) || (numel(dY)==1 && abs(dY)==0)
                dZ = 0;
            else
                dZ = Kop*dY;
            end
            dZ = dZ + Jthetamv(this.K,dth1,th1,Y);
            if not(isempty(this.normLayer))
                dZ = Jmv(this.normLayer,dth4,dZ,th4,KY);
            end
            if not(isempty(this.Bin))
                dZ = dZ +  this.Bin*dth2;
            end
            dZ = dA.*dZ;
            if not(isempty(this.Bout))
                dZ = dZ + this.Bout*dth3;
            end
        end
        
        function [dtheta,dY] = JTmv(this,Z,theta,Y,KY,doDerivative)
            if not(exist('doDerivative','var')) || isempty(doDerivative)
               doDerivative =[1;0]; 
            end
            [th1, ~,~,th4]  = split(this,theta);
            [dA,KY] = getTempsForSens(this,theta,Y,KY);
            nd = ndims(Y);
            
            dY = [];
            if isscalar(Z) && Z==0
                dtheta = 0*theta; 
                dY     = 0*Y;
                return
            end
            Kop = getOp(this.K,th1);
            
            
            if not(isempty(this.Bout))
                dth3 = vec(sum( this.Bout'*Z ,nd));
            else 
                dth3 = [];
            end
                
            dAZ  = dA.*Z;
            
            if not(isempty(this.Bin))
                dth2   = vec(sum(this.Bin'*dAZ,nd));
            else
                dth2 = [];
            end
            
            if not(isempty(this.normLayer))
               [dth4,dAZ] = JTmv(this.normLayer,dAZ,th4,KY); 
            else
               dth4 = [];
            end
            dth1   = JthetaTmv(this.K, dAZ,theta,Y);
            
            if nargout==2 || doDerivative(2)==1
                dY   = Kop'*dAZ;
            end
            dtheta = [dth1(:); dth2(:); dth3(:); dth4(:)];
            
            if nargout==1 && all(doDerivative==1)
                dtheta = [dtheta(:);dY(:)];
            end

        end
        
        function dtheta = JthetaTmv(this,Z,theta,Y,KY)
            [~, ~,~,th4]  = split(this,theta);
            [dA,KY] = getTempsForSens(this,theta,Y,KY);
            nd = ndims(Y);
            
            if not(isempty(this.Bout))
                dth3      = vec(sum(this.Bout'*Z,nd));
            else
                dth3 = [];
            end
            
            dAZ       = dA.*Z;
            
            if not(isempty(this.Bin))
                dth2      = vec(sum(this.Bin'*dAZ,nd));
            else
                dth2 = [];
            end
            
            if not(isempty(this.normLayer))
               [dth4,dAZ] = JTmv(this.normLayer,dAZ,th4,KY); 
            else
               dth4 = [];
            end
            
            dth1      = JthetaTmv(this.K,dAZ,theta,Y);
            dtheta = [dth1(:); dth2(:); dth3(:);dth4(:)];
        end
        
        function dY = JYTmv(this,Z,theta,Y,KY)
            [th1, ~,~,th4]  = split(this,theta);
            [dA,KY] = getTempsForSens(this,theta,Y,KY);

            if all(Z(:)==0)
                dY = 0*Y;
                return
            end
            Kop = getOp(this.K,th1);
            
            dAZ   = dA.*Z;
            if not(isempty(this.normLayer))
                dAZ = JYTmv(this.normLayer,dAZ,th4,KY);
            end
            dY    = Kop'*dAZ;
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
            
            if not(exist('getRP','var'))
                getRP = [];
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
                this.Bin  = gpuVar(value, this.precision, this.Bin);
                this.Bout  = gpuVar(value, this.precision, this.Bout);
                if not(isempty(this.normLayer))
                    this.normLayer.useGPU = value;
                end
            end
        end
        function this = set.precision(this,value)
            if not(strcmp(value,'single') || strcmp(value,'double'))
                error('precision must be single or double.')
            else
                this.K.precision = value;
                this.Bin  = gpuVar(this.useGPU, value, this.Bin);
                this.Bout = gpuVar(this.useGPU, value, this.Bout);
                if not(isempty(this.normLayer))
                    this.normLayer.precision = value;
                end
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


