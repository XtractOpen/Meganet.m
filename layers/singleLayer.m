classdef singleLayer < abstractMeganetElement
    % classdef singleLayer < abstractMeganetElement
    %
    % Implementation of single layer model
    %
    % Y(th,Y0) = activation( K(th_1)*Y0)+ Bin*th_2)+Bout*th_3
    %
    properties
        activation  % activation function
        K      % transformation type
        nLayer  % normalization layer
        Bin         % bias inside nonlinearity
        Bout        % bias outside nonlinearity
        useGPU      % flag for GPU computing (derived from trafo)
        precision   % flag for precision (derived from trafo)
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
            nLayer     = [];
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
                Bin = zeros(nFeatOut(K),0);
            end
            this.Bin = gpuVar(K.useGPU, K.precision, Bin);
                
            if isempty(Bout)
                Bout = zeros(nFeatOut(K),0);
            end
            this.Bout = gpuVar(K.useGPU, K.precision, Bout);
            this.nLayer = nLayer;
            this.K      = K;
            this.activation = activation;
            
        end
        function [th1,th2,th3,th4] = split(this,theta)
            th1 = theta(1:nTheta(this.K));
            cnt = numel(th1);
            th2 = theta(cnt+(1:size(this.Bin,2)));
            cnt = cnt + numel(th2);
            th3 = theta(cnt+(1:size(this.Bout,2)));
            cnt = cnt + numel(th3);
            th4 = theta(cnt+1:end);
        end
        
        function [Ydata,Y,tmp] = apply(this,theta,Y,varargin)
            doDerivative  = (nargout>1);
            for k=1:2:length(varargin)     % overwrites default parameter
                eval([varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            
            nex = numel(Y)/nFeatIn(this);
            Y   = reshape(Y,[],nex);
            [th1,th2,th3,th4] = split(this,theta);
            tmp = cell(1,2);
            
            Y      =  getOp(this.K,th1)*Y + this.Bin * th2;
            if not(isempty(this.nLayer))
                [Y,~,tmp{1}] = apply(this.nLayer,th4,Y);
            end
            [Y,tmp{2}] = this.activation(Y,'doDerivative',doDerivative);
            Y = Y +this.Bout*th3;
            Ydata = Y;
        end
        
        function n = nTheta(this)
            n = nTheta(this.K)+size(this.Bin,2) + size(this.Bout,2);
            if not(isempty(this.nLayer))
                n = n + nTheta(this.nLayer);
            end
        end
        
        function n = nFeatIn(this)
            n = nFeatIn(this.K);
        end
        
        function n = nFeatOut(this)
            n = nFeatOut(this.K);
        end
        
        function n = nDataOut(this)
            n = nFeatOut(this.K);
        end
        
        function theta = initTheta(this)
           theta = [vec(initTheta(this.K)); 0.1*ones(size(this.Bin,2),1) ; 0.1*ones(size(this.Bout,2),1) ];
           if not(isempty(this.nLayer))
               theta = [theta; initTheta(this.nLayer)];
           end
        end
        
        
        function [dZ] = Jthetamv(this,dtheta,theta,Y,tmp)
            dA  = tmp{2};
            nex = numel(Y)/nFeatIn(this);
            Y   = reshape(Y,[],nex);
            [th1,th2,~,th4]= split(this,theta);
            [dth1,dth2,dth3,dth4]= split(this,dtheta);
            
            dZ = Jthetamv(this.K,dth1,th1,Y) + this.Bin*dth2;
            if not(isempty(this.nLayer))
                Kop = getOp(this.K,th1);
                dZ  = Jmv(this.nLayer,dth4,dZ,th4,Kop*Y+this.Bin*th2,tmp{1});
            end
            dZ = dA.*dZ+this.Bout*dth3;
        end
        
        function [dZ] = JYmv(this,dY,theta,Y,tmp)
            dA  = tmp{2};
            nex = numel(dY)/nFeatIn(this);
            Y  = reshape(Y,[],nex);
            
            [th1,th2,~,th4] = split(this,theta);
            Kop = getOp(this.K,th1);
            dY   = reshape(dY,[],nex);
            dZ = Kop*dY;
            if not(isempty(this.nLayer))
                dZ = JYmv(this.nLayer,dZ,th4,Kop*Y+this.Bin*th2,tmp{1});
            end
            dZ = dA.*dZ;
        end
        
        function [dZ] = Jmv(this,dtheta,dY,theta,Y,tmp)
            dA = tmp{2};
            nex = numel(Y)/nFeatIn(this);
            Y   = reshape(Y,[],nex);
            [th1,th2,~,th4]= split(this,theta);
            [dth1,dth2,dth3,dth4]= split(this,dtheta);
            Kop = getOp(this.K,th1);
            if isempty(dY) || (numel(dY)==1 && abs(dY)==0)
                dZ = 0;
            else
                dY = reshape(dY,[],nex);
                dZ = Kop*dY;
            end
            dZ = dZ + Jthetamv(this.K,dth1,th1,Y) + this.Bin*dth2;
            if not(isempty(this.nLayer))
                dZ = Jmv(this.nLayer,dth4,dZ,th4,Kop*Y+this.Bin*th2,tmp{1});
            end
            
            dZ = dA.*dZ + this.Bout*dth3;
        end
        
        function [dtheta,dY] = JTmv(this,Z,~,theta,Y,tmp,doDerivative)
            if not(exist('doDerivative','var')) || isempty(doDerivative)
               doDerivative =[1;0]; 
            end
            dA = tmp{2};
            nex       = numel(Y)/nFeatIn(this);
            dY = [];
            if isscalar(Z) && Z==0
                dtheta = 0*theta; 
                dY     = 0*Y;
                return
            end
            Z         = reshape(Z,[],nex);
            [th1,th2,~,th4] = split(this,theta);
            Kop = getOp(this.K,th1);
            
            dth3      = vec(sum(this.Bout'*Z,2));
            dAZ       = dA.*Z;
            
            if not(isempty(this.nLayer))
               [dth4,dAZ] = JTmv(this.nLayer,dAZ,[],th4,Kop*Y+this.Bin*th2,tmp{1}); 
            else
               dth4 = [];
            end
            dth2   = vec(sum(this.Bin'*reshape(dAZ,[],nex),2));
            dth1   = JthetaTmv(this.K, dAZ,theta,Y);
            
            if nargout==2 || doDerivative(2)==1
                dY   = Kop'*dAZ;
            end
            dtheta = [dth1(:); dth2(:); dth3(:); dth4(:)];
            
            if nargout==1 && all(doDerivative==1)
                dtheta = [dtheta(:);dY(:)];
            end

        end
        
        function dtheta = JthetaTmv(this,Z,~,theta,Y,tmp)
            dA        = tmp{2};
            nex       = numel(Y)/nFeatIn(this);
            Z         = reshape(Z,[],nex);
            dth3      = vec(sum(this.Bout'*Z,2));
            dAZ       = dA.*Z;
            if not(isempty(this.nLayer))
               [th1,th2,~,th4]  = split(this,theta);
               Kop       = getOp(this.K,th1);
               [dth4,dAZ] = JTmv(this.nLayer,dAZ,[],th4,Kop*Y+this.Bin*th2,tmp{1}); 
            else
               dth4 = [];
            end
            
            dth1      = JthetaTmv(this.K,dAZ,theta,Y);
            dth2      = vec(sum(this.Bin'*reshape(dAZ,[],nex),2));
            dtheta = [dth1(:); dth2(:); dth3(:);dth4(:)];
        end
        
        function dY = JYTmv(this,Z,~,theta,Y,tmp)
            dA    = tmp{2};
            nex   = numel(Y)/nFeatIn(this);
            [th1,th2,~,th4] = this.split(theta);
            if all(Z(:)==0)
                dY = 0*Y;
                return
            end
            Kop = getOp(this.K,th1);
            
            Z     = reshape(Z,[],nex);
            dAZ   = dA.*Z;
            if not(isempty(this.nLayer))
                dAZ = JYTmv(this.nLayer,dAZ,[],th4,Kop*Y+this.Bin*th2,tmp{1});
            end
            dY    = Kop'*dAZ;
        end
        
        
        % ------- functions for handling GPU computing and precision ---- 
        function this = set.useGPU(this,value)
            if (value~=0) && (value~=1)
                error('useGPU must be 0 or 1.')
            else
                this.K.useGPU  = value;
                this.Bin  = gpuVar(this.useGPU, this.precision, this.Bin);
                this.Bout  = gpuVar(this.useGPU, this.precision, this.Bout);
            end
        end
        function this = set.precision(this,value)
            if not(strcmp(value,'single') || strcmp(value,'double'))
                error('precision must be single or double.')
            else
                this.K.precision = value;
                this.Bin  = gpuVar(this.useGPU, this.precision, this.Bin);
                this.Bout = gpuVar(this.useGPU, this.precision, this.Bout);
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


