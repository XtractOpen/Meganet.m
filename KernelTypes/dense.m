classdef dense
    % classdef dense < handle
    % 
    % linear transformation given by dense matrix
    %
    %   Y(theta,Y0)  = reshape(theta,nK) * Y0 
    %
    %   nK is (num output features)-by-(num input features)
    
    properties
        nK
        Q
        q 
        useGPU
        precision
    end
    
    methods
        function this = dense(nK,varargin)
            this.nK = nK;
            useGPU  = 0;
            Q = [];
            q = [];
            precision = 'double';
            for k=1:2:length(varargin)     % overwrites default parameter
                eval([ varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            this.Q = Q;
            this.q = q;
            this.useGPU = useGPU;
            this.precision = precision;
            
        end
        function this = gpuVar(this,useGPU,precision)
            [this.Q,this.q] = gpuVar(useGPU,precision,this.Q,this.q);
        end
        function this = set.useGPU(this,value)
            if (value~=0) && (value~=1)
                error('useGPU must be 0 or 1.')
            else
                this.useGPU = value;
            end
        end
        function this = set.precision(this,value)
            if not(strcmp(value,'single') || strcmp(value,'double'))
                error('precision must be single or double.')
            else
                this.precision = value;
            end
        end
        
        function n = nTheta(this)
            if isempty(this.Q)
                n = prod(this.nK);
            else
                n = size(this.Q,2);
            end
        end
        
        function n = sizeFeatIn(this)
            n = this.nK(2);
        end

        function n = sizeFeatOut(this)
            n = this.nK(1);
        end
        
        function n = numelFeatIn(this) % same as sizeFeat bc nK is 2-D
            n = this.nK(2);
        end
        
        function n = numelFeatOut(this) % same as sizeFeat bc nK is 2-D
            n = this.nK(1);
        end
        
        function theta = initTheta(this)
            sd = sqrt(2/this.nK(2));
            theta = sd*randn(nTheta(this),1);
        end
            
        function A = getOp(this,theta)
            if not(isempty(this.Q))
                theta = this.Q*theta + this.q;
            end
            A = reshape(vec(theta),this.nK);
        end
        
        function dY = Jthetamv(this,dtheta,~,Y,~)
            if not(isempty(this.Q))
                dtheta = this.Q*dtheta;
            end
            A = reshape(vec(dtheta),this.nK);
            dY = A*Y;
        end
        
        function J = getJthetamat(this,~,Y,~)
            J      = kron(Y',speye(this.nK(1)));
        end
        
       function dtheta = JthetaTmv(this,Z,~,Y,~)
            % Jacobian transpose matvec.
            dtheta   = vec(Z*Y');
            if not(isempty(this.Q))
                dtheta = this.Q'*dtheta;
            end
            
       end
        
       function Z = implicitTimeStep(this,theta,Y,h)
           Kop = getOp(this,theta);
           Z   = (h*(Kop'*Kop) + eye(size(Kop,2)))\Y; 
       end
           

    end
end

