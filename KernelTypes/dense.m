classdef dense
    % classdef dense < handle
    % 
    % linear transformation given by dense matrix
    %
    %   Y(theta,Y0)  = reshape(theta,nK) * Y0 
    %
    
    properties
        nK
        useGPU
        precision
    end
    
    methods
        function this = dense(nK,varargin)
           this.nK = nK;
           useGPU = 0;
            precision = 'double';
            for k=1:2:length(varargin)     % overwrites default parameter
                eval([ varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            this.useGPU = useGPU;
            this.precision = precision;
            
        end
        function this = gpuVar(this,useGPU,precision)
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
            n = prod(this.nK);
        end
        
        function n = nFeatIn(this)
            n = this.nK(2);
        end
        
        function n = nFeatOut(this)
            n = this.nK(1);
        end
        
        function theta = initTheta(this)
            theta = rand(this.nK);
            if this.nK(1)==this.nK(2)
                theta = theta - theta';
            end
        end
            
        function A = getOp(this,theta)
            A = reshape(theta,this.nK);
        end
        
        function dY = Jthetamv(this,dtheta,~,Y,~)
            nex    =  numel(Y)/nFeatIn(this);
            Y      = reshape(Y,[],nex);
            dY = getOp(this,dtheta)*Y;
        end
        
        function J = getJthetamat(this,~,Y,~)
            nex    =  numel(Y)/nFeatIn(this);
            Y      = reshape(Y,[],nex);
            J      = kron(Y',speye(this.nK(1)));
        end
        
       function dtheta = JthetaTmv(this,Z,~,Y,~)
            % Jacobian transpose matvec.
            nex    =  numel(Y)/nFeatIn(this);
            Y      = reshape(Y,[],nex);
            Z      = reshape(Z,[],nex);
            dtheta   = (Y*Z')';
       end
        
       function Z = implicitTimeStep(this,theta,Y,h)
           
           Kop = getOp(this,theta);
           Z   = (h*(Kop'*Kop) + eye(size(Kop,2)))\Y; 
       end
           

    end
end

