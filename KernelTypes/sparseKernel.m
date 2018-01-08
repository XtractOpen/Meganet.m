classdef sparseKernel
    % classdef sparseKernel < handle
    % 
    % linear transformation given by sparse matrix
    %
    %   Y(theta,Y0)  = sparse(theta(:,1),theta(:,2),theta(:,3)) * Y0 
    %
    
    properties
        nK
        ival
        jval
        Qs
        useGPU
        precision
    end
    
    methods
        function this = sparseKernel(nK,varargin)
           this.nK = nK;
           this.ival = [];
           this.jval = [];
           this.Qs   = []; 
            useGPU = 0;
            precision = 'double';
            for k=1:2:length(varargin)     % overwrites default parameter
                eval([ varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            this.useGPU = useGPU;
            this.precision = precision;
            this.ival = ival;
            this.jval = jval;
            this.Qs   = Qs;
            
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
            n = size(this.Qs,2);
        end
        
        function n = nFeatIn(this)
            n = this.nK(2);
        end
        
        function n = nFeatOut(this)
            n = this.nK(1);
        end
        
        function theta = initTheta(this)
            theta = rand(nTheta(this),1);
        end
            
        function A = getOp(this,theta)
            A = sparse(this.ival,this.jval,this.Qs*theta,this.nK(1),this.nK(2));
        end
        
       
       function dY = Jthetamv(this,dtheta,~,Y,~)
           dY = getOp(this,dtheta)*Y;
       end
       
       function dtheta = JthetaTmv(this,Z,~,Y,~)
            % Jacobian T matvec.
            
            %n = nTheta(this);
            %As = getOp(this,ones(n,1));
            %As = As~=0;
            %dtheta1 = this.Qs'*nonzeros(((As').*(Y*Z'))');            
            %t = 0;
            %for i=1:size(Y,2)
            t = sum(Z(this.ival,:) .* Y(this.jval,:),2);
            %end
            dtheta = this.Qs'*t;
            
       end
        
       function Z = implicitTimeStep(this,theta,Y,h)
           
           Kop = getOp(this,theta);
           Z   = (h*(Kop'*Kop) + speye(size(Kop,2)))\Y; 
       end


    end
end

