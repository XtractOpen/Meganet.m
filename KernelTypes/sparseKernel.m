classdef sparseKernel
    % classdef sparseKernel < handle
    %
    % linear transformation given by sparse matrix
    %
    %   Y(theta,Y0)  = sparse(ival,jval,Qs*theta) * Y0
    %
    % The kernel is described by providing the row and column indices of
    % the non-zero elements in the sparse kernel and the size of the
    % resulting matrix.
    
    properties
        nK           % size of matrix
        ival         % row-indices of non-zero elements in sparse matrix
        jval         % column-indices of non-zero elements in sparse matrix
        Qs           % basis for non-zero elements (default: speye)
        useGPU
        precision
    end
    
    methods
        function this = sparseKernel(ival,jval,nK,varargin)
            % constructor, required arguments are ival, jval, nK
            Qs   = [];
            useGPU = 0;
            precision = 'double';
            for k=1:2:length(varargin)     % overwrites default parameter
                eval([ varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            this.useGPU = useGPU;
            this.precision = precision;
            
            this.nK = nK;
            if not(numel(jval)==numel(ival))
                error('number of column indices must equal number of row indices');
            end
            this.ival = ival;
            this.jval = jval;
            
            if (isempty(Qs))
                Qs = speye(numel(jval));
            end
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
            n = size(this.Qs,2); %%%% TODO
        end
        
        function n = sizeFeatIn(this)
            n = this.nK(2);
        end
        
        function n = sizeFeatOut(this)
            n = this.nK(1);
        end
        
        function n = numelFeatIn(this) % nK is 2-D so same as sizeFeatIn
            n = this.nK(2); 
        end
        
        function n = numelFeatOut(this) % nK is 2-D so same as sizeFeatOut
            n = this.nK(1);
        end
        
        function theta = initTheta(this)
            theta = rand(nTheta(this),1);
        end
        
        function A = getOp(this,theta)
            A = sparse(this.ival,this.jval,theta,this.nK(1),this.nK(2));
        end
        
        function dY = Jthetamv(this,dtheta,~,Y,~)
            dY = getOp(this,dtheta)*Y;
        end
        
        function dtheta = JthetaTmv(this,Z,~,Y,~)
            t = sum(Z(this.ival,:) .* Y(this.jval,:),2);
            dtheta = this.Qs'*t;
        end
        
        function Z = implicitTimeStep(this,theta,Y,h)
            Kop = getOp(this,theta);
            Z   = (h*(Kop'*Kop) + speye(size(Kop,2)))\Y;
        end
        
        
    end
end

