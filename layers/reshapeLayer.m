classdef reshapeLayer < abstractMeganetElement
    % classdef reshapeLayer < abstractMeganetElement
    %
    % reshapes the feature matrix (used in semantic segmentation)
    %
    % this layer has no trainable weights
    %
    properties
        nY          % dimension of input data
        perm        % permutation applied to input data
        nf          % 2D array, number of input features and number of output
        useGPU      % flag for GPU computing 
        precision   % flag for precision 
    end
    methods
        function this = reshapeLayer(nY,nf,varargin)
            if nargin==0
                help(mfilename)
                return;
            end
            perm  = [1 2 3];
            useGPU     = 0;
            precision  = 'double';
            for k=1:2:length(varargin)     % overwrites default parameter
                    eval([varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            
            this.useGPU = useGPU;
            this.precision = precision;
            this.perm = perm;
            this.nY = nY;
            this.nf = nf;
            
        end
        function [Y,dA] = apply(this,theta,Y,varargin)
            Y   = reshape(Y,this.nY(1),this.nY(2),[]);
            Y   = permute(Y,this.perm);
            dA  = [];
            Y   = reshape(Y,this.nf(2),[]);
        end
        
        
        function n = nTheta(this)
            n = 0;
            
        end
        
        function n = nFeatIn(this)
            n = this.nf(1);
        end
        
        function n = nFeatOut(this)
            n = this.nf(2);
        end
       
        
        function theta = initTheta(this)
            theta = [];
        end
        
        
        function [dY] = Jthetamv(this,dtheta,theta,Y,~)
           dY = reshape(0*Y,this.nf(2),[]);
        end
        
        function dtheta = JthetaTmv(this,Z,~,theta,Y,~)
            dtheta = [];
        end
       
        
        function [dY] = JYmv(this,dY,theta,~,~)
           [dY] = apply(this,theta,dY);
        end
        
        function Z = JYTmv(this,Z,~,theta,~,~)
           Z = reshape(Z,this.nf(this.perm(1)),this.nY(this.perm(2)),[]);
           Z = ipermute(Z,this.perm);
           Z = reshape(Z,this.nf(1),[]);            
        end
        
        
        % ------- functions for handling GPU computing and precision ---- 
        function this = set.useGPU(this,value)
            if (value~=0) && (value~=1)
                error('useGPU must be 0 or 1.')
            else
                this.useGPU  = value;
            end
        end
        function this = set.precision(this,value)
            if not(strcmp(value,'single') || strcmp(value,'double'))
                error('precision must be single or double.')
            else
                this.precision = value;
            end
        end
        function useGPU = get.useGPU(this)
            useGPU = this.useGPU;
        end
        function precision = get.precision(this)
            precision = this.precision;
        end
    end
end


