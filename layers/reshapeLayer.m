classdef reshapeLayer < abstractMeganetElement
    % classdef reshapeLayer < abstractMeganetElement
    %
    % reshapes the feature matrix (used in semantic segmentation)
    %
    % this layer has no trainable weights
    %
    properties
        nIn         % shape of input tensor
        nOut        % shape of output tensor
        perm        % permutation applied to input data
        useGPU
        precision
    end
    methods
        function this = reshapeLayer(nIn,nOut,varargin)
            if nargin==0
                help(mfilename)
                return;
            end
            perm  = [1 2 3 4];
            for k=1:2:length(varargin)     % overwrites default parameter
                    eval([varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            
            this.perm = perm;
            this.nOut = nOut;
            this.nIn = nIn;
        end
        function [Y,dA] = forwardProp(this,~,Y,varargin)
            Y   = permute(Y,[this.perm 4]);
            dA  = [];
            Y   = reshape(Y,this.nOut(1),[]);
        end
        
        
        function n = nTheta(this)
            n = 0;
        end
        
        function n = sizeFeatIn(this)
                n = this.nIn;
        end
        function n = sizeFeatOut(this)
                n = this.nOut;
        end
        
        function theta = initTheta(this)
            theta = [];
        end
        
        
        function dY = Jthetamv(this,dtheta,theta,Y,~)
           dY = reshape(0*Y,this.nOut);
        end
        
        function dtheta = JthetaTmv(this,Z,theta,Y,~)
            dtheta = [];
        end
       
        
        function dY = JYmv(this,dY,theta,~,~)
           dY = forwardProp(this,theta,dY);
        end
        
        function Z = JYTmv(this,Z,theta,~,~)
           szT = this.nIn([this.perm]);
           Z = reshape(Z,szT(1),szT(2),szT(3),[]);
           Z = ipermute(Z,[this.perm 4]);
        end
        end
end


