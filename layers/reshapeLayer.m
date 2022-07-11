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
            if nargout==0 && nargin==0
                this.runMinimalExample;
                return;
            end
            perm  = [1 2 3 4];
            for k=1:2:length(varargin)     % overwrites default parameter
                    eval([varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            
            this.perm = perm(1:numel(nIn));
            this.nOut = nOut;
            this.nIn = nIn;
        end
        function [Y,dA] = forwardProp(this,~,Y,varargin)
            nex = sizeLastDim(Y);
            Y   = permute(Y,[this.perm numel(this.perm)+1]);
            dA  = [];
            Y   = reshape(Y,[ this.nOut nex]);
        end
        
        
        function runMinimalExample(~)
            nc = 4;
            nx = 16;
            ny = 28;
            nex = 12;
            Y = randn(nc,nx,ny,nex);
            rshp = reshapeLayer([nc,nx,ny], nc*nx*ny);
            rshpInv = reshapeLayer(nc*nx*ny ,[nc,nx,ny] );
            Ym = forwardProp(rshp,[],Y);
            Yt = forwardProp(rshpInv,[],Ym);
            size(Y)
            size(Ym)
            size(Yt)
            norm(Y(:)-Yt(:),'fro')
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
           nex = sizeLastDim(Y);
           dY = reshape(0*Y,[this.nOut nex]);
        end
        
        function dtheta = JthetaTmv(this,Z,theta,Y,~,varargin)
            dtheta = [];
        end
       
        
        function dY = JYmv(this,dY,theta,~,~)
           dY = forwardProp(this,theta,dY);
        end
        
        function Z = JYTmv(this,Z,theta,~,~,varargin)
           nex = sizeLastDim(Z);
           szT = this.nIn([this.perm]);
           Z = reshape(Z,[szT nex]);
           Z = ipermute(Z,[this.perm numel(this.perm)+1]);
        end
        end
end


