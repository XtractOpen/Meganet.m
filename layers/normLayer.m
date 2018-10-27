classdef normLayer < abstractMeganetElement
    % classdef normLayer < abstractMeganetElement
    %
    % Normalizes features over specified dimensions. Assume features are
    % given by a 3D tensor of size nData
    %
    
    properties
        nData       % size of data #pixels x #channels x #examples
        doNorm      % specifies dimensions along which to normalize
        eps         % smoothing factor
        useGPU      % flag for GPU computing (derived from trafo)
        precision   % flag for precision (derived from trafo)
        
    end
    methods
        function this = normLayer(nData,varargin)
            if nargin==0 && nargout==0
               this.runMinimalExample();
               return;
            end
            if nargin==0
                help(mfilename)
                return;
            end
            doNorm     = [0;1;0]; % normalize along channels
            eps        = 1e-3;
            useGPU     = [];
            precision  = [];
            for k=1:2:length(varargin) % overwrites default parameter
                    eval([varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            if not(isempty(useGPU))
                this.useGPU = useGPU;
            end
            if not(isempty(precision))
                this.precision=precision;
            end
            this.eps = eps;
            this.doNorm = doNorm;
            this.nData = nData;
        end
        
        function Y = compMean(this,Y)
            % computes mean along all dimensions in this.doNorm
            nEl = 1; 
            
            for k=1:3
                if this.doNorm(k)
                    nEl = nEl/size(Y,k);
                    Y = sum(Y,k);
                end
            end
            Y = nEl*Y;
        end
        
        function [Y,dA] = forwardProp(this,~,Y,varargin)
            dA = [];
            for k=1:2:length(varargin)     % overwrites default parameter
                eval([varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            
            % subtract mean across pixels
            Y  = Y-compMean(this,Y);
            
            % normalize
            Y  = Y./sqrt(compMean(this,Y.^2)+this.eps);
        end
        
        function n = nTheta(this)
            n = 0;
        end
        
        function n = sizeFeatIn(this)
            n = this.nData;
        end
        
        function n = sizeFeatOut(this)
            n = this.nData;
        end
        
        function theta = initTheta(this)
           theta = [];
        end
        
        
        function [dZ] = Jthetamv(this,dtheta,theta,Y,dA)
            dZ = 0*Y;
        end
        
        function [dZ] = JYmv(this,dY,theta,Y,dA)
            %
            % Z(Y) = (Y- Av*Y)./sqrt((Y-Av*Y).^2 + eps)
            %
            % write as
            %
            % Z(Y) = F(Y)./sqrt(Av*F(Y).^2 + eps), F(Y) = Y-Av*Y;
            %
            % Z'(F) =  1./sqrt(Av*F(Y).^2+eps) - F(Y).*(Av*F(Y))./(Av*F(Y).^2+eps)^(3/2)
            %
            % F'(Y) = I - Av;
            %
            % T = 1./sqrt(A*(T).^2 + eps)
            %
            % A = A*diag(Fy)*Fy --> A' = A*diag(Fy)
            
%             Y = reshape(Y,this.nData(1), this.nData(2), this.nData(3), []);
%             dY = reshape(dY,this.nData(1), this.nData(2), this.nData(3), []);
%             
            Fy  = Y-compMean(this,Y);
            FdY = dY-compMean(this,dY);
            den = sqrt(compMean(this,Fy.^2)+this.eps);
            
            dZ = FdY./den  - ( Fy.* (compMean(this,Fy.*FdY) ./(den.^3))) ;
        end
        
        function [dZ] = Jmv(this,~,dY,theta,Y,dA)
            if numel(dY)==1 && dY==0
                dZ = 0*Y;
            else
                dZ = this.JYmv(dY,theta,Y,dA);
            end
            
        end
        
        function [dtheta,dY] = JTmv(this,Z,~,theta,Y,dA,doDerivative)
            dtheta = [];
            if not(exist('doDerivative','var')) || isempty(doDerivative)
               doDerivative =[1;0]; 
            end
            dY     = JYTmv(this,Z,[],theta,Y,dA);
            if nargout==1 && all(doDerivative==1)
                dtheta = [dtheta(:);dY(:)];
            end
        end
        
        function dtheta = JthetaTmv(this,Z,theta,Y,dA)
            dtheta = [];
        end
        
        function dY = JYTmv(this,Z,~,theta,Y,dA)
                        
            Fy  = Y-compMean(this,Y);
            FdY = Z-compMean(this,Z);
            den = sqrt(compMean(this,Fy.^2)+this.eps);
            
            dY = FdY./den  - ( Fy.* (compMean(this,Fy.*FdY) ./(den.^3)));
            clear Fy FdY;
            dY = dY - compMean(this,dY);
        end
        
        function runMinimalExample(this)
            
            nFeat = 10;
            nfilt = 5;
            Y0 = randn(nfilt*nFeat,100)+10;
            Y0 = Y0.*(1:size(Y0,2));
            
            L = batchNormLayer(nfilt,nFeat);
            
            
            [Y,dA]   = L.forwardProp([],Y0);
            dY0  = randn(size(Y0));
            
            dY = L.Jmv([],dY0,[],Y0,dA);
            for k=1:14
                hh = 2^(-k);
                
                Yt = L.forwardProp([],Y0+hh*dY0);
                
                E0 = norm(Yt(:)-Y(:));
                E1 = norm(Yt(:)-Y(:)-hh*dY(:));
                
                fprintf('h=%1.2e\tE0=%1.2e\tE1=%1.2e\n',hh,E0,E1);
            end
            
            W = randn(size(Y));
            t1  = W(:)'*dY(:);
            
            [~,dWY] = L.JTmv(W,[],[],Y0,dA);
            t2 = dY0(:)'*dWY(:);
            
            fprintf('adjoint test: t1=%1.2e\tt2=%1.2e\terr=%1.2e\n',t1,t2,abs(t1-t2));
            
        end
        
        % ------- functions for handling GPU computing and precision ---- 
        function this = set.useGPU(this,value)
            if (value~=0) && (value~=1)
                error('useGPU must be 0 or 1.')
            end
        end
        function this = set.precision(this,value)
            if not(strcmp(value,'single') || strcmp(value,'double'))
                error('precision must be single or double.')
            end
        end
        function useGPU = get.useGPU(this)
            useGPU = [];
        end
        function precision = get.precision(this)
            precision = '';
        end
    end
end


