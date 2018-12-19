classdef tvNormLayer < abstractMeganetElement
    % classdef tvNormLayer < abstractMeganetElement
    %
    % tailored implementation of total variation normalization layer applicable to CNNs. 
    %
    % The idea is to normalize pixel-wise by the length of the feature
    % vector, i.e.,
    %
    %  Z = diag( sqrt( sum(Y.^2,3)+eps)) Y
    %
    properties
        nData       % describe size of data, at least first three dim must be correct.
        isWeight    % boolean, 1 if trainable weights for an affine transformation are provided.
        useGPU      % flag for GPU computing 
        precision   % flag for precision 
        eps
    end
    methods
        function this = tvNormLayer(nData,varargin)
            if nargin==0
                help(mfilename)
                this.runMinimalExample
                return;
            end
            useGPU     = 0;
            precision  = 'double';
            eps = 1e-4;
            isWeight = 0;
            for k=1:2:length(varargin)     % overwrites default parameter
                    eval([varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            
            this.useGPU = useGPU;
            this.precision = precision;
            this.nData = nData;
            this.eps = eps;
            this.isWeight=isWeight;
        end
        
        function runMinimalExample(this)
            data = load('clown');
            Y    = data.X;
            nImg = size(Y);
            
            K     = convFFT(nImg, [3 3 1 2]);
            dx    = [0 0 0; -1 1 0; 0 0 0];
            dy    = [0 1 0; 0 -1 0; 0 0 0];
            theta = [dx(:); dy(:)];
            nL    = feval(mfilename,[nImg 2]);
            layer = doubleSymLayer(K,'normLayer1', nL, 'activation',@identityActivation);
            nt = 100; h = 0.2;
            net   = ResNN(layer,nt,h);
            
            YN = forwardProp(net,repmat(theta,nt,1),Y);
            
            
            figure(1); clf;
            subplot(2,2,1);
            imagesc(Y); 
            title('input image')
            
            subplot(2,2,2);
            imagesc(YN);
            title('tv denoised image')
            
            subplot(2,2,3);
            surf(Y,'EdgeColor','none'); 
            title('input image, surf plot')
            
            subplot(2,2,4);
            surf(YN,'EdgeColor','none');
            title('tv denoised image, surf plot')
                
        end
        
        function [s,b] = split(this,theta)
            if this.isWeight
                s = reshape(theta(1:this.nData(3)),1,1,this.nData(3),1);
                cnt = numel(s);
                b = reshape(theta(cnt+(1:this.nData(3))),1,1,this.nData(3),1);
            else
                s = []; b = [];
            end
        end
        
        function [Y,dA] = forwardProp(this,theta,Y,varargin)
           dA = [];
           % normalization
           Y  = Y./sqrt(sum(Y.^2,3)+this.eps);
           
           if this.isWeight
               % affine scaling along channels
               [s,b] = split(this,theta);           
               Y = Y.*s;
               Y = Y + b;
           end
        end
        
        
        function n = nTheta(this)
            n = this.isWeight*2*this.nData(3);
        end
        
        function n = sizeFeatIn(this)
            n = this.nData(1:3);
        end
        
        function n = sizeFeatOut(this)
            n = this.nData(1:3);
        end
        
        function theta = initTheta(this)
            [s,b] = split(this,ones(this.nTheta,1));
            theta = [s(:); 0*b(:);];
            theta = gpuVar(this.useGPU,this.precision,theta);
        end
        
        
        function [dY] = Jthetamv(this,dtheta,theta,Y,~)
           
           if this.isWeight
               % compute derivative when affine scaling layer is present
               dY  = Y./sqrt(sum(Y.^2,3)+this.eps);
               [ds,db] = split(this,dtheta);
               dY = dY.*ds;
               dY = dY + db;
           else
               dY = 0*Y;
           end
        end
        
        function dtheta = JthetaTmv(this,Z,theta,Y,~)
            if this.isWeight
                % compute derivative when affine scaling layer is present
                Y  = Y./sqrt(sum(Y.^2,3)+this.eps);
                
                W = Y.*Z;
                dtheta     = vec(sum(sum(sum(W,1),2),4));
                dtheta = [dtheta; vec(sum(sum(sum(Z,1),2),4))];
            else
                dtheta = [];
            end
        end
       
        
        function dY = JYmv(this,dY,theta,Y,~)
            
            den = sqrt(sum(Y.^2,3)+this.eps);
            dY = dY./den  - (Y.* (sum(Y.*dY,3) ./(den.^3))) ;
            
            if this.isWeight
                % affine scaling
                s = split(this,theta);
                dY = dY.*s;
            end
        end
        
        function dY = JYTmv(this,dY,theta,Y,~)
            
            if this.isWeight
                % affine scaling
                s = split(this,theta);
                dY = dY.*s;
            end
            
           % normalization
           den = sqrt(sum(Y.^2,3)+this.eps);
           
           tt = sum(Y.*dY,3) ./(den.^3);
           dY = dY./den  - Y.*tt;
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


