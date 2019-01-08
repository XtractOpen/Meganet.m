classdef batchNormLayer2 < abstractMeganetElement
    % classdef batchNormLayer2 < abstractMeganetElement
    %
    % simple implementation of batch normalization layer. Here we normalize
    % the images using essentially
    %
    % Z = (Y-mean(Y,[1,2,4]))./var(Y,[1,2,4])
    % 
    % i.e., we compute the batch statistics across examples for every pixel
    % and channel. By default, running mean and variances are stored and used 
    % in the evaluation. 
    
    properties
        nData       % describe size of data, at least first two dim must be correct.
        isWeight    % boolean, 1 if trainable weights for an affine transformation are provided.
        momentum    % momentum for running statistics. If empty, no stats are kept
        runningMean % running mean
        runningVar  % running variance
        useGPU      % flag for GPU computing
        precision   % flag for precision
        eps
    end
    methods
        function this = batchNormLayer2(nData,varargin)
            if nargin==0
                help(mfilename)
                this.runMinimalExample() 
                return;
            end
            % default properties
            useGPU      = 0;
            precision   = 'double';
            eps         = 1e-4;
            isWeight    = 0;
            momentum    = 0.9;
            runningMean = 0;
            runningVar  = 1;
            for k=1:2:length(varargin)     % overwrites default parameter
                    eval([varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            
            this.useGPU = useGPU;
            this.precision = precision;
            this.nData = nData;
            this.eps = eps;
            this.isWeight=isWeight;
            this.momentum = momentum;
            this.runningMean = runningMean;
            this.runningVar = runningVar;
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
            doDerivative = (nargout>1);
            for k=1:2:length(varargin)     % overwrites default parameter
                eval([varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            dA = [];
            
            if doDerivative || isempty(this.momentum) || isempty(this.runningMean) || isempty(this.runningVar)
                % use batch statistics to normalize
                Ym = compMean(this,Y);
                Y  = Y-Ym;
                Yv = compMean(this,Y.^2);
                Y  = Y./sqrt(Yv+this.eps);
                if not(isempty(this.momentum))
                   this.runningMean = this.momentum*this.runningMean + (1-this.momentum)*Ym;
                   this.runningVar = this.momentum*this.runningVar + (1-this.momentum)*Yv;
                end
            else
                % use running statistics to normalize
                Y = (Y-this.runningMean)./sqrt(this.runningVar+this.eps);
            end
 
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
            [s2,b2] = split(this,ones(this.nTheta,1));
            theta = [s2(:); 0*b2(:);];
            theta = gpuVar(this.useGPU,this.precision,theta);
        end
        
        function Ym = compMean(this,Y)
           szY = size(Y);
           Ym = reshape(Y,[],szY(3),szY(4));
           Ym = mean(Ym,3);
           Ym = mean(Ym,1);
           Ym = reshape(Ym,1,1,szY(3),1);
        end
        
        
        function [dY] = Jthetamv(this,dtheta,theta,Y,~)
            if this.isWeight
                % compute derivative when affine scaling layer is present
                Y  = Y- compMean(this,Y);
                Y  = Y./sqrt(compMean(this,Y.^2)+this.eps);
                
                % scaling
                [ds,db] = split(this,dtheta);
                dY = Y.*ds;
                dY = dY + db;
            else
                dY = 0*Y;
            end
        end
        
        function dtheta = JthetaTmv(this,Z,theta,Y,~)
            if this.isWeight
                % compute derivative when affine scaling layer is present
                Y  = Y-compMean(this,Y);
                Y  = Y./sqrt(compMean(this,Y.^2)+this.eps);
                
                W = Y.*Z;
                dtheta     = vec(sum(sum(sum(W,1),2),4));
                dtheta = [dtheta; vec(sum(sum(sum(Z,1),2),4))];
            else
                dtheta = [];
            end
        end
        
        
        function dY = JYmv(this,dY,theta,Y,~)
            
            Fy  = Y-compMean(this,Y);
            dY = dY-compMean(this,dY);
            den = sqrt(compMean(this,Fy.^2)+this.eps);
            
            dY = dY./den  - (Fy.* (compMean(this,Fy.*dY) ./(den.^3))) ;
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
           Fy  = Y-compMean(this,Y);
           dY = dY-compMean(this,dY);
           den = sqrt(compMean(this,Fy.^2)+this.eps);
           dY = dY./den  - (Fy.* (compMean(this,Fy.*dY) ./(den.^3))) ;
            
        end
        
        
        % ------- functions for handling GPU computing and precision ---- 
        function this = set.useGPU(this,value)
            if (value~=0) && (value~=1)
                error('useGPU must be 0 or 1.')
            else
                this.useGPU  = value;
                [this.runningMean,this.runningVar] = gpuVar(this.useGPU,this.precision,this.runningMean,this.runningVar);
            end
        end
        
        function this = set.precision(this,value)
            if not(strcmp(value,'single') || strcmp(value,'double'))
                error('precision must be single or double.')
            else
                this.precision = value;
                [this.runningMean,this.runningVar] = gpuVar(this.useGPU,this.precision,this.runningMean,this.runningVar);
            end
        end
        function useGPU = get.useGPU(this)
            useGPU = this.useGPU;
        end
        function precision = get.precision(this)
            precision = this.precision;
        end
        function runMinimalExample(this)
           nData = [32 48 4 50];
           layer = feval(mfilename,nData,'isWeight',1,'momentum',[]);
           
           Y = randn(nData);
           dY = randn(size(Y));
           th = initTheta(layer);
           dth = randn(size(th));
           
           [Z] = forwardProp(layer,th,Y,'doDerivative',true);
           dZ  = layer.Jmv(dth,dY,th,Y,[]);
           W = randn(size(Y));
           t1  = W(:)'*dZ(:);
           
           [dthY,dWY] = layer.JTmv(W,th,Y,[]);
           t2 =  dY(:)'*dWY(:) + dth(:)'*dthY(:);
           fprintf('adjoint test: t1=%1.2e\tt2=%1.2e\trel.err=%1.2e\n',t1,t2,abs(t1-t2)/abs(t1));
            for k=1:10
                hh = 2^(-k);
                Zt = layer.forwardProp(th+hh*dth(:),Y+hh*dY,'doDerivative',true);
                
                E0 = norm(Zt(:)-Z(:));
                E1 = norm(Zt(:)-Z(:)-hh*dZ(:));
                
                fprintf('h=%1.2e\tE0=%1.2e\tE1=%1.2e\n',hh,E0,E1);
            end
        end
    end
end


