classdef batchNormLayer < abstractMeganetElement
    % classdef batchNormLayer < abstractMeganetElement
    %
    % simple implementation of batch normalization layer. Here we normalize
    % the images using essentially
    %
    % Z = (Y-mean(Y,4))./var(Y,4)
    % 
    % i.e., we compute the batch statistics across examples for every pixel
    % and channel. See batchNormLayer2 for another version that also
    % averages all the pixels. 
    
    properties
        nData       % describe size of data, at least first two dim must be correct.
        isWeight    % boolean, 1 if trainable weights for an affine transformation are provided.
        useGPU      % flag for GPU computing 
        precision   % flag for precision 
        eps
    end
    methods
        function this = batchNormLayer(nData,varargin)
            if nargin==0
                help(mfilename)
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
           Y  = Y-mean(Y,4);
           Y  = Y./sqrt(mean(Y.^2,4)+this.eps);
           
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
        
        
        function [dY] = Jthetamv(this,dtheta,theta,Y,~)
            if this.isWeight
                % compute derivative when affine scaling layer is present
                Y  = Y-mean(Y,4);
                Y  = Y./sqrt(mean(Y.^2,4)+this.eps);
                
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
                Y  = Y-mean(Y,4);
                Y  = Y./sqrt(mean(Y.^2,4)+this.eps);
                
                W = Y.*Z;
                dtheta     = vec(sum(sum(sum(W,1),2),4));
                dtheta = [dtheta; vec(sum(sum(sum(Z,1),2),4))];
            else
                dtheta = [];
            end
        end
        
        
        function dY = JYmv(this,dY,theta,Y,~)
            
            Fy  = Y-mean(Y,4);
            FdY = dY-mean(dY,4);
            den = sqrt(mean(Fy.^2,4)+this.eps);
            
            dY = FdY./den  - (Fy.* (mean(Fy.*FdY,4) ./(den.^3))) ;
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
           Fy  = Y-mean(Y,4);
           dY = dY-mean(dY,4);
           den = sqrt(mean(Fy.^2,4)+this.eps);
           
           tt = mean(Fy.*dY,4) ./(den.^3);
           dY = dY./den;
           clear den;
           dY = dY - Fy.*tt;
        end
        
        
        % ------- functions for handling GPU computing and precision ---- 
        function set.useGPU(this,value)
            if (value~=0) && (value~=1)
                error('useGPU must be 0 or 1.')
            else
                this.useGPU  = value;
            end
        end
        function set.precision(this,value)
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


