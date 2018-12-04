classdef affineScalingLayer < abstractMeganetElement
    % classdef affineScalingLayer < abstractMeganetElement
    %
    % Scales and shifts the 3D feature tensor along each dimension. This is
    % useful, e.g., in batch normalization. 
    %
    % kron(s3,kron(s2,s1)) * vec(Y) + kron(b3,kron(e2,e1)) +
    % kron(e3,kron(b2,e1)) + kron(e3,kron(e2,b1));
    %
    properties
        nData       % describe size of data, at least first two dim must be correct.
        isWeight     % transformation type
        useGPU      % flag for GPU computing 
        precision   % flag for precision 
    end
    methods
        function this = affineScalingLayer(nData,varargin)
            if nargin==0
                help(mfilename)
                return;
            end
            isWeight   = [1;1;0]; %all weights trainable
            useGPU     = 0;
            precision  = 'double';
            for k=1:2:length(varargin)     % overwrites default parameter
                    eval([varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            
            this.useGPU = useGPU;
            this.precision = precision;
            this.nData = nData;
            this.isWeight = isWeight;
            
        end
        function [s1,b1,s2,b2,s3,b3] = split(this,theta)
            s1 = []; s2 = []; s3 = [];
            b1 = []; b2 = []; b3 = [];
           
           cnt = 0;
           if this.isWeight(1)
               s1 = reshape(theta(1:this.nData(1)),this.nData(1),1,1);
               cnt = cnt + numel(s1);
               b1 = reshape(theta(cnt+(1:this.nData(1))),this.nData(1),1,1);
               cnt = cnt + numel(b1);
           end
           
           if this.isWeight(2)
               s2 = reshape(theta(cnt+(1:this.nData(2))),1,this.nData(2),1);
               cnt = cnt+numel(s2);
               b2 = reshape(theta(cnt+(1:this.nData(2))),1,this.nData(2),1);
               cnt = cnt+numel(b2);
           end
           if this.isWeight(3)
               s3 = reshape(theta(cnt+(1:this.nData(3))),1,1,this.nData(3));
               cnt = cnt + numel(s3);
               b3 = reshape(theta(cnt+(1:this.nData(3))),1,1,this.nData(3));
           end
        end
        
        function [Y,dA] = forwardProp(this,theta,Y,varargin)
            Y   = reshape(Y,this.nData(1), this.nData(2),[]); dA = [];
           nex = size(Y,3);
           if this.isWeight(3) && nex ~= this.nData(3)
               error('number of examples (%d) must match number of weights (%d)',nex,this.nData(3));
           end
           [s1,b1,s2,b2,s3,b3] = split(this,theta);
           
           Y = scaleCoord(this,Y,s1,s2,s3);
           
           if this.isWeight(1) || not(isempty(b1))
               Y = Y + b1;
           end
           
           if this.isWeight(2) || not(isempty(b2))
               Y = Y + b2;
           end
           
           if this.isWeight(3) || not(isempty(b3))
               Y = Y + b3;
           end
           
           Y = reshape(Y,[],nex);
        end
        
        function Y = scaleCoord(this,Y,s1,s2,s3)
           if  not(isempty(s1))
               Y = s1.*Y;
           end
           
           if  not(isempty(s2))
               Y = Y.*s2;
           end
           
           if not(isempty(s3))
               Y = Y.*s3;
           end
        end
        
        function n = nTheta(this)
            n = 2*sum(vec(this.nData).*vec(this.isWeight));
        end
        
        function n = sizeFeatIn(this)
            n = this.nData(1:2);  
        end
        
        function n = sizeFeatOut(this)
            n = this.nData(1:2);
        end
        
        function theta = initTheta(this)
            [s1,b1,s2,b2,s3,b3] = split(this,ones(this.nTheta,1));
            theta = [s1(:); 0*b1(:); s2(:); 0*b2(:); s3(:); 0*b3(:)];
            theta = gpuVar(this.useGPU,this.precision,theta);
        end
        
        
        function dY = Jthetamv(this,dtheta,theta,Y,~)
           % Y   = reshape(Y,this.nData(1), this.nData(2),[]);
           % nex = size(Y,3);
           if this.isWeight(3) && nex ~= this.nData(3)
               error('number of examples (%d) must match number of weights (%d)',nex,this.nData(3));
           end
           [ds1,db1,ds2,db2,ds3,db3] = split(this,dtheta);
           [s1,b1,s2,b2,s3,b3] = split(this,theta);
           
           if this.isWeight(1) || not(isempty(ds1))
               dY = scaleCoord(this,Y,ds1,s2,s3);
           else
               dY = 0*Y;
           end
           
           if this.isWeight(2) || not(isempty(ds2))
               dY = dY+ scaleCoord(this,Y,s1,ds2,s3);
           end
           
           if this.isWeight(3) || not(isempty(ds3))
               dY = dY+scaleCoord(this,Y,s1,s2,ds3);
           end
           
           if this.isWeight(1)
               dY = dY + db1;
           end
           
           if this.isWeight(2)
               dY = dY + db2;
           end
           
           if this.isWeight(3)
               dY = dY + db3;
           end
           
        end
        
        function dtheta = JthetaTmv(this,Z,theta,Y,~)
            Y   = reshape(Y,this.nData(1), this.nData(2),[]); % TODO: are these reshapes necessary?
            Z   = reshape(Z,this.nData(1), this.nData(2),[]);
            nex = size(Y,3);
            if this.isWeight(3) && nex ~= this.nData(3)
                error('number of examples (%d) must match number of weights (%d)',nex,this.nData(3));
            end
            [s1,~,s2,~,s3,~] = split(this,theta);
            if isempty(s1); s1 = 1; end
            if isempty(s2); s2 = 1; end
            if isempty(s3); s3 = 1; end
            
            W = Y.*Z;
            dtheta = [];
            if this.isWeight(1)
                dtheta = [dtheta; vec(sum(sum((W.*s2).*s3,2),3))];
                dtheta = [dtheta; vec(sum(sum(Z,2),3))];
            end
            if this.isWeight(2)
                dtheta = [dtheta; vec(sum(sum((s1.*W).*s3,1),3))];
                dtheta = [dtheta; vec(sum(sum(Z,1),3))];
            end
            if this.isWeight(3)
                dtheta = [dtheta; vec(sum(sum((s1.*W).*s2,1),2))];
                dtheta = [dtheta; vec(sum(sum(Z,1),2))];
            end
        end
       
        
        function dY = JYmv(this,dY,theta,~,~)
           szdY = size(dY);
           dY   = reshape(dY,this.nData(1), this.nData(2),[]); 
           nex = size(dY,3);
           if this.isWeight(3) && nex ~= this.nData(3)
               error('number of examples (%d) must match number of weights (%d)',nex,this.nData(3));
           end
           [s1,~,s2,~,s3,~] = split(this,theta);
           
           if this.isWeight(1) || not(isempty(s1))
               dY = s1.*dY;
           end
           
           if this.isWeight(2) || not(isempty(s2))
               dY = dY.*s2;
           end
           
           if this.isWeight(3) || not(isempty(s3))
               dY = dY.*s3;
           end
           
           dY = reshape(dY,szdY);
        end
        
        function Z = JYTmv(this,Z,theta,~,~)
           szZ = size(Z);
           Z   = reshape(Z,this.nData(1), this.nData(2),[]); % TODO reshapes unnecessary?
           nex = size(Z,3);
           if this.isWeight(3) && nex ~= this.nData(3)
               error('number of examples (%d) must match number of weights (%d)',nex,this.nData(3));
           end
           [s1,~,s2,~,s3,~] = split(this,theta);
           Z = scaleCoord(this,Z,s1,s2,s3);
           Z = reshape(Z,szZ);
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


