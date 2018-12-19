classdef scalingKernel < handle
    % classdef scalingKernel < handle
    %
    % scales features along dimensions.  The operator is equivalent to
    %
    % kron(s3,kron(s2,s1)) * vec(Y)
    % 
	% Assume Y can be reshaped as reshape(Y,this.nData(1),
	% this.nData(2),nData(3));
    %
    % nData(1) = size(Y,1) - number of features
    % nData(2) = size(Y,2) - number of channels
    % nData(3) = size(Y,3) - number of examples
    %
    % This operator then applies scaling along these dimensions given by
    % vectors s1,s2,s3. The flag isWeight that specifies along which
    % dimensions the weights of the scaling are trained.
    %
    properties
        nData
        isWeight
        useGPU
        precision
    end
    
    methods
        function this = scalingKernel(nData,varargin)
            
            isWeight = [1;1;0]; %all weights trainable
            useGPU = 0;
            precision = 'double';
            for k=1:2:length(varargin)     % overwrites default parameter
                eval([ varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            this.nData = nData;
            this.isWeight=isWeight;
            
            this.useGPU = useGPU;
            this.precision = precision;
            
        end
        
        function n = nTheta(this)
            n = sum(vec(this.nData).*vec(this.isWeight));
        end
        
        function n = sizeFeatIn(this)
            n = this.nData(1:2);
        end
        
        function n = sizeFeatOut(this)
            n = this.nData(1:2);
        end
        
        function n = numelFeatIn(this)
            n = prod(this.nData(1:2));
        end
        
        function n = numelFeatOut(this)
            n = prod(this.nData(1:2));
        end
       
        function A = getOp(this,theta)
            n   = sizeFeatIn(this);
            Af  = @(Y) this.Amv(theta,Y);
            ATf = @(Y) this.Amv(theta,Y);
            A   = LinearOperator(n,n,Af,ATf);
        end
        
        function [s1,s2,s3] = split(this,theta)
           s1 = []; s2 = []; s3 = [];
           
           if this.isWeight(1)
               s1 = reshape(theta(1:this.nData(1)),this.nData(1),1,1);
           end
           
           if this.isWeight(2)
               s2 = reshape(theta(numel(s1)+(1:this.nData(2))),1,this.nData(2),1);
           end
           if this.isWeight(3)
               s3 = reshape(theta(numel(s2)+numel(s1)+(1:this.nData(3))),1,1,this.nData(3));
           end
        end
        
        function [Y,tmp] = Amv(this,theta,Y)
           % Y   = reshape(Y,this.nData(1), this.nData(2),[]);
           nex = size(Y,3); % TODO....do we need nex?
           if this.isWeight(3) && nex ~= this.nData(3)
               error('number of examples (%d) must match number of weights (%d)',nex,this.nData(3));
           end
           [s1,s2,s3] = split(this,theta);
           
           if this.isWeight(1) || not(isempty(s1))
               Y = s1.*Y;
           end
           
           if this.isWeight(2) || not(isempty(s2))
               Y = Y.*s2;
           end
           
           if this.isWeight(3) || not(isempty(s3))
               Y = Y.*s3;
           end
           % Y = reshape(Y,[],nex);
        end
        
        function theta = initTheta(this)
            theta = ones(this.nTheta,1);
        end
        
        function dY = Jthetamv(this,dtheta,theta,Y,~)
           % Y   = reshape(Y,this.nData(1), this.nData(2),[]);
           nex = size(Y,3);
           if this.isWeight(3) && nex ~= this.nData(3)
               error('number of examples (%d) must match number of weights (%d)',nex,this.nData(3));
           end
           [ds1,ds2,ds3] = split(this,dtheta);
           [s1,s2,s3] = split(this,theta);
           
           if this.isWeight(1) || not(isempty(ds1))
               dY = Amv(this,[ds1(:);s2(:);s3(:)],Y);
           else
               dY = zeros(size(Y));
           end
           
           if this.isWeight(2) || not(isempty(ds2))
               dY = dY+ Amv(this,[s1(:);ds2(:);s3(:)],Y);
           end
           
           if this.isWeight(3) || not(isempty(ds3))
               dY = dY+Amv(this,[s1(:);s2(:);ds3(:)],Y);
           end
           % dY = reshape(dY,[],nex);
        end
        
        
        function dtheta = JthetaTmv(this,Z,theta,Y,~)
            % Y   = reshape(Y,this.nData(1), this.nData(2),[]);
            % Z   = reshape(Z,this.nData(1), this.nData(2),[]);
            nex = size(Y,3);
            if this.isWeight(3) && nex ~= this.nData(3)
                error('number of examples (%d) must match number of weights (%d)',nex,this.nData(3));
            end
            [s1,s2,s3] = split(this,theta);
            if isempty(s1); s1 = 1; end
            if isempty(s2); s2 = 1; end
            if isempty(s3); s3 = 1; end
            
            W = Y.*Z;
            dtheta = [];
            if this.isWeight(1)
                dtheta = [dtheta; vec(sum(sum((W.*s2).*s3,2),3))];
            end
            if this.isWeight(2)
                dtheta = [dtheta; vec(sum(sum((s1.*W).*s3,1),3))];
            end
            if this.isWeight(3)
                dtheta = [dtheta; vec(sum(sum((s1.*W).*s2,1),2))];
            end
            
       end
       
       function this = set.useGPU(this,value)
            if (value~=0) && (value~=1)
                error('useGPU must be 0 or 1.')
            else
                this.useGPU = value;
            end
        end
        function this = set.precision(this,value)
            if strcmp(value,'single') && strcmp(value,'double')
                error('precision must be single or double.')
            else
                this.precision = value;
            end
        end
    end
end

