classdef opPool < LinearOperator
    % average pooling operator
    %
    % NOTE: This code is written only using MATLAB built-ins. A more efficient
    % variant that requires compilation can be found in opPoolMCN.
    
    properties 
        nImg     % image resolution
        pool     % pooling factor, pool==2 --> average 2x2 patches
        stride   % stride for pooling operator
        pad      % padding applied to data
    end
    
    methods
        function this = opPool(nImg,pool,varargin)
            % constructor, needs nImg, pool. Other properties can be
            % manipulated with varargin

            if numel(nImg)==2
                % assume number of channels is 1
                nImg = [nImg(:); 1]';
            end
            this.stride = pool;
            this.pad    = 0;
            
            this.nImg = nImg;
            this.pool = pool;
            
            this.m = [this.nImg(1:2)./this.stride this.nImg(3)];
            this.n = this.nImg;
            this.Amv  = @(x) applyPool(this,x);
            this.ATmv = @(x) applyTranspose(this,x);
        end
        
        function Z = applyPool(this,Y)
            % apply average pooling to data Y
            nex = numel(Y)/prod(this.n);
            Y = reshape(Y,this.nImg(1),this.nImg(2),[]);
            % filter for averaging operator
            F = ones(this.pool,this.pool,'like',Y); 
            F = F/sum(F(:));
            Z = convn(Y,F,'same');
            Z = Z(1:this.stride:end,1:this.stride:end,:);
            Z   = reshape(Z,[],nex);
        end
        
        function Y = applyTranspose(this,Z)
            % apply transpose of average pooling operator to Z
            nex = numel(Z)/prod(this.m);
            Z   = reshape(Z,this.nImg(1)./this.pool, this.nImg(1)./this.pool,[]);
            Y   = imresize(Z,2,'nearest')/this.pool^2;
            Y   = reshape(Y,[],nex);
        end
    end
end

