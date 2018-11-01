classdef opPoolMCN < LinearOperator
    % average pooling operator using MatConvNet
    %
    % NOTE: This code requires compiled binaries of vl_nnpool for the
    % current operating system.  Instructions can be found at
    %           http://www.vlfeat.org/matconvnet/install/
    %
    % For a less efficient but equivalent variant, see opPool.m
    
    properties 
        nImg     % image resolution
        pool     % number of pooling steps
        stride   % stride for pooling operator
        pad      % padding applied to data
    end
    
    methods
        function this = opPoolMCN(nImg,pool,varargin)
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
            this.ATmv = @(x) applyPoolTranspose(this,x);
        end
        
        function Z = applyPool(this,Y)
            % apply average pooling to data Y
            Z   = vl_nnpool(Y,this.pool, 'stride',this.stride,'method','avg','pad',this.pad);
        end
        
        function Y = applyPoolTranspose(this,Z)
            % apply transpose of average pooling operator to Z
            nex = numel(Z)/prod(this.m);
            Yd  = zeros([this.nImg nex],'like',Z);
            Y   = vl_nnpool(Yd,this.pool, Z,'stride',this.stride,'method','avg','pad',this.pad);
        end
    end
end

