classdef opPoolMCN < LinearOperator
    % average pooling operator based on MatConvNet
    
    properties
        nImg
        pool
        stride
        pad
%         method
    end
    
    methods
        function this = opPoolMCN(nImg,pool,varargin)

            this.stride = pool;
            this.pad    = 0;
            
            this.nImg = nImg;
            this.pool = pool;
            
            this.m = prod(this.nImg(1:2)./this.stride)*this.nImg(3);
            this.n = prod(this.nImg);
            this.Amv = @(x) applyPool(this,x);
            this.ATmv = @(x)applyTranspose(this,x);
        end
        
        function Z = applyPool(this,Y)
            nex = numel(Y)/this.n;
            Y   = reshape(Y,[this.nImg nex]);
            Z   = vl_nnpool(Y,this.pool, 'stride',this.stride,'method','avg','pad',this.pad);
            Z   = reshape(Z,[],nex);
        end
        
        function Y = applyTranspose(this,Z)
            nex = numel(Z)/this.m;
            Z   = reshape(Z,[this.nImg(1:2)./this.pool this.nImg(3) nex]);
            Yd  = zeros([this.nImg nex],'like',Z);
            Y   = vl_nnpool(Yd,this.pool, Z,'stride',this.stride,'method','avg','pad',this.pad);
            Y   = reshape(Y,[],nex);
        end
    end
end

