classdef opCNNBias < LinearOperator
    % bias operator for CNN. Assume bias is along last image data dimension
    % (corresponds to channels)
    
    properties 
        nImg     % size of image data, nImg = [nx, ny, nc] in 2D
    end
    
    methods
        function this = opCNNBias(nImg,varargin)
            % constructor, needs nImg, pool. Other properties can be
            % manipulated with varargin
            this.nImg = nImg;
            this.m = nImg;
            this.n = nImg(end);
            szBias = ones(size(nImg));
            szBias(end) = this.n;
            this.Amv  = @(x) reshape(x,szBias); 
            this.ATmv = @(x) applyTranspose(this,x);
        end
        
      
        function Y = applyTranspose(this,Z)
            
            perm = 1:ndims(Z);
            perm(end-1:end)  = [ndims(Z) ndims(Z)-1];
            Z = permute(Z,perm);
            Z = reshape(Z,[],this.nImg(end));
            Y = vec(sum(Z,1));
        end
    end
end

