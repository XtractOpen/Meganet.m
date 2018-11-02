classdef opCNNavg < LinearOperator
    % operator that replicates an averaging connector block
    % useful for a final connector block to generalize the model
    
    properties 
        nImg     % size of image data, nImg = [nx, ny, nc] in 2D
    end
    
    methods
        function this = opCNNavg(nImg,varargin)
            % constructor, needs nImg, pool. Other properties can be
            % manipulated with varargin
            this.nImg = nImg;
            this.m = nImg;
            this.n = nImg(end);
            op = struct('sz', num2cell( this.m ));
            % szOp = size(nImg);
            % B = reshape(C,op.sz,[]);
            % szOp(end) = this.n;
            this.Amv  = @(x) reshape(x,op.sz,[]); 
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
