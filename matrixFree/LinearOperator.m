classdef LinearOperator 
    % basic class for linear operators (i.e. overloading *,...)
    %
    % There are several use-cases for linear operators in the package. 
    %
    % Examples:
    %   1) Aop    = LinearOperator(A); % A is a matrix (sparse, dense, ...).
    %   2) IdOp   = LinearOperator(10,10, @(x) x, @(x) x); 
    %                   matrix free representation of the identity, see
    %                   also opEye(10);
    %   3) convOp = LinearOperator([24 24 4], [24 24 2]), @(Y) ..., @(Z) ...)
    %                   matrix-free representation of a convolution operator. 
    %                   convOp takes images of size 24 x 24 x 2 (i.e., two input channels)
    %                   and yields new images with 4 channels. This operator can
    %                   handle more than one example at a time, i.e., 
    %                   convOp*randn(24,24,2,30) will convolve all 30 input images.
    %  4) JYop    = getJOp(layer,theta,Y,tmp) 
    %                  returns Jacobian w.r.t. input features for a layer.
    %                  In CNN layers, JYOp.n and JYop.m will be the sizes of
    %                  the input tensors and output tensors, respectively,
    %                  i.e., size(Y) = JYop.n. numel(JYop.n) == 4. 
    %  5) JthOp  = getJthetaOp(layer,theta,Y,tmp)
    %                  returns Jacobian w.r.t. theta for a layer.
    %                  JthOp.n will be equal to numel(theta). The type and
    %                  size of JthOp.m will be the number of output
    %                  features of the layer. In CNNs, JthOp.m will be a
    %                  vector.
    % 
    % LinearOperator(A) only supported for 2D matrix A 
    
    properties
        m       % output size, vector-valued for tensors, scalar for vectors
        n       % input size, vector-valued for tensors, scalar for vectors
        Amv     % function handle for computing A*v
        ATmv    % function handle for computing A'*w    
    end
    
    methods
        function this = LinearOperator(varargin)
            if nargin==0
                this.runMinimalExample;
                return;
            end
             
           if nargin==1 && isnumeric(varargin{1})
               A = varargin{1};
               this.m = size(A,1);
               this.n = size(A,2);
               this.Amv = @(x)  A  * x;
               this.ATmv = @(x) A' * x;
           elseif nargin>=4
               this.m = varargin{1};
               this.n = varargin{2};
               this.Amv = varargin{3};
               this.ATmv = varargin{4};
           else
               error('%s - invalid number of inputs',mfilename);
           end
        end
        
        function szA = size(this,dim)
            % computes the size of linear operator. Here, szA will always be a two-dimensional
            % vector. If this.m or this.n are vector-valued we take the prod.
            if nargin==1
                szA = [prod(this.m), prod(this.n)];
            elseif nargin==2 && dim==1
                szA = prod(this.m);
            elseif nargin==2 && dim==2
                szA = prod(this.n);
            end    
        end
        
        function nn = numel(this)
            nn = prod(size(this));
        end
        
        function Ax = mtimes(this,B)
            % multiply a LinearOperator with another object B. Action
            % depends on the type of B
            
            if isscalar(B)
                Ax = LinearOperator(this.m,this.n,@(x) B*this.Amv(x), @(x) B*this.ATmv(x));
            elseif isscalar(this) && isa(B,'LinearOperator')
                Ax = LinearOperator(B.m,B.n,@(x) this*B.Amv(x), @(x) this*B.ATmv(x));
            elseif isa(B,'LinearOperator')
                if all(this.n==B.m) || all(isinf(B.m))  % all() combines the logical values 
                   if all(isinf(B.m))
                       n = this.n;
                   else
                       n = B.n;
                   end
                   Ax = LinearOperator(this.m,n, @(x) this*(B*x), @(x) B'*(this'*x));
                else
                    error('Inner dimensions must agree');
                end
            else
                Ax = this.Amv(B);
            end
        end
        
        function AB = plus(this,B)
            % adds an object B to a LinearOperator 
            if isnumeric(this)
                this = LinearOperator(this);
            end
            if isnumeric(B)
                B = LinearOperator(B);
            end
            if isempty(this)
                AB = B;
                return;
            end
            
            if any(isinf(size(this)))
                % szAB = size(B);
                ABm  = B.m;
                ABn  = B.n;
            elseif any(isinf(size(B)))
                % szAB = size(this);
                ABm  = this.m;
                ABn  = this.n;
            elseif  any( [this.m this.n] ~= [B.m B.n] )
                error('A and B must have same size');
            else
                % szAB = size(B);
                ABm  = B.m;
                ABn  = B.n;
            end
            ABf  = @(x) this.Amv(x) + B.Amv(x);
            ABTf = @(x) this.ATmv(x) + B.ATmv(x);
            AB = LinearOperator(ABm, ABn, ABf, ABTf);
        end
        
        function AB = minus(this,B)
            % subtracts an object B to a LinearOperator 
            if isnumeric(B)
                B = LinearOperator(B);
            end
            if any( [this.m this.n] ~= [B.m B.n] )
                error('A and B must have same size');
            end
            ABf  = @(x) this.Amv(x) - B.Amv(x);
            ABTf = @(x) this.ATmv(x) - B.ATmv(x);
            AB = LinearOperator(this.m,this.n, ABf, ABTf);
        end
        
        function AB = blkdiag(this,varargin)
            ops = {this,varargin{:}};
            AB = opBlkdiag(ops{:});
        end
            

         function AB = hcat(this,B)
            if isnumeric(B)
                B = LinearOperator(B);
            end
            if any( this.m ~= B.m )
                error('hcat - first dimension/argument m must agree');
            end
         
            if isscalar(this.n) && isscalar(B.n)
                mAB = this.m;
                nAB = prod(this.n) + prod(B.n);
                ABf  = @(x) this.Amv(reshape(x(1:prod(this.n)),this.n)) + B.Amv(reshape(x(prod(this.n)+1:end),B.n));
                ABTf = @(x) [vec(this.ATmv(x)); vec(B.ATmv(x))];
                AB = LinearOperator(mAB,nAB,ABf,ABTf);
            else
                error('hcat - second dimension/argument n must be 1 dimension');
            end
        end

        
        function AT = transpose(this)
            AT = LinearOperator(this.n, this.m, this.ATmv, this.Amv);
        end
        
        function AT = ctranspose(this)
            AT = LinearOperator(this.n, this.m, this.ATmv, this.Amv);
        end
        
        function  runMinimalExample(~)
            A = randn(4,6,9);
            B = randn(4,6,9);
            
            Amv  = @(x) reshape(A,4*6,9)*x;
            ATmv = @(x) reshape(A,4*6,9)'*x;
            Bmv  = @(x) reshape(B,8*9,9)*x;
            BTmv = @(x) reshape(B,8*9,9)'*x;
            Aop = LinearOperator([4 6],9,Amv, ATmv);
            Bop = LinearOperator([4 6],9,Bmv, BTmv);
            % X = mtimes(Aop,Bop);
            X = hcat(Aop,Bop);
            Y = Aop + Bop;

        end
    end
end

