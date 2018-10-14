classdef LinearOperator 
    % basic class for linear operators (i.e. overloading *, \, ...)
    %
    % LinearOperator(A) only supported for 2D matrix A 
    
    properties
        m % output tensor size
        n % input tensor size
        Amv
        ATmv        
    end
    
    methods
        function this = LinearOperator(varargin)
            if nargin==0
                this.runMinimalExample;
                return;
            end
             
           if nargin==1 && isnumeric(varargin{1})
               A = varargin{1};
               [this.m,this.n] = size(A);
               this.Amv = @(x) A*x;
               this.ATmv = @(x) A'*x;
           elseif nargin==3
               A = varargin{1};
               this.m = varargin{2};
               this.n = varargin{3};
               this.Amv = @(x) reshape(A,prod(this.m),prod(this.n))  * x;
               this.ATmv= @(x) reshape(A,prod(this.m),prod(this.n))' * x;
           elseif nargin>=4
               this.m = varargin{1};
               this.n = varargin{2};
               this.Amv = varargin{3};
               this.ATmv = varargin{4};
           else
               error('%s - invalid number of inputs',mfilename);
           end
        end
        
        % views A as a matrix with dimesnions prod(m) and prod(n)
        function szA = size(this,dim)
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
            if isscalar(B)
                Ax = LinearOperator(this.m,this.n,@(x) B*this.Amv(x), @(x) B*this.ATmv(x));
            elseif isscalar(this)
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
            AB = LinearOperator(ABm,ABn, ABf, ABTf);
        end
        
        function AB = minus(this,B)
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
                nAB = this.n + B.n;
                ABf  = @(x) this.Amv(x(1:prod(this.n),:)) + B.Amv(x(prod(this.n)+1:end,:));
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
            
            
%             A=randn(4,6);
%             Aop = LinearOperator(A);
        end
    end
end

