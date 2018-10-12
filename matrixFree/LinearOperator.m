classdef LinearOperator 
    % basic class for linear operators (i.e. overloading *, \, ...)
    %
    % LinearOperator(A) only supported for 2D matrix A 
    
    properties
        m % output tensor size....NEEDS TO BE DONE
        n % input tensor size ... NEEDS TO BE DONE
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
                if size(this,2)==size(B,1) || isinf(size(B,1)) % ?????Tensor form?????
                   if isinf(size(B,2))
                       n = size(this,2);
                   else
                       n = size(B,2);
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
                szAB = size(B);
            elseif any(isinf(size(B)))
                szAB = size(this);
            elseif  any(size(this)~=size(B))
                error('A and B must have same size');
            else
                szAB = size(B);
            end
            ABf  = @(x) this.Amv(x) + B.Amv(x);
            ABTf = @(x) this.ATmv(x) + B.ATmv(x);
            AB = LinearOperator(szAB(1),szAB(2), ABf, ABTf);
        end
        
        function AB = minus(this,B)
            if isnumeric(B)
                B = LinearOperator(B);
            end
            if any(size(this)~=size(B))
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
            if this.m ~=B.m
                error('hcat - first dimension must agree');
            end
         
            mAB = this.m;
            nAB = this.n + B.n;
            ABf  = @(x) this.Amv(x(1:this.n,:)) + B.Amv(x(this.n+1:end,:));
            ABTf = @(x) [vec(this.ATmv(x)); vec(B.ATmv(x))];
            AB = LinearOperator(mAB,nAB,ABf,ABTf);
        end

        
        function AT = transpose(this)
            AT = LinearOperator(this.n, this.m, this.ATmv, this.Amv);
        end
        
        function AT = ctranspose(this)
            AT = LinearOperator(this.n, this.m, this.ATmv, this.Amv);
        end
        
        function  runMinimalExample(~)
            A = randn(4,6,8,9);
            Aop = LinearOperator(A);
        end
    end
end

