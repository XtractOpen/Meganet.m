classdef GMRES
    
    properties
        m = []                       % dimension of Krylov subspace
        % https://www.cs.cornell.edu/~bindel/class/cs6210-f16/lec/2016-11-16.pdf
        out = 0
        tol = 1e-10             % not used
    end
    
    methods
        function this = GMRES(varargin)

            for k = 1:2:length(varargin)     % overwrites default parameter
                this.(varargin{k})= varargin{k+1};
            end

        end
        
        function [x,para,z,V,H,funEvals,flag] = solve(this,A,b,x0,~)
            % Adapted from Saad, Iterative Methods, 2003
            %
            % Non-restarted gmres to solve A*x = b 
            %
            % Inputs:
            %   A:   n x n matrix
            %   b:   n x 1 right-hand side
            %   x0:  n x 1 initial guess
            %
            % Outputs:
            %   x : approximate solution at k-th iteration 
            %   z : coefficients of x, x=V(:,1:k)*z
            %   V : n x (k+1) matrix of basis vectors
            %   H : k x k upper Hessenberg matrix
            
            funEvals = 0; % every application of A requires a forward and backward pass
            if isempty(this.m)
                this.m = numel(b);
            end
            
            useGPU = isa(b,'gpuArray');
            if isa(gather(b),'single')
                precision = 'single';
            else
                precision = 'double';
            end
            
            if not(exist('x0','var')) || isempty(x0)
                x0 = zeros(size(b),'like',b);
            end
            
            n = length(x0);
            
            % initialize
            V = zeros(n,this.m+1);
            H = zeros(this.m+1,this.m);
            Hm = zeros(this.m+1,this.m);
            
            % compute residual 
            r0 = b - A * x0;
            r0 = double(gather(r0));
            funEvals = funEvals + 2;
            
            beta    = norm(r0); 
            V(:,1)  = r0 / beta;
            
            e1      = eye(this.m,1);
            [cs,sn] = deal(zeros(this.m,1));
            s       = beta * e1;
            % main iteration
            flag = -1;
            for j = 1:this.m
                vj = gpuVar(useGPU,precision,V(:,j));
                w = A * vj;
                w = double(gather(w));
                funEvals = funEvals + 2;
                
                for i = 1:j % Gram-Schmidt
                    H(i,j) = dot(V(:,i), w);
                    w = w - H(i,j) * V(:,i);
                end
                H(j+1,j) = norm(w);
                V(:,j+1) = w / H(j+1,j);
                Hm(:,j)  = H(:,j);
                
                % Givens rotations
                for i = 1:j-1
                    temp      = cs(i) * Hm(i,j) + sn(i) * Hm(i+1,j);
                    Hm(i+1,j) = -sn(i) * Hm(i,j) + cs(i) * Hm(i+1,j);
                    Hm(i,j)   = temp;
                end
                Hm(j+1,j) = norm(w);
                
                % approximate residual norm
                [cs(j),sn(j)] = symOrtho(this,Hm(j,j),Hm(j+1,j));
                s(j+1)        = -sn(j) * s(j);
                s(j)          = cs(j) * s(j);
                Hm(j,j)       = cs(j) * Hm(j,j) + sn(j) * Hm(j+1,j);
                Hm(j+1,j)     = 0;
                
                err = abs(s(j+1)) / beta;
                if this.out > 1
                    fprintf('iter=%d\trel.err=%1.2e\n',j,err);
                end
                
                if err < this.tol 
                    flag = 0;
                    break
                end
            end
            
            z   = Hm(1:j,1:j) \ s(1:j);
            x   = x0 + V(:,1:j) * z;
            H   = H(1:j+1,1:j);
            V   = V(:,1:j+1);

            para = [err,j,flag];
            
            if this.out >= 1
                switch flag
                    case 0
                       fprintf('gmres reached desired tolerance %1.2e at iteration %d. Returned solution has rel. res %1.2e\n',this.tol,j,err);                    
                    case -1
                       fprintf('gmres iterated %d times without reaching desired tolerance %1.2e. Returned solution has rel. res %1.2e\n',j,this.tol,err);
                end
            end
        end
        
        function [c,s,r] = symOrtho(this,a,b)
            %       Computes a Givens rotation
            % Implementation is based on Table 2.9 in
            % Choi, S.-C. T. (2006).
            % Iterative Methods for Singular Linear Equations and Least-squares Problems.
            % Phd thesis, Stanford University.
            %
            
            c = 0; s = 0; r = 0;
            
            if b == 0
                s = 0;
                r = abs(a);
                if a == 0
                    c = 1;
                else
                    c = sign(a);
                end
            elseif a == 0
                c = 0.0;
                s = sign(b);
                r = abs(b);
            elseif abs(b) > abs(a)
                tau = a / b;
                s   = sign(b) / sqrt(1 + tau^2);
                c   = s * tau;
                r   = b / s;
            elseif abs(a) > abs(b)
                tau = b / a;
                c   = sign(a) / sqrt(1 + tau^2);
                s   = c * tau;
                r   = a / c;
            end
            
        end
        
        function[str,frmt] = hisNames(this)
            str = {'relresGMRES','iterGMRES','flagGMRES'};
            frmt = {'%-12.2e','%-12d','%-12d'};
        end
        
        function[para] = hisVals(this,para)
            
        end
        
        function test(this)
            
            M = 5;
            
            A = randn(M);
            xTrue = randn(M,1);
            b = A * xTrue;
            x0 = randn(M,1);
            
            this.m = 5;
            [V,H] = this.solve(A,b,x0);
            
            r0 = b - A * x0;
            beta = norm(r0);
            xOpt = x0 + V(:,1:this.m) * (H(1:this.m,:) \ (beta * eye(this.m,1)));
            
            disp(['Rel. Err. = ', num2str(norm(xTrue - xOpt) / norm(xTrue))]);
            
        end
    end
    
    
end

