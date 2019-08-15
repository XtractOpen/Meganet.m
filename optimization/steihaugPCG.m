classdef steihaugPCG
    % classdef steihaugPCG
    %
    % steihaug preconditioned conjugate gradient scheme for solving linear
    % system with SPD matrix in a trust region.
    
    properties
        tol
        maxIter
        Delta   % trust region radius
        PC      % preconditioner
    end
    
    methods
        function this = steihaugPCG(varargin)
            if (nargout==0) && (nargin==0)
                this.runMinimalExample;
                return;
            end
            
            this.tol     = 1e-1;
            this.maxIter = 10;
            this.Delta   = Inf;
            this.PC      = [];
            for k=1:2:length(varargin)     % overwrites default parameter
                eval(['this.' varargin{k},'=varargin{',int2str(k+1),'};']);
            end
        end
        
        function [xOpt,para] = solve(this,A,b,x,PC)
%             if isa(A,'opBlkdiag') && (isempty(PC) || isa(PC,'opBlkdiag'))
%                 % see if the block sizes are the same and if so call this
%                 % code for each block separately to use the block structure
%                 nbA = numel(A.blocks);
%                 if false && isempty(PC) || numel(PC.blocks) == nbA
%                     cnt = 0;
%                     flag =[]; relres = []; iter = []; resvec = cell(nbA,1);
%                     
%                     if isempty(x); x = 0*b; end
%                     for k=1:nbA
%                         nAk = size(A.blocks{k},1);
%                         xk = x(cnt +(1:nAk));
%                         bk = b(cnt + (1:nAk));
%                         
%                         if isempty(PC)
%                             [xk,fl,rr,it,rv] = solve(this,A.blocks{k},bk,xk);
%                         else
%                             [xk,fl,rr,it,rv] = solve(this,A.blocks{k},bk,xk,PC.blocks{k});
%                         end
%                         x(cnt+(1:nAk)) = xk;
%                         flag = [flag; fl]; 
%                         relres = [relres; rr];
%                         iter = [iter;it];
%                         resvec{k} = rv;
%                         cnt = cnt+nAk;
%                     end
%                     return;
%                 end
%             end
%             
            
            if not(isempty(this.PC))
                % overwrite PC provided by the objective function
                PC = this.PC;
            end
                
            if not(exist('PC','var')) || isempty(PC)
                PC = opEye(size(b,1));
            end
            
            if isnumeric(PC) 
                PC = LinearOperator(size(b,1),size(b,1),@(x) PC\x, @(x) PC\x);
            end
            
            
            if norm(b)==0
                x=0*b;
                para.flag = -2;
                para.relres = 0;
                para.iterOpt = 0; % change
                para.resvec = 0;
                return;
            end
            
            if not(exist('x','var')) || isempty(x)
                x = 0*b;
                r = b;
            else
                r = b-A*x;
            end
            
            z = PC*r;
            p = z;
            
            resvec = zeros(this.maxIter+1,1);
            resvec(1) = gather(norm(b));
            para.flag = 1;
            
            xOpt    = x;
            resOpt  = resvec(1); % optimal residual
            para.iterOpt = 1; % iteration of optimal residual - should we start with []???
            
            for iter=1:this.maxIter
                Ap    = A*p;
                gamma = r'*z;
                curv  = p'*Ap;
                alpha = gather(gamma./curv);
                
                if alpha==Inf || alpha<0
                    % find tau such that x+t*p == Delta and m(x) = b'*x+.5*x'*A*x -> min
                    % solve: x'*x + 2*tau*p'*x + tau^2 p'*p = Delta
                    xx = gather(x'*x);
                    px = 2*gather(p'*x);
                    pp = gather(p'*p);
                    
                    tau1 =  -(px + sqrt(px^2 + 4*this.Delta^2*pp - 4*pp*xx))/(2*pp);
                    tau2 =  -(px - sqrt(px^2 + 4*this.Delta^2*pp - 4*pp*xx))/(2*pp);
                    if norm(r - tau1*Ap) < norm(r-tau2*Ap)
                        tau = tau1;
                    else
                        tau= tau2;
                    end
                    x = x + tau*p;
                    r = r - tau*Ap;
                    resvec(iter+1) = norm(r);
                    para.flag = 2;
                    %para.resvec = resvec;
                    %para.printOuts(1) = iter;
                    break;
                end
                
                xt = x + (gamma./curv)*p;
                if norm(xt) >= this.Delta
                    % find tau such that x+t*p == Delta
                    % solve: x'*x + 2*tau*p'*x + tau^2 p'*p = Delta
                    xx = gather(x'*x);
                    px = 2*gather(p'*x);
                    pp = gather(p'*p);
                    tau =  -(px + sqrt(px^2 + 4*this.Delta^2*pp - 4*pp*xx))/(2*pp);
                    
                    x = x + tau*p;
                    r = r - tau*Ap;
                    resvec(iter+1) = gather(norm(r));
                    
                    % para.resvec = resvec;
                    para.flag = 3;
                    break
                end
                
                x = xt;
                r = r -  (gamma./curv)*Ap;
                
                resvec(iter+1) = gather(norm(r));
                if resvec(iter+1) < resOpt
                    resOpt = resvec(iter+1);
                    para.iterOpt = iter;
                    xOpt = x;
                end
                if resvec(iter+1)/resvec(1) <= this.tol
                    para.flag = 0;
                    break;
                end
                
                z = PC*r;
                beta = z'*r / gamma;
                p    = z + beta*p;
            end
            para.resvec = resvec(1:iter+1);
            para.relres = resOpt/resvec(1);  % relres

        end
        
        function [str,frmt] = hisNames(this)
            % define the labels for each column in his table
            str  = {'iterCG','relresCG'};
            frmt = {'%-12d','%-12.2e'};
        end
        
        function his = hisVals(this,para)
            his = [para.iterOpt, para.relres];
        end
            
        
        function runMinimalExample(~)
            spcg = feval(mfilename,'tol',1e-15,'maxIter',12);
            A    = randn(10,10);
            A  = A'*A;
            xt = randn(10,1);
            rhs = A*xt;
            % [x,flag,relres,iter,resvec] = solve(spcg,A,rhs);
            [x,~] = solve(spcg,A,rhs);
            norm(A*x-rhs)/norm(rhs)
        end
    end
    
end

