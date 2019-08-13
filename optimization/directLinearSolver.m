classdef directLinearSolver
    % classdef directLinearSolver
    %
    % Solve a linear system using backslash
    
    properties
        PC      % preconditioner
    end
    
    methods
        function this = directLinearSolver(varargin)
            if (nargout==0) && (nargin==0)
                this.runMinimalExample;
                return;
            end
            
            this.PC      = [];
            for k=1:2:length(varargin)     % overwrites default parameter
                eval(['this.' varargin{k},'=varargin{',int2str(k+1),'};']);
            end
        end
        
        function [xOpt,para] = solve(this,A,b,x,PC)
            
            para = [];
            
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
            
            
            xOpt = (PC * A) \ (PC * b);
            
        end
        
        function [str,frmt] = hisNames(this)
            % define the labels for each column in his table
            str  = [];
            frmt = [];
        end
        
        function his = hisVals(this,para)
            his = [];
        end
        
        function runMinimalExample(~)
            spcg = feval(mfilename,'tol',1e-15,'maxIter',12);
            A    = randn(10,10);
            xt = randn(10,1);
            rhs = A * xt;
            % [x,flag,relres,iter,resvec] = solve(spcg,A,rhs);
            [x,para] = solve(spcg,A,rhs);
            norm(A*x-rhs)/norm(rhs)
        end
    end
    
end

