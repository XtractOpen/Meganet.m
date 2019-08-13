classdef directLinearSolver
    % classdef directLinearSolver
    %
    % Solve A * x = b where A is SPD.
    % To be used as a linear solver within a Newton iteration.
    
    properties
        
    end
    
    methods
        function this = directLinearSolver(varargin)
            if (nargout==0) && (nargin==0)
                this.runMinimalExample;
                return;
            end
            
            for k=1:2:length(varargin)     % overwrites default parameter
                eval(['this.' varargin{k},'=varargin{',int2str(k+1),'};']);
            end
        end
        
        function [xOpt,para] = solve(this,A,b,~,~)
            
            % no additional outputs
            para = [];

            % solve
            % xOpt = A \ b;
            
            R = chol(A);
            xOpt = R \ (R' \ b);
            
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
            spcg = feval(mfilename);
            A    = randn(10,10);
            xt = randn(10,1);
            rhs = A * xt;
            % [x,flag,relres,iter,resvec] = solve(spcg,A,rhs);
            [x,para] = solve(spcg,A,rhs);
            norm(A*x-rhs)/norm(rhs)
        end
    end
    
end

