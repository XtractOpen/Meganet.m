classdef leastSquaresSolver < optimizer
    % classdef leastSquaresSolver < optimizer
    %
    % solves least squares-problems
    % 
    % To use this code, the objective function must be of type
    % regressionLoss.
    %
    % The property 'out' controls the verbosity of the method. If out==1
    % each iteration will produce some output in the command window. 
    %
    
    properties
        solver    % chooses LS solver, options {'qr','lsqr',...};
        maxIter   % maximum number of iterations for iterative solvers
        tol       % tolerance for iterative solvers
        out       % flag controlling the verbosity, 
                  %       out==0 -> no output
                  %       out==1 -> print status at each iteration
    end
    
    methods
        
        function this = leastSquaresSolver(varargin)
            % constructor, no input required to get instance with default
            % parameters
            this.solver  = 'qr';
            this.maxIter = 10;
            this.tol     = 1e-3;
            this.out     = 0;
            for k=1:2:length(varargin)     % overwrites default parameter
                eval(['this.' varargin{k},'=varargin{',int2str(k+1),'};']);
            end;
        end
        
        function [str,frmt] = hisNames(this)
            % define the labels for each column in his table
            str  = {'iter', 'Jc','|x-xOld|', '|dJ|/|dJ0|','iterCG','relresCG','mu','LS'};
            frmt = {'%-12d','%-12.2e','%-12.2e','%-12.2e','%-12d','%-12.2e','%-12.2e','%-12d'};
        end
        
        function [W,His] = solve(this,fctn,xc,fval)
            % minimizes fctn starting with xc. (optional) fval is printed
            if not(exist('fval','var')); fval = []; end;
            
            switch this.solver
                case 'lsqr'
                    error('to be implemented')
                otherwise
                    % use QR
                    
                    % grab output Y and true values C
                    Y = fctn.Y;
                    C = fctn.C;
                    
                    % if W contains bias
                    if fctn.pLoss.addBias
                       Y = padarray(Y,[1,0],1,'post');
                    end
                    
                    % regularization
                    if not(isempty(fctn.pRegW))
                        if isa(fctn.pRegW,'tikhonovReg')
                            if isa(fctn.pRegW.B,'opEye')
                                 Y = [Y, sqrt(fctn.pRegW.alpha)*speye(size(Y,1))];
                            else
                                error('B must be identity');
                            end
                            
                            if fctn.pRegW.xref==0
                                % C = padarray(C, [1,0], 0, 'post');
                                C = [C,  zeros(size(C,1),size(Y,1),'like',Y)];
                            else
                                C = [C; fctn.pRegW.xref];
                            end
                       else
                           error('regularizer not supported')
                       end
                    end
                    W = C / Y;
                    
                    His = [];
            end
            
            % His = struct('str',{str},'frmt',{frmt},'his',his(1:min(iter,this.maxIter),:));
        end
    end
end