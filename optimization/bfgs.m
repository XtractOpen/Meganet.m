classdef bfgs < optimizer
    % classdef BFGS < optimizer
    %
    % BFGS method for minimizing non-linear objective function. 
    % 
    % To use this code, the objective function must be of type objFctn or a 
    % function handle that can be called like
    %
    % [Jc,para,dJ] = fctn(xc);
    %
    % where 
    %   Jc   - function value gradient 
    %   para - struct containing information about the objective used for
    %          plotting or printing (see, e.g., method hisVals in objFctn)
    %   dJ   - gradient about xc
    %   
    % iter - current iteration
    % Jc   - objective function value
    % |x-xOld| - size of step taken in this iteration
    % |dJ|/|dJ0| - relative norm of gradient
    % mu         - line search parameter
    % LS         - number of line search steps
    %
    % When the objective function is a subtype of 'objFctn' there will be
    % additional output specified by the respective class. For example, the
    % dnnObjFctn will produce the following output
    %
    % F          - loss
    % accuracy   - classification accuracy
    % R(theta)   - value of regularizer for network paramters
    % alpha      - regularization parameter for theta 
    % R(W)       - value of regularizer for classfication weights
    
    properties
        maxIter = 10  % maximum number of iterations
        atol    = 1e-3   % absolute tolerance for stopping, stop if norm(dJ)<atol
        rtol    = 1e-3  % relative tolerance for stopping, stop if norm(dJ)/norm(dJ0)<rtol
        maxStep = 1 % maximum step (similar to trust region methods)
        out     = 0  % flag controlling the verbosity, 
                  %       out==0 -> no output
                  %       out==1 -> print status at each iteration
        LS      = Wolfe()  % line search object. See, e.g., Armijo.m
        name
        maxWorkUnits = Inf
        topValFileName
    end
    
    methods
        
        function this = bfgs(varargin)
            % constructor, no input required to get instance with default
            % parameters
            
            for k = 1:2:length(varargin)
                this.(varargin{k}) = varargin{k+1};
            end
            
            if isa(this.LS,'Armijo')
                warning([mfilename,': Armijo conditions are not sufficient for BFGS'])
            end
            
            if isempty(this.name)
                this.name = class(this);
            end
        end
        
        function [str,frmt] = hisNames(this,fctn,fval)
            % define the labels for each column in his table
            str  = {'iter', 'Jc','|x-xOld|', '|dJ|/|dJ0|'};
            frmt = {'%-12d','%-12.2e','%-12.2e','%-12.2e'};
            
            % add line or trust region search names
            if ~isempty(this.LS)
                [strLS,frmtLS] = hisNames(this.LS);
                str  = [str,strLS];
                frmt = [frmt,frmtLS];
            end
            
            str = [str,'TotalWork'];
            frmt = [frmt, '%-12d'];
            
            if nargin > 1
                % objective function
                [strFctn,frmtFctn] = hisNames(fctn); 
                str = [str,strFctn];
                frmt = [frmt,frmtFctn];
                
                % validation function
                if exist('fval','var') && ~isempty(fval)
                    [strFval,frmtFval] = hisNames(fval); 
                    str = [str,strFval];
                    frmt = [frmt,frmtFval];
                end
            end
            
        end
        
        function [xc,His] = solve(this,fctn,xc,fval)
            % minimizes fctn starting with xc. (optional) fval is printed
            
            % check if validation function exists
            doVal = exist('fval','var');
            if ~doVal, fval = []; end
            
            % get names for printing out
            [str,frmt] = hisNames(this,fctn,fval);
            
            % store outputs
            optBreak = struct('iter',[],'reason',[],'flag',[],'stats',[]);
            His = struct('str',{str},'frmt',{frmt},'his',[],'optBreak',optBreak);
            
            % evaluate training and validation
            [Jc,~,dJ] = eval(fctn,xc);
            workUnits = 2;
            
           
            if this.out > 0
                fprintf('== %s (n=%d, maxIter=%d, maxWorkUnits=%d, maxStep=%1.1e) ===\n',...
                    this.name, numel(xc), this.maxIter, this.maxWorkUnits, this.maxStep);
                fprintf([repmat('%-12s',1,numel(str)) '\n'],str{:});
            end
            
            his = zeros(1,numel(str));
            nrm0 = norm(dJ(:));
            iter = 1;
            xOld = xc;
            I = eye(numel(xc));
            H = I;
            
            topVal = 0;
            while (iter <= this.maxIter && workUnits <= this.maxWorkUnits)
                
                hisIter = [iter,Jc,norm(xOld(:)-xc(:)),norm(dJ(:))/nrm0];
               
                if (norm(dJ(:))/nrm0 < this.rtol) || (norm(dJ(:)) < this.atol)
                    optBreak.iter = iter;
                    optBreak.reason = 'tolerance reached';
                    optBreak.flag   = 1;
                    optBreak.stats  = struct('rtol',norm(dJ(:))/nrm0,'atol',norm(dJ));
                    
                    if this.out > 0
                            fprintf([frmt{1:length(hisIter)},'Stopped by Tolerance Reached at iter. ',num2str(iter),'\n'],hisIter);
                    end
                        
                    break; 
                end
                
                % get search direction
                s = -H * dJ(:);
                
                if max(abs(s(:))) > this.maxStep
                    s = s / max(abs(s(:))) * this.maxStep; 
                end
                
                if isempty(this.LS)
                    xt = xc + s;
                else
                    % line search
                    if iter==1
                        mu=1.0;
                    else
                        mu = min(1,2.02*(Jc-JOld)/dot(dJ,s));
                    end
                    [xt,paraLS] = lineSearch(this.LS,fctn,xc,mu,s,Jc,dJ);
                    workUnits = workUnits + paraLS.funEvals;

                    hisIter = cat(2,hisIter,hisVals(this.LS,paraLS));
                    
                    if (paraLS.mu==0)
                        % disp([mfilename,': LSB at iter ',num2str(iter)]); %keyboard
                       
                        optBreak.iter = iter;
                        optBreak.reason = 'LSB';
                        optBreak.flag = 3;
                        optBreak.stats = paraLS;
                        
                        if this.out > 0
                            fprintf([frmt{1:length(hisIter)},'Stopped by LSB at iter. ',num2str(iter),'\n'],hisIter);
                        end

                        break;
                    end
                    
                    % set mu to new value
                    mu = paraLS.mu;

                    if paraLS.funEvals == 1
                        mu = min(1.5 * mu,1);
                    end
                end
                
                xOld       = xc;
                xc         = xt;
                
                JOld = Jc;
                [Jc,para,dJnew] = eval(fctn,xc);
                workUnits = workUnits + 2;
                hisIter = cat(2,hisIter,[workUnits,hisVals(fctn,para)]);
                
                if doVal
                    if ~isfield(para,'W'), para.W = []; end
                    [~,pVal] = eval(fval,[xc(:); para.W(:)]);
                    hisIter = cat(2,hisIter,hisVals(fval,pVal));
                    
                    if ((iter == 1 || hisIter(end) > topVal) && ~isempty(this.topValFileName))
                        net = fctn.net;
                        topVal = hisIter(end);
                        save(this.topValFileName,'iter','net','xc','para','hisIter')
                    end
                end
                
                 % print statistics
                if this.out > 0
                    fprintf([frmt{:},'\n'],hisIter);
                end
                
                % update Hessian
                sc   = mu * s;
                yc   = dJnew - dJ;
                if dot(yc,sc)>0 % ensure that approximate Hessians remain positive definite
                    H = (I - (sc*yc')/dot(sc,yc)) * H * (I - (yc*sc')/dot(sc,yc)) + (sc*sc')/dot(yc,sc);
                else
                    if this.out > 0
                        warning([mfilename,' detected negative curvature. Resetting Hessian'])
                    end
                    H = eye(numel(xc));
                end
                dJ = dJnew;

                % store convergence statistics
                his = cat(1,his,hisIter);
                
                % move to next iteration
                iter = iter + 1;
            end
            
            % store output statistics
            if isempty(optBreak.reason)
                optBreak.iter = iter;
                optBreak.reason = 'maxIter';
                optBreak.flag = 4;
            end
            
            
            His.his = his;
            His.optBreak = optBreak;
        end
    end
end