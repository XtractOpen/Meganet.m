classdef bfgs < optimizer
    % classdef bfgs < optimizer
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
        maxIter   % maximum number of iterations
        atol      % absolute tolerance for stopping, stop if norm(dJ)<atol
        rtol      % relative tolerance for stopping, stop if norm(dJ)/norm(dJ0)<rtol
        maxStep   % maximum step (similar to trust region methods)
        out       % flag controlling the verbosity, 
                  %       out==0 -> no output
                  %       out==1 -> print status at each iteration
        LS        % line search object. See, e.g., Armijo.m
    end
    
    methods
        
        function this = bfgs(varargin)
            % constructor, no input required to get instance with default
            % parameters
            this.maxIter = 10;
            this.atol    = 1e-3;
            this.rtol    = 1e-3;
            this.maxStep = 1.0;
            this.out     = 0;
            this.LS      = Armijo();
            for k=1:2:length(varargin)     % overwrites default parameter
                eval(['this.' varargin{k},'=varargin{',int2str(k+1),'};']);
            end;
        end
        
        function [str,frmt] = hisNames(this)
            % define the labels for each column in his table
            str  = {'iter', 'Jc','|x-xOld|', '|dJ|/|dJ0|','mu','LS'};
            frmt = {'%-12d','%-12.2e','%-12.2e','%-12.2e','%-12.2e','%-12d'};
        end
        
        function [xc,His] = solve(this,fctn,xc,fval)
            % minimizes fctn starting with xc. (optional) fval is printed
            if not(exist('fval','var')); fval = []; end;
            
            [str,frmt] = hisNames(this);
            
            % parse objective functions
            [fctn,objFctn,objNames,objFrmt,objHis]     = parseObjFctn(this,fctn);
            str = [str,objNames{:}]; frmt = [frmt,objFrmt{:}];
            [fval,obj2Fctn,obj2Names,obj2Frmt,obj2His] = parseObjFctn(this,fval);
            str = [str,obj2Names{:}]; frmt = [frmt,obj2Frmt{:}];
            doVal     = not(isempty(obj2Fctn));
            
            % evaluate training and validation
            [Jc,para,dJ] = fctn(xc); pVal = [];
            if doVal
                if isa(objFctn,'dnnVarProBatchObjFctn') || isa(objFctn,'dnnVarProObjFctn')
                    [Fval,pVal] = fval([xc; para.W(:)]);
                else
                    [Fval,pVal] = fval(xc);
                end
            end
            
            if this.out>0
                fprintf('== bfgs (n=%d,maxIter=%d,maxStep=%1.1e) ===\n',...
                    numel(xc), this.maxIter, this.maxStep);
                fprintf([repmat('%-12s',1,numel(str)) '\n'],str{:});
            end
            
            his = zeros(1,numel(str));
            nrm0 = norm(dJ(:));
            iter = 1;
            xOld = xc;
            I = eye(numel(xc));
            H = I;
            while iter <= this.maxIter
                
                his(iter,1:4)  = [iter,gather(Jc),gather(norm(xOld(:)-xc(:))),gather(norm(dJ(:))/nrm0)];
                if this.out>0
                    fprintf([frmt{1:4}], his(iter,1:4));
                end
                if (norm(dJ(:))/nrm0 < this.rtol) || (norm(dJ(:))< this.atol), break; end
                
                % get search direction
                s = -H*dJ(:);
                
                if max(abs(s(:))) > this.maxStep
                    s = s/max(abs(s(:))) * this.maxStep; 
                end
                % line search
                if iter == 1; mu = 1.0; end
                [xt,mu,lsIter] = lineSearch(this.LS,fctn,xc,mu,s,Jc,dJ);
                if (lsIter > this.LS.maxIter)
                    disp('LSB in bfgs'); %keyboard
                    his = his(1:iter,:);
                    break;
                end
                
                his(iter,5:6) = [mu lsIter];
                if this.out>0
                    fprintf([frmt{5:6}], his(iter,5:6));
                end
                if lsIter == 1
                    mu = mu*1.5;
                end
                xOld       = xc;
                xc         = xt;
                if doVal
                    if isa(objFctn,'dnnVarProBatchObjFctn') || isa(objFctn,'dnnVarProObjFctn')
                        [Fval,pVal] = fval([xc; para.W(:)]);
                    else
                        [Fval,pVal] = fval(xc);
                    end
                end
                [Jc,para,dJnew] = fctn(xc);
                
                % update Hessian
                sc   = mu*s;
                yc   = dJnew - dJ;
                if dot(yc,sc)>0 % ensure that approximate Hessians remain positive definite
                    H     = (I - (sc*yc')/dot(sc,yc)) * H * (I - (yc*sc')/dot(sc,yc)) + (sc*sc')/dot(yc,sc);
                else
                    if this.out>0
                        warning('bfgs detected negative curvature. Resetting Hessian')
                    end
                    H = eye(numel(xc));
                end
                dJ = dJnew;

                if size(his,2)>=7
                    his(iter,7:end) = [gather(objHis(para)), gather(obj2His(pVal))];
                    if this.out>0
                        fprintf([frmt{7:end}],his(iter,7:end));
                    end
                end
                if this.out>0
                    fprintf('\n');
                end
                iter = iter + 1;
            end
            His = struct('str',{str},'frmt',{frmt},'his',his(1:min(iter,this.maxIter),:));
        end
    end
end