classdef nlcg < optimizer
    % classdef nlcg < optimizer
    %
    % nonlinear conjugate gradient scheme for minimizing nonlinear
    % objective
    
    properties
        maxIter
        atol
        rtol
        maxStep
        out
        LS
    end
    
    methods
        
        function this = nlcg(varargin)
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
            str  = {'iter', 'Jc','|x-xOld|', '|dJ|/|dJ0|','mu','LS'};
            frmt = {'%-12d','%-12.2e','%-12.2e','%-12.2e','%-12.2e','%-12d'};
        end
        
        function [xc,His] = solve(this,fctn,xc,fval)
            if not(exist('fval','var')); fval = []; end;
            
            [str,frmt] = hisNames(this);
            
            % parse objective functions
            [fctn,objFctn,objNames,objFrmt,objHis]     = parseObjFctn(this,fctn);
            str = [str,objNames{:}]; frmt = [frmt,objFrmt{:}];
            [fval,obj2Fctn,obj2Names,obj2Frmt,obj2His] = parseObjFctn(this,fval);
            str = [str,obj2Names{:}]; frmt = [frmt,obj2Frmt{:}];
            doVal     = not(isempty(obj2Fctn));
            
            % evaluate training and validation
            xc = this.LS.P(xc);
            [Jc,para,dJ] = fctn(xc); pVal = [];
            if doVal
                if isa(objFctn,'dnnVarProBatchObjFctn') || isa(objFctn,'dnnVarProObjFctn')
                    [Fval,pVal] = fval([xc; para.W(:)]);
                else
                    [Fval,pVal] = fval(xc);
                end
                
            end
            
            if this.out>0
                fprintf('== nlcg (n=%d,maxIter=%d,maxStep=%1.1e) ===\n',...
                    numel(xc), this.maxIter, this.maxStep);
                fprintf([repmat('%-12s',1,numel(str)) '\n'],str{:});
            end
            
            his = zeros(1,numel(str));
            nrm0 = norm(dJ(:));
            iter = 1;
            xOld = xc;
            while iter <= this.maxIter
                
                his(iter,1:4)  = [iter,gather(Jc),gather(norm(xOld(:)-xc(:))),gather(norm(dJ(:))/nrm0)];
                if this.out>0
                    fprintf([frmt{1:4}], his(iter,1:4));
                end
                if (norm(dJ(:))/nrm0 < this.rtol) || (norm(dJ(:))< this.atol), break; end
                
                % get search direction
                if norm(dJ)>1
                    dJ = dJ/norm(dJ(:));
                end
                
                if iter==1
                    s = -dJ;
                else
                    y    = dJ - dJOld;
                    beta = dot( y - 2*s*norm(y)^2/dot(s,y) , dJ/dot(s,y));
                    s    = -dJ + beta*s;
                end
                
                if max(abs(s(:))) > this.maxStep
                    s = s/max(abs(s(:))) * this.maxStep;
                end
                
                % line search
                if iter == 1; mu = 1.0; end
                [xt,mu,lsIter] = lineSearch(this.LS,fctn,xc,mu,s,Jc,dJ);
                if (lsIter > this.LS.maxIter)
                    disp('LSB in nlcg'); %keyboard
                    his = his(1:iter,:);
                    break;
                end
                his(iter,5:6) = [mu lsIter];
                if this.out>0
                    fprintf([frmt{5:6}], his(iter,5:6));
                end
                if lsIter == 1
                    mu = min(mu*1.5,1);
                end
                xOld       = xc; dJOld = dJ;
                xc         = xt;
                [Jc,para,dJ] = fctn(xc);
                if doVal
                    if isa(objFctn,'dnnVarProBatchObjFctn') || isa(objFctn,'dnnVarProObjFctn')
                        [Fval,pVal] = fval([xc; para.W(:)]);
                    else
                        [Fval,pVal] = fval(xc);
                    end
                    
                end
                
                if size(his,2)>=6
                    his(iter,7:end) = [gather(objHis(para)), gather(obj2His(pVal))];
                    if this.out>0
                        fprintf([frmt{7:end}],his(iter,7:end));
                    end
                end
                fprintf('\n');
                iter = iter + 1;
            end
            His = struct('str',{str},'frmt',{frmt},'his',his(1:min(iter,this.maxIter),:));
        end
    end
end