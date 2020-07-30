classdef lBFGS < optimizer
    
    properties
       maxIter  = 10
       atol     = 1e-3
       rtol     = 1e-3
       maxStep  = 1
       nBFGS    = 10 % number of vectors to keep
       out      = 0
       LS       = Wolfe()
       maxWorkUnits = Inf
    end
    

    methods
        
        function this = lBFGS(varargin)
            
            for k = 1:2:length(varargin)
                this.(varargin{k}) = varargin{k+1};
            end
            
        end
        
        function[str,frmt] = hisNames(this,fctn,fval)
            
            % define the labels for each column in output
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
            
            if nargin > 1 && ismethod(fctn,'hisNames')
                % objective function
                [strFctn,frmtFctn] = hisNames(fctn); 
                str = [str,strFctn];
                frmt = [frmt,frmtFctn];
                
                % validation function
                if exist('fval','var') && ~isempty(fval) && ismethod(fval,'hisNames')
                    [strFval,frmtFval] = hisNames(fval); 
                    str = [str,strFval];
                    frmt = [frmt,frmtFval];
                end
            end
            
        end
        
        function[xc,His,xAll] = solve(this,fctn,xc,fval)
            
            % check if validation function exists
            doVal = exist('fval','var');
            if ~doVal, fval = []; end
            
            % get names for printing out
            [str,frmt] = hisNames(this,fctn,fval);
            
            % store outputs
            optBreak = struct('iter',[],'reason',[],'flag',[],'stats',[]);
            His = struct('str',{str},'frmt',{frmt},'his',[],'optBreak',optBreak);
            
            % evaluate training and validation
%             [Jc,~,dJ] = eval(fctn,xc);
%           workUnits = 2;
            
            hisIter = zeros(1,length(str));
            his     = [];  
            iter = 0;
            % evaluate function at current x
            [Jc,para,dJ] = eval(fctn,xc);
            % workUnits = 2;  % forward and backward
            % validation function
            
            tmp = hisVals(fctn,para);
            if doVal
                % if not varpro
                if ~isfield(para,'W'), para.W = []; end
                [~,pVal] = eval(fval,[xc(:); para.W(:)]);  
                tmp = [tmp,hisVals(fval,pVal)];
            end       
            % hisIter(1:5) = [iter,Jc,0,1,norm(dJ(:))];
            hisIter(1:5) = [iter,Jc,0,1,0];
            hisIter(end-length(tmp)+1:end) = tmp;
            his = cat(1,his,hisIter);
            
            if this.out > 0
                fprintf('== %s (n=%d,maxIter=%d,maxWorkUnits=%d,maxStep=%1.1e,nFGS=%d,rtol=%1.2e, atol=%1.2e) ===\n',...
                    mfilename, numel(xc), this.maxIter, this.maxWorkUnits, this.maxStep, this.nBFGS, this.rtol, this.atol);
                fprintf([repmat('%-12s',1,numel(str)) '\n'],str{:});
                fprintf([frmt{1:length(hisIter)},'\n'],hisIter);
            end
            
            % Hessian information
            xAll = [];
            theta = 1;
            H0 = theta * speye(numel(xc));
            Y = [];
            S = [];
            
            xOld = xc;
            nrm0 = norm(dJ);
            
            iter = 1;
            workUnits = 2;
            while (iter <= this.maxIter && workUnits <= this.maxWorkUnits)

                hisIter = [iter,Jc,norm(xc(:)-xOld(:)),norm(dJ)/nrm0];
                
                if nargout > 2, xAll = cat(2,xAll,xc); end

                s = this.bfgsrec(Y,S,H0,-dJ,size(Y,2));
                
                if true || iter == 1
                    mu = 1.0;
                else
                    mu = min(1, 2.02 * (Jc - JOld) / dot(dJ, s));
                end

                % test if s is a descent direction
                if s(:)' * dJ(:) > 0
                    warning('s is not a descent direction, try -s');
                    s = -dJ;
                end    
                JOld = Jc;
                xOld = xc;
                dJOld = dJ;
                
                
                % line search
                [xc,paraLS] = lineSearch(this.LS,fctn,xc,mu,s,Jc,dJ);
                workUnits = workUnits + paraLS.funEvals;
                
                if (paraLS.funEvals > this.LS.maxFunEvals)
                    % disp([mfilename,': LSB at iter ',num2str(iter)]); %keyboard

                    optBreak.iter = iter;
                    optBreak.reason = 'LSB';
                    optBreak.flag = 3;
                    optBreak.stats = paraLS;

                    if this.out > 0
                        hisIter = cat(2,hisIter,hisVals(this.LS,paraLS));
                        fprintf([frmt{1:length(hisIter)},'Stopped by LSB at iter. ',num2str(iter),'\n'],hisIter);
                    end
                    
                    break;
                end
                
                % set mu to new value
                mu = paraLS.mu;

                if paraLS.funEvals == 1
                    mu = min(1.5 * mu,1);
                end

                hisIter = cat(2,hisIter,hisVals(this.LS,paraLS));
                
                [Jc,para,dJ] = eval(fctn,xc);
                workUnits = workUnits + 2;
                hisIter = cat(2,hisIter,[workUnits,hisVals(fctn,para)]);
                
                if doVal
                    if ~isfield(para,'W'), para.W = []; end
                    [~,pVal] = eval(fval,[xc(:); para.W(:)]);
                    hisIter = cat(2,hisIter,hisVals(fval,pVal));
                end
                
                 % print statistics
                if this.out > 0
                    fprintf([frmt{1:length(hisIter)},'\n'],hisIter);
                end
                

                if (norm(dJ) < this.atol) || (norm(dJ) < this.rtol * nrm0)
                    
                    
                    optBreak.iter = iter;
                    optBreak.reason = 'tolerance reached';
                    optBreak.flag   = 1;
                    optBreak.stats  = struct('rtol',norm(dJ(:))/nrm0,'atol',norm(dJ));
                    
                    if this.out > 0
                            fprintf([frmt{1:length(hisIter)},'Stopped by Tolerance Reached at iter. ',num2str(iter),'\n'],hisIter);
                    end

                    His.optBreak = optBreak;
                    His.his = his;
                    
                    return;
                    
                end
                
                % update Hessian approximation
                yk = dJ - dJOld;
                sk = xc - xOld;
                if dot(yk,sk)>0
                    i0 = 2 - (size(Y,2) < this.nBFGS);
                    Y = [Y(:,i0:end),yk];
                    S = [S(:,i0:end),sk];
                else
                    warning('negative curvature -> skip update');
                end
               
                his = cat(1,his,hisIter);
                
                if workUnits > this.maxWorkUnits
                    break;
                end
                
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
       
        
        
        function[] = test(this,showPlot,testGrad)

            if ~exist('testGrad','var')
                testGrad = 0;
            end
                
            x = randn(2,10);
            lBFGS.Rosenbrock(x,testGrad);
            
            E = dnnRosenbrockObjFctn();
            W = [8;4];
            this.maxIter = 200;
            this.rtol = 1e-15;
            this.atol = 1e-15;
            this.out = 1;
            [xc,his] = solve(this,E,W);
            fprintf('numerical solution: x = [%1.4f, %1.4f]\n',xc);
            
            if exist('showPlot','var') && showPlot
                figure(1); clf;
                subplot(1,2,1)
                    x1 = linspace(-2,2,128);
                    x2 = linspace(-2,2,128);
                    [X1,X2] = meshgrid(x1,x2);
                    F = reshape(lBFGS.Rosenbrock([X1(:) X2(:)]'),128,128);
                    [~,idmin] = min(F(:));
                    xmin = [X1(idmin); X2(idmin)];
                    % contour(X1,X2,F,100,'LineWidth',3)
                    surf(X1,X2,F);
                    hold on;
                    plot(xmin(1),xmin(2),'sb','MarkerSize',50);
                    plot(xc(1),xc(2),'.r','MarkerSize',30);
                    hold off;
                subplot(1,2,2)
                semilogy(his.his(:,2),'LineWidth',3);
                set(gca,'FontSize',20);
                title('optimality condition');
            end

        end
        
        
    end
    
    
    methods (Static)
        
        function s = bfgsrec(Y,S,H0,s,n)
            % recursion to solve s = H0\s
            if n == 0
%                 s = H0 \ s;
                s=s;
            else
                alpha = (S(:,n)' * s) / (Y(:,n)' * S(:,n));
                s = s - alpha * Y(:,n);
                s = lBFGS.bfgsrec(Y,S,H0,s,n-1);
                s = s + (alpha - (Y(:,n)' * s)/(Y(:,n)' * S(:,n))) * S(:,n);
            end
        end
        
        function s = mulBFGS(Y,S,H0,theta,rhs,shift)
            
            if exist('shift','var') && not(isempty(shift)) && isnumeric(shift)
                H0 = H0 + diag(sparse(shift));
            end
            
            Wk = [Y, theta * S];
            Dk = diag(sum(Y .* S,1));
            Lk = tril(S' * Y,-1);
            Mkinv = [-Dk, Lk'; Lk, theta * (S' * S)];
            Vk = - (Mkinv \ Wk');
            % if not(isempty(Wk))
            %     Bk = H0 + Wk*Vk;  
            % else
            %     Bk = H0;
            % end
            % s = Bk*rhs;
            if not(isempty(Wk))
                s = H0 * rhs + Wk * (Vk * rhs);  
            else
                s = H0 * rhs;
            end
        end
        
        function s = invBFGS(Y,S,H0,theta,rhs,shift)
            
            if exist('shift','var') && not(isempty(shift)) && isnumeric(shift)
                H0 = H0 + diag(shift);
            end
            
            Wk = [Y, theta * S];
            Dk = diag(sum(Y .* S,1));
            Lk = tril(S' * Y,-1);
            Mkinv = [-Dk, Lk'; Lk, theta * (S' * S)];
            Vk = - (Mkinv \ Wk');

            % use Woodbury identity
            Ik = eye(size(Wk,2));
            s = H0 \ rhs - H0 \ (Wk * ((Ik + Vk * (H0 \ Wk)) \ (Vk * (H0 \ rhs))));
        end
        
        function[f,df] = Rosenbrock(x,testGrad)
            f = 100 * (x(2,:) - x(1,:).^2).^2 + (1 - x(1,:)).^2;
            df = [-400 * (x(2,:) - x(1,:).^2) .* x(1,:) - 2 * (1 - x(1,:));  200 * (x(2,:) - x(1,:).^2)];

            
            if exist('testGrad','var') && testGrad
                % test derivative
                xc = x(:,1);
                fc = f(1);
                dfc = df(:,1);

                dx = randn(size(xc));
                dx = dx / norm(dx(:));

                dfdx = dfc(:)' * dx(:);

                N = 8;
                for i = 1:N
                    h = 10^(-i);

                    xt = xc + h * dx;
                    ft = 100 * (xt(2) - xt(1).^2).^2 + (1 - xt(1)).^2;
                    err0 = norm(fc - ft);
                    err1 = norm(fc + h * dfdx - ft);

                    fprintf('%-12.2e %-12.2e %-12.2e\n',h,err0,err1);
                end
            end
        end
        
    end
   

end