classdef trnewton < optimizer
    
    % trust-region newton with Arnoldi
    
    properties
       maxIter  = 10
       atol     = 1e-3         % absolute tolerance ||dJ|| < atol
       rtol     = 1e-3         % relative tolerance ||dJ||/||dJ0|| < rtol
       dtol     = 1e-5         % stop when trust region becomes too small
       Delta    = []
       out      = 0             % verbose
       name                     % name of solver
       rho0     = 1e-4;
       rhoL     = 0.1; 
       rhoH     = 0.75;
       wdown    = 0.5; 
       wup      = 1.5;
       C        = 1e4;
       
       maxWorkUnits = Inf
       linSol = GMRES()
    end
    
    
    methods
        
        function[this] = trnewton(varargin)
            
            for k = 1:2:length(varargin)
                this.(varargin{k}) = varargin{k+1};
            end
            
            if isempty(this.name)
                this.name = class(this);
            end
        end
        
        
        function[str,frmt] = hisNames(this,fctn,fval)
            str  = {'iter', 'Jc','|x-xOld|', '|dJ|/|dJ0|','|dJ|'};
            frmt = {'%-12d','%-12.2e','%-12.2e','%-12.2e','%-12.2e'};
                        
            % linear solver
            [s1,f1] = hisNames(this.linSol);
            str  = [str,s1];
            frmt = [frmt,f1];
            
            % add line or trust region search names
            str  = [str,{'|s|','Delta','trFlag','ared/pred'}];
            frmt = [frmt,{'%-12.2e', '%-12.2e','%-12d','%-12.2e'}];

            % total work
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
        
        function[xc,His,HBar,Vm1] = solve(this,fctn,xc,fval)
            % fctn - objective function
            % xc   - starting guess
            % fval - (optional) validation objective function
            
            % find out where xc lives
            useGPU = isa(xc,'gpuArray');
            if isa(gather(xc),'single')
                precision = 'single';
            else
                precision = 'double';
            end
            
            
            startTime = tic;
            
            % check if validation function exists
            doVal = exist('fval','var');
            if ~doVal, fval = []; end
            
            % get names for printing out
            [str,frmt] = hisNames(this,fctn,fval);  % optimizer

            % store outputs
            optBreak = struct('iter',[],'reason',[],'flag',[],'stats',[]);
            His = struct('str',{str},'frmt',{frmt},'his',[],'optBreak',optBreak);
            
            
            hisIter = zeros(1,length(str));
            his     = [];  
            iter = 0;
            % evaluate function at current x
            [Jc,para,dJ,d2J,PC] = eval(fctn,xc);
            % workUnits = 2;  % forward and backward
            % validation function
            
            tmp = hisVals(fctn,para);
            if doVal
                % if not varpro
                if ~isfield(para,'W'), para.W = []; end
                [~,pVal] = eval(fval,[xc(:); para.W(:)]);  
                tmp = [tmp,hisVals(fval,pVal)];
            end       
            hisIter(1:5) = [iter,Jc,0,1,norm(dJ(:))];
            hisIter(end-length(tmp)+1:end) = tmp;
            his = cat(1,his,hisIter);
            
            % print optimizer
            if this.out > 0
                % fprintf('== %s (n=%d, maxIter=%d) ===\n',mfilename,numel(xc),this.maxIter);
                fprintf('== %s (n=%d, maxIter=%d, maxWorkUnits=%d) ===\n',this.name,numel(xc),this.maxIter,this.maxWorkUnits);
                fprintf([repmat('%-12s',1,numel(str)) '\n'],str{:});
                fprintf([frmt{:},'\n'],hisIter);
            end
            
            
            
            
            % initialize
             % store convergence values
            nrm0    = norm(dJ(:));      % norm of initial gradient
            iter    = 1;                % iteration count
            workUnits = 2;
            xOld    = xc;               % old x
            % main iteration
            while (iter <= this.maxIter && workUnits <= this.maxWorkUnits)
                
                % hisIter = [iter,Jc,norm(xOld(:)-xc(:)),norm(dJ(:))/nrm0,norm(dJ(:))];
                hisIter = zeros(1,5);
                
%                 % for circle hyperplane example
%                 if (~mod(iter,15) || iter == 1) && ~isa(fctn,'classObjFctn') % start with first iteration
%                     if numel(xc) > nTheta(fctn.net)
%                         W = xc(nTheta(fctn.net)+1:end);
%                     else
%                         W = para.W;
%                     end
% %                     circleHyperplanes(fctn.net,xc(1:nTheta(fctn.net)),W,...
% %                         'Y',fctn.Y,'C',fctn.C,...
% %                         'doSave',1,'fileName',['circle_iter',num2str(iter)]);
%                     %pause;
%                 end
%                     
                
        
                % check stopping criteria
                if (norm(dJ(:))/nrm0 < this.rtol || norm(dJ) < this.atol)
                    optBreak.iter = iter;
                    optBreak.reason = 'tolerance reached';
                    optBreak.flag   = 0;
                    optBreak.stats  = struct('rtol',norm(dJ(:))/nrm0,'atol',norm(dJ));
                    optBreak.hisIter = hisIter;
                    % print statistics
                    if this.out > 0
                        fprintf([frmt{1:length(hisIter)},'Stopped by Tolerance Reached at iter. ',num2str(iter),'\n'],hisIter);
                    end
                    break;
                end
                
                % lambda = 0 case
                if  (~exist('trFlag','var') || trFlag > -2)
%                     only recompute Krylov space if updated xc
                    [s,paraLinSol,z,Vm1d,HBard,funEvals] = solve(this.linSol,d2J,-dJ,[]);
                    workUnits = workUnits + funEvals;
                    hisIter = cat(2,hisIter,paraLinSol);
                else
                    % did not use gmres, so no gmres iters
                    hisIter = cat(2,hisIter,[paraLinSol(1),0,paraLinSol(end)]);
                end
                
                if isempty(this.Delta) && (iter==1)
                    this.Delta=norm(z);
                end
                
                if norm(z) > this.Delta
                    % solve regularized LS: min_s  norm(HBar * s - Vm1' * dJ)^2 +  lambda^2 * norm(s)^2
                    [UH,SH,VH] = svd(HBard,0);
                    SH = diag(SH);
                    rhs1 = Vm1d' * double(gather(dJ));
                    rhs = -UH'*rhs1;
                    solveRLS= @(lambda) VH*((SH.*(rhs))./((SH.^2+lambda.^2)));
                    
                    ff= @(lambda) norm((SH.*(rhs))./((SH.^2+lambda.^2)))-this.Delta;
                    lambdaUp = norm(rhs1)/this.Delta;
                    if ff(0)*ff(lambdaUp)>0
                        keyboard
                    end
                    
                    lambdaOpt = fzero(ff, [0 lambdaUp]);                    
                    z =  solveRLS(lambdaOpt);
                    s = Vm1d(:,1:end-1) * z;
                end

                if norm(z)==0
                    keyboard
                end
                
                [s,Vm1,HBar] = gpuVar(useGPU,precision,s,Vm1d,HBard);
                    
                

                % trust region step
                xOld = xc;
                [xc, trFlag, this.Delta, ared, pred, funEvals] = this.trtest(fctn,Jc,dJ,d2J,HBar,Vm1,xc,s,z);
                workUnits = workUnits + funEvals;
                hisIter = cat(2,hisIter,norm(s),this.Delta,trFlag,ared/pred);
                
                % re-evaluate objective functions
                if trFlag > -2
                    % skip update if weights didn't change!
                    [Jc,para,dJ,d2J,PC] = eval(fctn,xc);
                    workUnits = workUnits + 2;
                    hisIter = cat(2,hisIter,[workUnits,hisVals(fctn,para)]);
                else
                    hisIter = cat(2,hisIter,[workUnits,hisVals(fctn,para)]);
                end
                
                hisIter(1:5) = [iter,Jc,norm(xOld(:)-xc(:)),norm(dJ(:))/nrm0,norm(dJ(:))];
                 
                
                % validation function
                if doVal
                    % if not varpro
                    if ~isfield(para,'W'), para.W = []; end
                    [~,pVal] = eval(fval,[xc(:); para.W(:)]);
                    hisIter = cat(2,hisIter,hisVals(fval,pVal));                                    
                end                
                % print statistics
                if this.out > 0
                    fprintf([frmt{:},'\n'],hisIter);
                end
                
                % store convergence statistics
                his = cat(1,his,hisIter);
                
                % move to next iteration
                iter = iter + 1;
            end
            
            % store output statistics
            if isempty(optBreak.reason)
                optBreak.iter = iter;
                optBreak.reason = 'maxIter';
                optBreak.flag = -1;
                optBreak.hisIter = hisIter;
            end
            
            if isempty(his)
                his = hisIter;
            end
            
            His.his = his;
            His.optBreak = optBreak;
            
            endTime = toc(startTime);
            His.endTime = endTime;
        end
        
        function [xc, trFlag, Delta, ared, pred, funEvals] = trtest(this, fctn,Jc,dJ,d2J,HBar,Vm1,xc,s,z)
            %  see p. 51 of Tim Kelley's book  (https://archive.siam.org/books/textbooks/fr18_book.pdf)
            xt = xc + s;
            ared = Jc - fctn.eval(xt);
            funEvals = 1; 
            
%             pred = -dJ'*s - 0.5*s'*(d2J*s); 
            pred = z(1)*norm(dJ) - 0.5*z'*HBar(1:end-1,:)*z;
            if ared/pred < this.rho0
                % reject and decrease Delta
                Delta = min(this.wdown*this.Delta, this.wdown*double(gather(norm(s))));
                trFlag = -2;
                
                % what if trust region was just expanded?  set xc = xtOld
                
            elseif (this.rho0 <= ared/pred) && (ared/pred < this.rhoL)
                % accept and decrease Delta
                Delta = this.wdown * this.Delta;
                xc = xt;
                trFlag = 1;
            elseif (this.rhoL <= ared/pred)  && (ared/pred <= this.rhoH)
                % accept and keep Delta 
                xc = xt;
                Delta = this.Delta;
                trFlag = 0;
            elseif (abs(norm(s)-this.Delta)/this.Delta < 1e-5)  && (this.Delta <= this.C*norm(dJ))
                Delta = this.wup*this.Delta;
%                 xc = xt;                
                trFlag=-3;
            else
                Delta = this.Delta;
                xc = xt;                
                trFlag = 2;
            end                            
        end        
    end
end




