classdef tikParamFctn
    
    properties
        regSelectMethod = 'sGCV'            % parameter selection method {'sGCV','sDP','sUPRE'}
        optMethod       = 'trialPoints'     % selection technique {'trialPoints','constant','fminbnd','fmincon','newton'}
        noiseVariance   = 1e-2              % noise level in data
        tau2            = 2                 % 
        safeguard       = true              % 
        lowerBound      = 1e-7              % smallest possible Lambda (in magnitude)
        upperBound      = 1e3               % largest possible Lambda (in magnitude)
        allowNeg        = 1                 % allow Lambda to be negative
        
        % for newton
        maxIter         = 1                 % maximum iterations
        LS              = Armijo()          % linesearch
        
    end
    
    methods
        
        function[this] = tikParamFctn(varargin)
            
            % overwrite default parameter
            for k = 1:2:length(varargin)     
                this.(varargin{k}) = varargin{k + 1};
            end
            
            % select fixed Lambda
            if ~any(strcmp(this.regSelectMethod,{'sGCV','sDP','sUPRE','trialPoints'}))
                this.optMethod = 'none';
            end
        end
        
        function[Lambda,W] = solve(this,Z,C,M,W,sumLambda,Lambda)
            % ----------------------------------------------------------- %
            % initialize
            lb = this.lowerBound;
            ub = this.upperBound;
            
            % ----------------------------------------------------------- %
            % compute SVD
            [U,S,~] = svd(M,'econ');
            
            % precompute matrices
            ZU    = Z' * U; 
            WZC   = W * Z - C;
            WZCZU = WZC * ZU; 
            WU    = W * U;
            
            % constants
            n_target = size(C,1);
            n_calTk  = size(Z,2) * n_target;
            
            % ----------------------------------------------------------- %
            % objective function
            switch this.regSelectMethod
                case 'sDP'
                    objFcn = @(lambda) this.sDP(lambda,ZU,WZC,WZCZU,WU,S,sumLambda,this.tau2,n_calTk,this.noiseVariance);
                case 'sUPRE'
                    objFcn = @(lambda) this.sUPRE(lambda,ZU,WZC,WZCZU,WU,S,sumLambda,this.noiseVariance,n_target);
                case 'sGCV'
                    objFcn = @(lambda) this.sGCV(lambda,ZU,WZC,WZCZU,WU,S,sumLambda,n_calTk,n_target);
            end
            
            % ----------------------------------------------------------- %
            % optimization
            switch this.optMethod
                % - - - - - - - - - - - - - - - - - - - - - - - - - - - - %    
                case 'fminbnd'
                    Lambda  = fminbnd(objFcn, lb, ub);

                % - - - - - - - - - - - - - - - - - - - - - - - - - - - - % 
                case 'fmincon'

                    Lambda0 = sqrt(lb * ub);  
                    options = optimoptions('fmincon','Display', 'off', ...
                        'Algorithm','trust-region-reflective', ...
                        'SpecifyObjectiveGradient',true,...
                        'HessianFcn','objective');
                    Lambda = fmincon(objFcn,Lambda0,[],[],[],[],lb,ub,[],options);

                % - - - - - - - - - - - - - - - - - - - - - - - - - - - - %    
                case 'newton'
                    
                    Lambda0 = sqrt(ub * lb);
                    [f0, g, H] = objFcn(Lambda0);

                    Lambda = Lambda0;
                    f  = f0;
                    mu = 1; % for line search
                    iter = 0;
                    while iter <= this.maxIter

                        % search directioin
                        s = -g / H;

                        % check descent direction
                        if g' * s > 0, s = -s; end
                        
                        % apply linesearch
                        if ~isempty(this.LS)                 
                            [Lambda,mu,lsIter] = lineSearch(this.LS,objFcn,Lambda,mu,s,f,g);
                            % check for break
                        else
                            Lambda = Lambda + s;
                        end
                        
                        % do not compute gradient/hessian for last
                        % iteration
                        if iter == this.maxIter
                           f = objFcn(Lambda);
                        else
                           [f, g, H] = objFcn(Lambda);
                        end
                        
                        iter = iter + 1;
                       
                    end
                    
                    % check for feasibility
                    if Lambda < lb || Lambda > ub
                        lambdaBound = [lb,ub];
                        fTrial = objFcn(lambdaBound);
                        [~,idx] = min(fTrial);
                        Lambda = lambdaBound(idx);
                    end

                % - - - - - - - - - - - - - - - - - - - - - - - - - - - - %    
                case 'trialPoints'
                    
                    if this.allowNeg
                        % bounded by sumLambda and crosses close to 0
                        p = min(log10(sumLambda / 2),ub);
                        lambda = logspace(min(p,-16),max(p,-16),30);
                        lambda = [-fliplr(lambda),lambda];
                    else
                        lambda  = logspace(log10(lb),log10(ub),30);
                    end
                    % lambda = linspace(-sumLambda + lb,ub,100);
%                     lambda1 = logspace(log10(eps),log10(ub), 30);
%                     lambda2 = logspace(log10(eps),log10(sumLambda), 30);
%                     lambda = [-lambda2,lambda1];
                    % lambda = [-fliplr(lambda),lambda];
%                     
                    f       = objFcn(lambda);
                    [~,idx] = min(f);
                    Lambda  = lambda(idx);
                    
                    if Lambda + sumLambda < lb
                        Lambda = -sumLambda + lb;
                    end
                        
                    
                    % lambda = logspace(log10(lb),log10(ub),30);
%                     if this.allowNeg
%                         
%                         nLow = sumLambda - lb;
%                         
%                         if nLow <= 0
%                             % sumLambda already less than or equal to lower
%                             % bound
%                             lambda2 = [];
%                         else
%                             lambda2 = -logspace(log10(nLow),log10(eps),30);
%                         end
%                         
%                         lambda = [lambda,lambda2];
% 
% %                         p      = min(log10(sumLambda / 2),ub);
% %                         lambda = logspace(min(p,-16),max(p,-16),30);
% %                         lambda = [-fliplr(lambda),lambda];
%                     end
% 
%                     f       = objFcn(lambda);
%                     [~,idx] = min(f);
%                     Lambda  = lambda(idx);

                otherwise
                    % constant choice of lambda
                    % Lambda = sumLambda;
            end

            % ----------------------------------------------------------- %
            % solve for W using the Sherman-Morrison-Woodbury formula
            % 0.5 * ||S * [M, Z, sqrt(sumLambda) * I] - [0, WZC, Lambda / sqrt(sumLambda) * W||_F^2
            alpha = 1 ./ (Lambda + sumLambda);
            s     = diag(S);
            Binv  = alpha * speye(size(W,2)) - alpha^2 * (U .* (s.^2 ./ (1 + alpha * s.^2))') * U';
            W     =  W - (WZC * Z' + Lambda * W) * Binv;
            
            % check
            % S = [zeros(size(W,1),size(M,2)-size(Z,2)), WZC, Lambda /sqrt(Lambda + sumLambda) * W ]/ [M, sqrt(Lambda + sumLambda) * eye(size(M,1))];
            % W = W - (WZC * Z' + Lambda * W) / (M * M' + (Lambda + sumLambda) * eye(size(M,1)));
            
        end
        
                
        function[] = plotObjFcn(this,objFcn,Lambda,Lambda0,lb,ub,showNeg)
            if ~exist('lb','var') || isempty(lb), lb = this.lowerBound; end
            if ~exist('ub','var') || isempty(ub), ub = this.upperBound; end
            if ~exist('showNeg','var'), showNeg = 0; end
                
            % two plots
            x = logspace(log10(abs(lb)),log10(abs(ub)),10000);
            fPos = objFcn(x);
            fNeg = objFcn(-x);

            loglog(x,fPos,'LineWidth',2,'DisplayName','fPos');
            hold on;
            if showNeg
                loglog(x,fNeg,'LineWidth',2,'DisplayName','fNeg');
            end
            
            markSize = 20;
            label = sprintf('lb =%0.2e, f(lb)=%0.2e',lb,objFcn(lb));
            plot(lb,objFcn(lb),'.','MarkerSize',markSize,'DisplayName',label);
            xline(lb,'--','HandleVisibility','off')

            label = sprintf('ub =%0.2e, f(lb)=%0.2e',ub,objFcn(ub));
            plot(ub,objFcn(ub),'.','MarkerSize',markSize,'DisplayName',label);
            xline(ub,'--','HandleVisibility','off')

            if numel(Lambda) == 1
                label = sprintf('Lambda =%0.2e, f(Lambda)=%0.2e',Lambda,objFcn(Lambda));
                plot(Lambda,objFcn(Lambda),'.','MarkerSize',markSize,'DisplayName',label);
            else
                plot(Lambda,objFcn(Lambda),'o','MarkerSize',markSize / 2,'DisplayName','trial points');
            end

            if exist('Lambda0','var')
                label = sprintf('Lambda0 =%0.2e, f(Lambda0)=%0.2e',Lambda0,objFcn(Lambda0));
                plot(Lambda0,objFcn(Lambda0),'.','MarkerSize',markSize,'DisplayName',label);
            end
            legend('Location','best');
            hold off;
            
           
        end

    end
    
    methods (Static)

        function [f, g, H] = sDP(lambda,ZU,WZC,WZCZU,WU,S,sumLambda,tau2,n_calTk,sigma2)
        % get sampled DP function and derivatives

            if nargout == 1
                rnorm = tikParamFctn.resnorm(lambda, ZU, WZC, WZCZU, WU, S, sumLambda);
                f = (rnorm - tau2*n_calTk*sigma2).^2;

            elseif nargout == 2
                [rnorm,gres] = tikParamFctn.resnorm(lambda, ZU, WZC, WZCZU, WU, S, sumLambda);
                f = (rnorm - tau2*n_calTk*sigma2)^2;
                g = 2*(rnorm - tau2*n_calTk*sigma2).*gres;

            elseif nargout == 3
                [rnorm,gres,Hres] = tikParamFctn.resnorm(lambda, ZU, WZC, WZCZU, WU, S, sumLambda);
                f = (rnorm - tau2*n_calTk*sigma2).^2;
                g = 2*(rnorm - tau2*n_calTk*sigma2).*gres;
                H = 2*(rnorm - tau2*n_calTk*sigma2).*Hres +2*gres.*gres;
            end

        end

        function [f, g, H] = sUPRE(lambda,ZU, WZC, WZCZU, WU, S, sumLambda,sigma2, n_target)
        % get sampled UPRE function and derivatives

            if nargout == 1
                rnorm = tikParamFctn.resnorm(lambda, ZU, WZC, WZCZU, WU, S, sumLambda);
                tterm = tikParamFctn.traceterm(lambda, ZU, S, sumLambda, n_target);
                f = rnorm + 2*sigma2.*tterm;

            elseif nargout == 2
                [rnorm, gres] = tikParamFctn.resnorm(lambda, ZU, WZC, WZCZU, WU, S, sumLambda);
                [tterm,gtrace] = tikParamFctn.traceterm(lambda, ZU, S, sumLambda, n_target);
                f = rnorm + 2*sigma2.*tterm;
                g = gres  + 2*sigma2.*gtrace;

            elseif nargout == 3
                [rnorm, gres, Hres] = tikParamFctn.resnorm(lambda, ZU, WZC, WZCZU, WU, S, sumLambda);
                [tterm,gtrace, Htrace] = tikParamFctn.traceterm(lambda, ZU, S, sumLambda, n_target);
                f = rnorm + 2*sigma2.*tterm;
                g = gres  + 2*sigma2.*gtrace;
                H = Hres  + 2*sigma2.*Htrace;
            end

        end


        function [f, g, H] = sGCV(lambda,ZU, WZC, WZCZU, WU, S, sumLambda,n_calTk, n_target)
        % get sampled GCV function and derivatives

            if nargout == 1
                rnorm = tikParamFctn.resnorm(lambda, ZU, WZC, WZCZU, WU, S, sumLambda);
                tterm = tikParamFctn.traceterm(lambda, ZU, S, sumLambda, n_target);
                f     = rnorm ./ (n_calTk - tterm).^2;

            elseif nargout == 2
                [rnorm, gres]  = tikParamFctn.resnorm(lambda, ZU, WZC, WZCZU, WU, S, sumLambda);
                [tterm,gtrace] = tikParamFctn.traceterm(lambda, ZU, S, sumLambda, n_target);
                f  = rnorm ./ (n_calTk - tterm).^2;
                gn = gres .* (n_calTk - tterm) + 2 * rnorm .* gtrace;
                gd = (n_calTk- tterm).^3;
                g  = gn ./ gd;

            elseif nargout == 3
                [rnorm, gres, Hres]    = tikParamFctn.resnorm(lambda, ZU, WZC, WZCZU, WU, S, sumLambda);
                [tterm,gtrace, Htrace] = tikParamFctn.traceterm(lambda, ZU, S, sumLambda, n_target);
                f   = rnorm ./ (n_calTk - tterm).^2;
                gn  = gres .* (n_calTk - tterm) + 2 * rnorm .* gtrace;
                gd  = (n_calTk - tterm).^3;
                g   = gn./gd;
                dgn = gres .* (-gtrace) + Hres .* (n_calTk - tterm) + 2 * rnorm .* Htrace + 2 * gres .* gtrace;
                H   = (gd .* dgn - gn .* (-3 * (n_calTk- tterm).^2 .* gtrace)) ./ gd.^2;
            end

        end

        function [rnorm, g, H] = resnorm(lambda, ZU, WZC, WZCZU, WU, S, sumLambda)
            %
            % Computes the residual term, the gradient and the Hessian
            % (derivatives w.r.t. lambda)

            % ZU = Z' * U; WZC = W * Z - C; WZCZU = WZC * ZU; WU = W * U;
          
            sigma  = diag(S).^2 + sumLambda + lambda(:)';

            % residual norm
            m1    = WZCZU + WU .* reshape(lambda(:),1,1,[]);
            m2    = ZU' ./ reshape(sigma,size(sigma,1),1,[]);
            res   = pagemtimes(m1,m2) - WZC;
            rnorm = reshape(sum(res.^2,[1,2]),[],1);
            
            if nargout > 1
                % 1st derivative
                sigma2 = sigma.^2;
                
                dSdl = pagemtimes(WU, ZU' ./ reshape(sigma,size(sigma,1),1,[]));
                m1   = WZCZU + WU .* reshape(lambda(:),1,1,[]);
                m2   = ZU' ./ reshape(sigma2,size(sigma2,1),1,[]);
                dSdl = dSdl - pagemtimes(m1,m2);
                g    = reshape(2 * sum(res .* dSdl,'all'),[],1);

                if nargout > 2
                    % 2nd derivative
                    sigma3 = sigma.^3;
                    dS2    = -2 * pagemtimes(WU, ZU' ./ reshape(sigma2,size(sigma2,1),1,[]));
                    dS2    = dS2 + 2 * pagemtimes(WZCZU + WU .* reshape(lambda(:),1,1,[]),ZU'./reshape(sigma3,size(sigma3,1),1,[]));
                    H      = reshape(2 * sum(dSdl.^2,'all') + 2 * sum(res .* dS2,'all'),[],1);
                end
            end

        end

        function [tterm, g, H] = traceterm(lambda,ZU,S,sumLambda,n_target)
            %
            % Computes the trace term, the gradient and the Hessian
            % (derivatives w.r.t. lambda)

            % ZU = Z' * U; WZC = W * Z - C; WZCZU = WZC * ZU; WU = W * U;

            % add lambda to diag(S).^2 + sumLambda for each lambda
            sigma = diag(S).^2 + sumLambda + lambda(:)';
            sumZU = sum(ZU.^2,1);

            % define trace function
            tterm = reshape(n_target * (sumZU * (1 ./ sigma)),[],1);

            if nargout > 1
                % 1st derivative
                sigma2 = sigma.^2;
                g = reshape(-n_target * (sumZU * (1 ./ sigma2)),[],1);

                if nargout > 2
                    % 2nd derivative
                    sigma3 = sigma.^3;
                    H = reshape(2 * n_target * (sumZU * (1 ./ sigma3)),[],1);
                end
            end

        end


        
    end
    
end