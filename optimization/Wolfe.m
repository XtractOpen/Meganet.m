classdef Wolfe
    % classdef Wolfe
    %   
    % Wolfe linesearch object
    
    properties
        muMax       = 5
        c1          = 1e-4   % Armijo condition
        c2          = 0.9    % Wolfe condition
        maxFunEvals = 20     % number of function evaluations
    end
    
    methods
        function this = Wolfe(varargin)

            for k = 1:2:length(varargin) 
                this.(varargin{k}) = varargin{k+1};
            end
            
            if nargout == 0
                this.test();
                return;
            end
        end
        
        function [xt,para] = lineSearch(this,fctn,xc,mu,dx,phi0,dJ)

            funEvals = 0;

            dphi0 = dJ'*dx;
            muOld = 0;
            phiOld = phi0;
            dphiOld = dphi0;

            while funEvals < this.maxFunEvals
                xt = xc + mu*dx;
                [phi,~,dJt] = eval(fctn,xt);
                funEvals = funEvals + 1;
                
                dphi = dJt'*dx;
                if (phi > phi0 + this.c1*mu*dphi0) || ((funEvals>1) && (phi>=phiOld))
                    [xt,mu,funEvals] = WolfeZoom(this,fctn,xc,dx,phi0,dphi0,...
                        muOld,phiOld,dphiOld,mu,phi,dphi,funEvals);
                    para.mu = mu;
                    para.funEvals = funEvals;
                    return
                end
                if abs(dphi) <= -this.c2*dphi0
                    para.mu = mu;
                    para.funEvals = funEvals;
                    return
                end

                if dphi >= 0
                    [xt,mu,funEvals] = WolfeZoom(this,fctn,xc,dx,phi0,dphi0,...
                        mu,phi,dphi,muOld,phiOld,dphiOld,funEvals);
                    para.mu = mu;
                    para.funEvals = funEvals;
                    return
                end
                muOld = mu;
                phiOld = phi;
                dphiOld = dphi;

                % choose new mu in (mu,muMax)  quadratic extrapolation
                mu = -(dphi0*mu^2)/(2*(phi-phi0-dphi0*mu));
                if (mu>this.muMax) || (mu < muOld)
                    % Increase mu by some multiple chosen so that we can reach upper
                    % bound quickly enough while still having some space left
                    mu = muOld* 10^((funEvals/this.maxFunEvals)*log10(this.muMax/muOld));
                end
            end

            if phi < phi0 + this.c1*mu*dphi0
                % warning('Only Armijo was satisfied');
                para.mu = mu;
                para.funEvals = funEvals;
                return;
            end
            keyboard
%             error("%s: unable to find feasible solution within %d function evaluations",...
%                 mfilename,funEvals);
            
        end
        
        function [xt,mu,funEvals] = WolfeZoom(this,fctn,xc,dx,phi0,dphi0,muLo,phiLo,dphiLo,muHi,phiHi,dphiHi,funEvals)

            % sanity checks
            if phiLo > phiHi
                error('inputs are ordered incorrectly')
            end
            if dphiLo*(muHi-muLo)>0
                error('wrong inputs')
            end

            if ~exist('funEvals','var')
                funEvals = 1;
            end

            while funEvals <= this.maxFunEvals
                % test if minimizer of quadratic Hermite polynomial is between muHi and muLo
                mu  = -(dphiLo*muHi^2)/(2*(phiHi-phiLo-dphiLo*muHi));
                eps = 0.1*abs(muHi-muLo);   
                if (mu<=eps+min(muHi,muLo)) || (mu>=max(muHi,muLo)-eps) % use bisection    
                    mu = 0.5*(muHi+muLo);
                end

                xt = xc + mu*dx;
                [phit,~,dJt] = eval(fctn,xt);
                

                dphit = dJt'*dx;
                if (phit > phi0 + this.c1*mu*dphi0) || (phit >= phiLo)
                    muHi = mu;
                    phiHi = phit;
                    dphiHi = dphit;
                else
                    if (abs(dphit) <= -this.c2* dphi0)
                        return;
                    elseif dphit*(muHi-muLo)>=0
                        muHi   = muLo;
                        phiHi  = phiLo;
                        dphiHi = dphiLo;
                    end
                    muLo = mu;
                    phiLo = phit;
                    dphiLo = dphit;
                    dJLo = dJt;
                end
                
                funEvals = funEvals + 1;
            end
            
            if phiLo < phi0 + this.c1*muLo*dphi0
                mu = muLo;
                xt = xc + mu * dx;
            %     warning('only Armijo satisfied');
                return;
            end
            mu = 0;
            
%             error("%s: unable to find feasible solution within %d function evaluations",...
%                                                                 mfilename,funEvals);

        end
        
        function[str,frmt] = hisNames(~)
            % define the labels for each column in his table
            str  = {'funEvalsLS','muLS'};
            frmt = {'%-12d','%-12.2e'};
        end
        
        function his = hisVals(~,para)
            his = [para.funEvals,para.mu];
        end
        
        function[] = test(this,showPlot)
            mu = 1.4;
            xc = 0;
            dx = 1;
            
            this.c1 = 1e-4; 
            this.c2 = 0.5;
            this.muMax = 5;
            this.maxFunEvals = 20;
            
            
            function[fc,dfc] = testFun(x,c)
                 fc = [x.^5 x.^4 x.^3 x.^2 x ones(size(x))]*c;
                 dfc = [5*x.^4 4*x.^3 3*x.^2 2*x ones(size(x)) zeros(size(x))]*c;
            end
            
            c = -poly([0.25, 0.5, 2.5, 3, 6])';
            fctn = @(x) testFun(x,c);
            
            [fc,dfc] = fctn(xc);
            
            [xt,~] = lineSearch(this,fctn,xc,mu,dx,fc,dfc);
            [ft,dft] = fctn(xt);
            
            
            testArmijo = (ft < fc + this.c1 * mu * dfc);
            testWolfe  =  (abs(dft' * dx) <= this.c2 * abs(dfc' * dx));
    
            fprintf('mu = %1.2e satisfies Armijo? %d Wolfe? %d\n',mu,testArmijo,testWolfe);

            if exist('showPlot','var') && showPlot
                tt = linspace(0,4,101)';
                [ff,df] = fctn(tt);
                
                
                fig = figure(1); clf;
                fig.Name = 'Wolfe Line Search Test';
                    % function
                    plot(tt,ff,'LineWidth',2);
                    hold on;
                    
                    % Armijo
                    plot(tt,ff(1) + this.c1 * tt * df(1),'--','LineWidth',2);
                    
                    % Wolfe
                    plot(tt(1:10),ff(1) + this.c2 * tt(1:10) * df(1),'--','LineWidth',2);
                    
                    % find points that satisfy each condition
                    idArmijo = (ff <= ff(1) + this.c1 * tt * df(1));
                    idWolfe = (abs(df)<= this.c2 * abs(df(1)));
                    
                    plot(tt(idArmijo),0 * tt(idArmijo) + min(ff(:)),'.','MarkerSize',15);
                    plot(tt(idWolfe),0 * tt(idWolfe) + min(ff(:)) + 0.5,'o','MarkerSize',5);

                    legend('f','Armijo','Wolfe','Armijo points','Wolfe points','Location','best');
                    axis('tight');
                    
                    xc = 0;
                    [fc,df] = fctn(xc);
                    dx = 1;
                    for k = 2:numel(tt)
                        mu0 = tt(k);
                        [xt,mu,~]  = lineSearch(this,fctn,xc,mu0,dx,fc,df);

                        [ft,dft] = fctn(xt);
                        assert(ft < fc)
                        assert(abs(dft' * dx) <= this.c2 * abs(df' * dx));
                        figure(1);
                        plot(mu,ft,'.r','MarkerSize',20,'HandleVisibility','off')
                    end

                    hold off;
            end
            
        end
        
    end

    
end

