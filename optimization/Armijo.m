classdef Armijo
    % classdef Armijo
    %   
    % Armijo linesearch object
    
    properties
        gamma
        maxIter
        P
    end
    
    methods
        function this = Armijo(varargin)
            this.gamma   = 1e-3;
            this.maxIter = 10;
            this.P       = @(x) x;
            for k=1:2:length(varargin)     % overwrites default parameter
                eval(['this.' varargin{k},'=varargin{',int2str(k+1),'};']);
            end;
        end
        
        function [xt,mu,lsIter] = lineSearch(this,fctn,xc,mu,s,Jc,dJ)
            lsIter = 1; dJds = dot(s(:),dJ(:));
            while lsIter <=this.maxIter
               xt = this.P(xc + mu*s);
               Jt = fctn(xt);
               if Jt < Jc + this.gamma*dJds
                   break;
               end
               
               mu = mu/2;
               lsIter = lsIter +1;
            end
        end
        
    end
    
end

