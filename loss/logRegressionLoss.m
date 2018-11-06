classdef logRegressionLoss
    % classdef logRegressionLoss
    %
    % object describing logistic regression loss function
    
    properties
        theta
        addBias
    end
    
    
    methods
        function this = logRegressionLoss(varargin)
            this.theta   = 1e-3;
            this.addBias = 1;
            for k=1:2:length(varargin)     % overwrites default parameter
                eval(['this.' varargin{k},'=varargin{',int2str(k+1),'};']);
            end
        end
        
        
        
        function [F,para,dWF,d2WF,dYF,d2YF] = getMisfit(this,W,Y,C,varargin)
            doDY = (nargout>3);
            doDW = (nargout>1);
            for k=1:2:length(varargin)     % overwrites default parameter
                eval([varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            dWF = []; d2WF = []; dYF =[]; d2YF = [];
            
            szY  = size(Y);
            nex  = szY(2);
            if this.addBias==1
                Y   = [Y; ones(1,nex)];
            end
            szW  = [size(C,1),size(Y,1)];
             W    = reshape(W,szW);
            
            
            S  = W*Y;
            Cp   = getLabels(this,S);
            err  = nnz(C-Cp)/2;
            posInd = (S>0);
            negInd = (0 >= S);
            F = -sum(C(negInd).*S(negInd) - log(1+exp(S(negInd))) ) - ...
                sum(C(posInd).*S(posInd) - log(exp(-S(posInd))+1) - S(posInd));
            para = [F,nex,err];
            F  = F/nex;
            
            
            if (doDW) && (nargout>=2)
                dF  = (C - 1./(1+exp(-S)));
                dWF = -Y*dF';
                dWF = vec(dWF)/nex;
            end
            if (doDW) && (nargout>=3)
                d2F  = 1./(2*cosh(S/2)).^2;
                matW  = @(W) reshape(W,szW);
                d2WFmv = @(U) Y*(((d2F + this.theta).*(matW(U/nex)*Y)))';
%                 d2WFmv = @(U) (((d2F + this.theta).*(matW(U/nex)*Y)))*Y';
                d2WF = LinearOperator(prod(szW),prod(szW),d2WFmv,d2WFmv);
            end
            if doDY && (nargout>=4)
                if this.addBias==1
                    W = W(:,1:end-1);
                end
                dYF  =   -vec(W'*dF)/nex;
            end
            if doDY && nargout>=5
                WI     = @(T) W*T;  %kron(W,speye(size(Y,1)));
                WIT    = @(T) W'*T;
                matY   = @(Y) reshape(Y,szY);
%                  d2YFmv = @(T) vec(WIT(((d2F(WI(matY(T/nex)))))));
                d2YFmv = @(T) WIT((d2F + this.theta).*(WI(matY(T/nex))));
    
                d2YF = LinearOperator(prod(szY),prod(szY),d2YFmv,d2YFmv);
            end
        end
        
        function [str,frmt] = hisNames(this)
            str  = {'F','accuracy'};
            frmt = {'%-12.2e','%-12.2f'};
        end
        function str = hisVals(this,para)
            str = [para(1)/para(2),(1-para(3)/para(2))*100];
        end
        function [Cp,P] = getLabels(this,W,Y)
            if nargin==2
                S = W;
            else
                [nf,nex] = size(Y);
                W      = reshape(W,[],nf+1);
                if this.addBias==1
                    Y     = [Y; ones(1,nex)];
                end
                
                
                S      = W*Y;
            end
            posInd = (S>0);
            negInd = (0 >= S);
            P = 0*S;
            P(posInd) = 1./(1+exp(-S(posInd)));
            P(negInd) = exp(S(negInd))./(1+exp(S(negInd)));
                Cp = P>.5;
            
        end
    end
    
end

