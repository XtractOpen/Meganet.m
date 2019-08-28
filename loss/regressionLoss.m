classdef regressionLoss
    % classdef regressionLoss
    %   
    % class describing loss function for regression
    %
    % loss(W,Y) = 0.5*| W*Y - C|_Gamma^2
    
    properties
        addBias 
    end
    
    methods
        function this = regressionLoss(varargin)
            this.addBias = 1;
            for k=1:2:length(varargin)     % overwrites default parameter
                eval(['this.' varargin{k},'=varargin{',int2str(k+1),'};']);
            end
        end
        
        function [F,para,dWF,d2WF,dYF,d2YF] = getMisfit(this,W,Y,C,varargin)
            doDY = nargout>4;
            doDW = nargout>2;
            for k=1:2:length(varargin)     % overwrites default parameter
                eval([varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            dWF = []; d2WF = []; dYF =[]; d2YF = [];
            szW  = [size(C,1),size(Y,1)+this.addBias];
            szY  = size(Y);
            W    = reshape(W,szW);
            
            
            if this.addBias==1
                Y = padarray(Y,[1,0],'post');
            end
            WY  = W*Y;
            [F,para,dF,d2F] = eval(this,WY,C,'doDerivative',doDY || doDW);
            Cp        = getLabels(this,WY);
            para  = [para nnz(C-Cp)/2];
            
            
            if (doDW) && (nargout>=3)
               dWF  = vec(dF*Y');
            end
            if (doDW) && (nargout>=4)
                matW  = @(W) reshape(W,szW);
                d2WFmv  = @(U) vec(((matW(d2F*U)*Y)*Y'));
                d2WF = LinearOperator(prod(szW),prod(szW),d2WFmv,d2WFmv);
            end
            if doDY && (nargout>=5)
                if this.addBias==1
                    W = W(:,1:end-1);
                end
                dYF  = vec(W'*dF);
            end
            if doDY && nargout>=6
                matY   = @(Y) reshape(Y,szY);
                d2YFmv = @(U) W'*(W*matY(d2F*U));
                d2YF = LinearOperator(prod(szY),prod(szY),d2YFmv,d2YFmv);                
            end
        end
        
        function [F,para,dF,d2F] = eval(this,WY,C,varargin)
            doDerivative = (nargout>2);
            for k=1:2:length(varargin)     % overwrites default parameter
                eval([varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            dF = []; d2F = [];
            
            nex = size(C,2);
            res = WY - C;
            F   = .5*sum(vec(res.^2));
            para = [F,nex];
            F = F/nex;
            if doDerivative
                dF  = res/nex;
                d2F = 1/nex;
            end
        end
        
        
        
        function [str,frmt] = hisNames(this)
              str  = {'F','accuracy'};
            frmt = {'%-12.2e','%-12.2f'};
       end
        function str = hisVals(this,para)
            str = [para(1)/para(2),(1-para(3)/para(2))*100];
        end 
        
        function Cp = getLabels(this,WY)
            nex    = size(WY,2);
            [~,jj] = max(WY,[],1);
            Cp     = zeros(numel(WY),1);
            ind    = sub2ind(size(WY),jj(:),(1:nex)');
            Cp(ind)= 1;
            Cp     = reshape(Cp,size(WY,1),[]);
        end
    end
    
end

