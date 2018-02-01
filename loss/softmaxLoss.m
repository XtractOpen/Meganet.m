classdef softmaxLoss
    % classdef softmaxLoss
    %
    % object describing softmax loss function
    
    properties
       theta
       addBias
    end
   
    
    methods
        function this = softmaxLoss(varargin)
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
            WY   = W*Y;
            WY   = WY - max(WY,[],1);
            
            S    = exp(WY);
            
            Cp   = getLabels(this,S);
            err  = nnz(C-Cp)/2;
            F    = -sum(sum(C.*(WY))) + sum(log(sum(S,1)));
            para = [F,nex,err];
            F    = F/nex;

            
            if (doDW) && (nargout>=2)
               dF   = -C + S./sum(S,1); %S./sum(S,2));
               dWF  = vec(dF*(Y'/nex));
            end
            if (doDW) && (nargout>=3)
                d2F = @(U) this.theta *U + (U.*S)./sum(S,1) - ...
                    S.*(repmat(sum(S.*U,1)./sum(S,1).^2,size(S,1),1));
                matW  = @(W) reshape(W,szW);
                d2WFmv  = @(U) vec((d2F(matW(U/nex)*Y)*Y'));
                d2WF = LinearOperator(prod(szW),prod(szW),d2WFmv,d2WFmv);
            end
            if doDY && (nargout>=4)
                if this.addBias==1
                    W = W(:,1:end-1);
                end
                dYF  =   vec(W'*dF)/nex;
            end
            if doDY && nargout>=5
                WI     = @(T) W*T;  %kron(W,speye(size(Y,1)));
                WIT    = @(T) W'*T;
                matY   = @(Y) reshape(Y,szY);
                d2YFmv = @(T) vec(WIT(((d2F(WI(matY(T/nex)))))));
    
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
        function Cp = getLabels(this,W,Y)
            if nargin==2
                S = W;
                nex = size(S,2);
            else
                [nf,nex] = size(Y);
                W      = reshape(W,[],nf+1);
                if this.addBias==1
                    Y     = [Y; ones(1,nex)];
                end
                WY     = W*Y;
                WY     = WY - max(WY,[],1);
                S      = exp(WY);
            end
            P      = S./sum(S,1);
            [~,jj] = max(P,[],1);
            Cp     = zeros(numel(P),1);
            ind    = sub2ind(size(P),jj(:),(1:nex)');
            Cp(ind)= 1;
            Cp     = reshape(Cp,size(P,1),[]);
        end
    end
    
end

