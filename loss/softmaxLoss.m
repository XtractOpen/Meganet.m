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
            this.theta   = 0;
            this.addBias = 1;
            for k=1:2:length(varargin)     % overwrites default parameter
                eval(['this.' varargin{k},'=varargin{',int2str(k+1),'};']);
            end
        end
        
        
        
        function [F,para,dWF,d2WF,dYF,d2YF,dWYF,d2WYF] = getMisfit(this,W,Y,C,varargin)
        % [F,para,dWF,d2WF,dYF,d2YF] = getMisfit(this,W,Y,C,varargin)
        %
        % Input:
        %  
        %   W - vector containing the weights (nWeights-by-1)
        %   Y - 2D matrix, features (nFeatures-by-nExamples)
        %   C - 2D matrix, ground truth classes (nClasses-by-nExamples)
        %
        % Optional Input:
        %
        %   set via varargin
        %
        % Output:
        %
        %  F     - loss (average per example)
        %  para  - vector of 3 values: unaveraged loss, nExamples, error
        %  dWF   - gradient of F wrt W (nWeights-by-1)
        %  d2WF  - Hessian of F wrt W  (LinearOperator, nWeights-by-nWeights)
        %  dYF   - gradient of F wrt Y (nFeatures*nExamples-by-1)
        %  d2YF  - Hessian of F wrt Y  (LinearOperator, nFeat*nEx-by-nFeat*nEx)
        %  dWYF  - gradient of F wrt W*Y (nTargets*nExamples-by-1)
        %  d2WYF - Hessian of F wrt W*Y (LinearOperator, nTargets*nEx-by-nTargets*nEx)
            
            doDY = (nargout>3);
            doDW = (nargout>1);
            for k=1:2:length(varargin)     % overwrites default parameter
                eval([varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            dWF = []; d2WF = []; dYF =[]; d2YF = [];
            
            % check dimensions here and print error message
            szY  = size(Y);
            nex = szY(2);

            try
                if this.addBias==1
                    Y   = [Y; ones(1,nex)];
                end
                szW  = [size(C,1),size(Y,1)];
                W    = reshape(W,szW);
                WY   = W*Y;
            catch err
                error(['\nError. \nW and Y are expected to be 2-D matrices prior to W*Y.\n', ...
                    'dims of W: [%s]  and dims of Y: [%s] provided.'], num2str(size(W)), num2str(size(Y)) );
                % rethrow(err);
            end
            
            WY   = WY - max(WY,[],1);
            S    = exp(WY);
            
            Cp   = getLabels(this,S);
            err  = nnz(C-Cp)/2;
            F    = -sum(sum(C.*(WY))) + sum(log(sum(S,1)));
            para = [F,nex,err];
            F    = F/nex;

            
            if (doDW) && (nargout>=2)
               dWYF = (-C + S./sum(S,1))/ nex;
               dWF  = vec(dWYF*Y');
            end
            if (doDW) && (nargout>=3)
%                 d2WYF = @(U)  (1 / nex) *(this.theta *U + (U.*S)./sum(S,1) - ...
%                     S.*(sum(S.*U,1)./sum(S,1).^2));
                sumS = sum(S,1);
                d2WYF = @(U) this.d2Fmv(U,S,nex,sumS);
                matW  = @(W) reshape(W,szW);
                d2WFmv  = @(U) vec((d2WYF(matW(U)*Y)*Y'));
                d2WF = LinearOperator(prod(szW),prod(szW),d2WFmv,d2WFmv);
            end
            if doDY && (nargout>=4)
                if this.addBias==1
                    W = W(:,1:end-1);
                end
                dYF  =   vec(W'*dWYF);
            end
            if doDY && nargout>=5
                WI     = @(T) W*T;  %kron(W,speye(size(Y,1)));
                WIT    = @(T) W'*T;
                matY   = @(Y) reshape(Y,szY);
                d2YFmv = @(T) WIT(((d2WYF(WI(matY(T))))));
    
                d2YF = LinearOperator(prod(szY),prod(szY),d2YFmv,d2YFmv);
            end
        end

        function d2Fv = d2Fmv(this,U,S,nex,sumS)
            SU = S.*U;
            sumSU = sum(SU,1);
            
            d2Fv = SU./sumS - S.*(sumSU./sumS.^2);
            if this.theta > 0
                d2Fv = d2Fv + this.theta*U;
            end
            d2Fv = (1/nex)*d2Fv;
        end
        %%
        function [F,para,dF,d2F,H] = eval(this,WY,C,varargin)
            
            nex   = size(C,2);
            doDF  = (nargout>2);
            doD2F = (nargout>3);
            reduceDim=true;
            for k=1:2:length(varargin)     % overwrites default parameter
                eval([varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            dF = []; d2F = [];
            
             s  = max(WY,[],1);
            WY = WY-s;
            S    = exp(WY);
            [Cp,P]   = getLabels(this,S);
            err  = nnz(C-Cp)/2;
            F    = (1/nex)*(-sum(C.*(WY),1) +  log(sum(S,1)));
            para = [sum(F),nex,err];
            if reduceDim
                F    = sum(F,2);
            end
           
            if (doDF) && (nargout>=2)
               dF   = (-C + P)/nex; 
               if reduceDim
                   dF = sum(dF,2);
               end
            end
            if (doD2F) && (nargout>=3)
                H =(1/nex)* reshape(P,[],1,nex).*(eye(size(S,1)).*ones(1,1,nex) - reshape(P,1,[],nex));
                d2Fmv = @(x) squeeze(pagemtimes(H,reshape(x,[],1,nex)));
                szS = size(S);       
                d2F = LinearOperator(prod(szS),prod(szS),d2Fmv,d2Fmv);
            end
        end

        
        %%
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
                nex = size(S,2);
            else
                szY  = size(Y);
                nex  = szY(end);
                Y = reshape(Y,[],nex);
                if this.addBias==1
                    Y   = [Y; ones(1,nex)];
                end
                nf  = size(Y,1);
                W    = reshape(W,[],nf);
            
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

