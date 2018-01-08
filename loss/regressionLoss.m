classdef regressionLoss
    % classdef regressionLoss
    %   
    % class describing loss function for regression
    %
    % loss(W,Y) = 0.5*| W*Y - C|^2
    
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
            szW  = [size(C,1),size(Y,1)+1];
            W    = reshape(W,szW);
            
            nex = size(Y,2);
            if this.addBias==1
                Y = [Y; ones(1,nex)];
            end
            res = W*Y - C;
            F   = .5*sum(vec(res.^2))/nex;
            para = [nex*F,nex];
            
            if (doDW) && (nargout>=3)
               dWF  = vec(res*(Y'/nex));
            end
            if (doDW) && (nargout>=4)
                d2WF = kron(speye(size(C,1)),(Y*Y')/nex);
            end
            if doDY && (nargout>=5)
                if this.addBias==1
                    W = W(:,1:end-1);
                end
                dYF  = vec(res'*(W/nex));
            end
            if doDY && nargout>=6
                d2YF = W'*W/nex;
            end
        end
        
        
        
        function [str,frmt] = hisNames(this)
            str  = {'F'};
            frmt = {'%-12.2e'};
        end
        function str = hisVals(this,para)
            str = para(1)/para(2);
        end 
        
        function Cp = getLabels(this,W,Y)
            nex    = size(Y,2);
            if this.addBias==1
                Y     = [Y; ones(1,nex)];
            end
            P      = W*Y;
            [~,jj] = max(P,[],1);
            Cp     = zeros(numel(P),1);
            ind    = sub2ind(size(P),jj(:),(1:nex)');
            Cp(ind)= 1;
            Cp     = reshape(Cp,size(P,1),[]);
        end
    end
    
end

