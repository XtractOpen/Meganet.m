classdef classObjFctn < objFctn
    % classdef classObjFctn < objFctn
    %
    % Objective function for classification,i.e., 
    %
    %   J(W) = loss(h(W*Y), C) + R(W),
    %
    % where 
    % 
    %   W    - weights of the classifier
    %   h    - hypothesis function
    %   Y    - features
    %   C    - class labels
    %   loss - loss function object
    %   R    - regularizer (object)
    
    properties
        pLoss
        pRegW
        Y
        C
    end
    
    methods
        function this = classObjFctn(pLoss,pRegW,Y,C)
            
            if nargout==0 && nargin==0
                this.runMinimalExample;
                return;
            end
            
            this.pLoss  = pLoss;
            this.pRegW  = pRegW;
            this.Y      = Y;
            this.C      = C;
            
        end
        
        function [Jc,para,dJ,H,PC] = eval(this,W,idx)
            if not(exist('idx','var')) || isempty(idx)
                Y = this.Y;
                C = this.C;
            else
                Y = this.Y(:,idx);
                C = this.C(:,idx);
            end
            
            [Jc,hisLoss,dJ,H] = getMisfit(this.pLoss,W,Y,C);
            para = struct('F',Jc,'hisLoss',hisLoss);
            
            
            if not(isempty(this.pRegW))
                [Rc,hisReg,dR,d2R] = regularizer(this.pRegW,W);
                para.hisReg = hisReg;
                para.Rc     = Rc;
                Jc = Jc + Rc; 
                dJ = vec(dJ)+ vec(dR);
                H = H + d2R;
                para.hisRW = hisReg;
            end

            if nargout>4
                PC = opEye(numel(W));
            end
        end
        
        function [str,frmt] = hisNames(this)
            [str,frmt] = hisNames(this.pLoss);
            if not(isempty(this.pRegW))
                [s,f] = hisNames(this.pRegW);
                s{1} = [s{1} '(W)'];
                str  = [str, s{:}];
                frmt = [frmt, f{:}];
            end
        end
        
        function his = hisVals(this,para)
            his = hisVals(this.pLoss,para.hisLoss);
            if not(isempty(this.pRegW))
                his = [his, hisVals(this.pRegW,para.hisRW)];
            end
        end
        
        
        function str = objName(this)
            str = 'classObjFun';
        end
        
        function runMinimalExample(~)
            
            pClass = regressionLoss();
            
            nex = 400;
            nf  = 2;
            nc  = 2;
            
            Y = randn(nf,nex);
            C = zeros(nf,nex);
            
            C(1,Y(2,:)>0) = 1;
            C(2,Y(2,:)<=0) = 1;
            
            W = vec(randn(nc,nf+1));
            pReg   = tikhonovReg(speye(numel(W)));
            
            fctn = classObjFctn(pClass,pReg,Y,C);
            opt1  = newton('out',1,'maxIter',20);
            
%             checkDerivative(fctn,W);
            [Wopt,his] = solve(opt1,fctn,W);
        end
    end
end










