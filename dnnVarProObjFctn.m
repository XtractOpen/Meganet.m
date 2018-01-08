classdef dnnVarProObjFctn < objFctn
    % classdef dnnVarObjFctn < objFctn
    %
    % variable projection objective function for deep neural networks 
    %
    % J(theta) = loss(h(W(theta)*Y(theta)), C) + Rtheta(Kb) + R(W(theta)),
    %
    % where W(theta) = argmin_W loss(h(W*Y(theta))), C) + R(W)
    
    properties
        net
        pRegTheta
        pLoss
        pRegW
        Y
        C
        optClass 
    end
    
    methods
        function this = dnnVarProObjFctn(net,pRegTheta,pLoss,pRegW,optClass,Y,C)
            
            if nargout==0 && nargin==0
                this.runMinimalExample;
                return;
            end
            
            this.net    = net;
            this.pRegTheta = pRegTheta;
            this.pLoss  = pLoss;
            this.pRegW  = pRegW;
            this.optClass = optClass;
            this.Y      = Y;
            this.C      = C;
            
        end
        
        function [Jc,para,dJ,H,PC] = eval(this,theta)
            compGrad = nargout>2;
            compHess = nargout>3;
            dJ = 0.0; H = []; PC = [];
            
            % project onto W
            [YN,J] = linearizeTheta(this.net,theta,this.Y); % forward propagation
            fctn   = classObjFctn(this.pLoss,this.pRegW,YN,this.C);
            W      = solve(this.optClass,fctn,zeros(size(this.C,1)*(size(YN,1)+1),1));
            [F,hisLoss,~,~,dYF,d2YF] = getMisfit(this.pLoss,W,YN,this.C);
            if compGrad
                dJ = J'*dYF;
            end
            if compHess
                HKbmv = @(x) J'*(d2YF*(J*x));
                H   = LinearOperator(numel(theta),numel(theta),HKbmv,HKbmv);
            end
            
            para = struct('F',F,'hisLoss',hisLoss);
            Jc   = F;
            
            % evaluate regularizer for DNN weights
            if not(isempty(this.pRegTheta))
                [Rth,hisRth,dRth,d2Rth]      = regularizer(this.pRegTheta,theta);
                Jc = Jc + Rth;
                if compGrad
                    dJ = dJ + dRth;
                end
                if compHess
                    H  = H + d2Rth;
                end
                para.Rth = Rth;
                para.hisRth = hisRth;
            end
            
            if not(isempty(this.pRegW))
                [RW,hisRW]         = regularizer(this.pRegW, W);
                Jc = Jc + RW;
                para.RW = RW;
                para.hisRW = hisRW;
            end
            para.W = W;
            if nargout>4
                PC = getPC(this.pRegTheta);
            end
            
        end
        
        
        function [str,frmt] = hisNames(this)
            [str,frmt] = hisNames(this.pLoss);
            if not(isempty(this.pRegTheta))
                [s,f] = hisNames(this.pRegTheta);
                s{1} = [s{1} '(theta)'];
                str  = [str, s{:}];
                frmt = [frmt, f{:}];
            end
            if not(isempty(this.pRegW))
                [s,f] = hisNames(this.pRegW);
                s{1} = [s{1} '(W)'];
                str  = [str, s{:}];
                frmt = [frmt, f{:}];
            end
        end
        
        function his = hisVals(this,para)
            his = hisVals(this.pLoss,sum(para.hisLoss,1));
            if not(isempty(this.pRegTheta))
                his = [his, hisVals(this.pRegTheta,para.hisRth)];
            end
            if not(isempty(this.pRegW))
                his = [his, hisVals(this.pRegW,para.hisRW)];
            end
        end
        
        
        function str = objName(this)
            str = 'dnnVarPro';
        end
        
        function runMinimalExample(~)
            
            nex    = 400; nf =2;
            
            blocks = cell(2,1);
            blocks{1} = NN({singleLayer(dense([2*nf nf]))});
            blocks{2} = ResNN(singleLayer(dense([2*nf 2*nf])),1,.1);
            net    = Meganet(blocks);
            nth = nTheta(net);
            theta  = randn(nth,1);
            
            Y = randn(nf,nex);
            C = zeros(nf,nex);
            C(1,Y(2,:)>Y(1,:).^2) = 1;
            C(2,Y(2,:)<=Y(1,:).^2) = 1;
            
            pLoss = softmaxLoss();
            W = vec(randn(2,2*nf+1));
            pRegW        = tikhonovReg(opEye(numel(W)));
            pRegTheta    = tikhonovReg(opEye(numel(theta)));
            
            newtInner =newton('out',0,'maxIter',5);
            
            fctn = dnnVarProObjFctn(net,pRegTheta,pLoss,pRegW,newtInner,Y,C);
            %[Jc,para,dJ,H,PC] = fctn([Kb(:);W(:)]);
            %  checkDerivative(fctn,Kb(:))
            newtOuter =newton('out',1,'maxIter',60);
            
            [KbWopt,his] = solve(newtOuter,fctn,theta(:));
        end
    end
end










