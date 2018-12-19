classdef dnnObjFctn < objFctn
    % classdef dnnObjFctn < objFctn
    %
    % objective function for deep neural networks 
    %
    % J(theta,W) = loss(h(W*Y(theta)), C) + Rtheta(theta) + R(W),
    %
    
    properties
        net
        pRegTheta
        pLoss
        pRegW
        Y
        C
        useGPU
        precision
    end
    
    methods
        function this = dnnObjFctn(net,pRegTheta,pLoss,pRegW,Y,C,varargin)
            
            if nargout==0 && nargin==0
                this.runMinimalExample;
                return;
            end
            useGPU = [];
            precision = [];
            for k=1:2:length(varargin)     % overwrites default parameter
                eval([varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            this.net    = net;
            this.pRegTheta = pRegTheta;
            this.pLoss  = pLoss;
            this.pRegW  = pRegW;
            
            if not(isempty(useGPU))
                this.useGPU = useGPU;
            end
            
            if not(isempty(precision))
                this.precision=precision;
            end
            [Y,C] = gpuVar(this.useGPU,this.precision,Y,C);
            this.Y         = Y;
            this.C         = C;
            
            
        end
        function [theta,W] = split(this,thetaW)
            nth = nTheta(this.net);
            theta  = thetaW(1:nth);
            W   = thetaW(nth+1:end);
        end
        
        function [Jc,para,dJ,H,PC] = eval(this,thetaW,idx)
            if not(exist('idx','var')) || isempty(idx)
                Y = this.Y;
                C = this.C;
            else
                Y = this.Y(:,idx);
                C = this.C(:,idx);
            end
            compGrad = nargout>2;
            compHess = nargout>3;
            
            dJth = []; dJW = []; Hth = []; HW = []; PC = [];
            
            [theta,W] = split(this,thetaW);
            
            % evaluate loss
            if compGrad
                %                 [YN,Yall,dA]                   = fwd(this.net,Kb,this.Y);
                [YN,J] = linearizeTheta(this.net,theta,Y);
                
                szYN  = size(YN);
                nex = szYN(end);
                YN = reshape(YN,[],nex); % loss expects 2D input
                [F,hisLoss,dWF,d2WF,dYF,d2YF] = getMisfit(this.pLoss,W,YN,C);
                dYF = reshape(dYF,szYN);
                
                dJth = J'*dYF;
                dJW  = dWF;
                Jc   = F;
                if compHess
                    Hthmv = @(x) J'*(d2YF*(J*x)); %  JTmv(this.net, reshape(d2YF* Jmv(this.net,x,[],Kb,Yall,dA),size(YN)), Kb,Yall,dA);
                    Hth   = LinearOperator(numel(theta),numel(theta),Hthmv,Hthmv);
                    HW    = d2WF;
                end
            else
                [YN]                   = forwardProp(this.net,theta,Y);
                szYN  = size(YN);
                nex = szYN(end);
                YN = reshape(YN,[],nex); % loss expects 2D input
                [F,hisLoss] = getMisfit(this.pLoss,W,YN,C);
                Jc = F;
            end
            para = struct('F',F,'hisLoss',hisLoss);

            % evaluate regularizer for DNN weights             
            if not(isempty(this.pRegTheta))
                [Rth,hisRth,dRth,d2Rth]      = regularizer(this.pRegTheta,theta);
                Jc = Jc + Rth;
                if compGrad
                    dJth = dJth + dRth;
                end
                if compHess
                       Hth  = Hth + d2Rth;
                end
                para.Rth = Rth;
                para.hisRth = hisRth;
            end
            
            % evaluare regularizer for classification weights
            if not(isempty(this.pRegW))
                [RW,hisRW,dRW,d2RW]         = regularizer(this.pRegW, W);
                Jc = Jc + RW;
                if compGrad
                    dJW = dJW + dRW;
                end
                if compHess
                    HW = HW + d2RW;
                end
                para.RW = RW;
                para.hisRW = hisRW;
            end
            
            % stack the derivatives
            dJ   = [dJth; dJW];
            if compHess
                H  = blkdiag(Hth, HW);
            end
            if nargout>4
%                 PC = blkdiag(getPC(this.pRegTheta),getPC(this.pRegW));
                PC = blkdiag(opEye(numel(theta)),opEye(numel(W)));
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
            his = hisVals(this.pLoss,para.hisLoss);
            if not(isempty(this.pRegTheta))
                his = [his, hisVals(this.pRegTheta,para.hisRth)];
            end
            if not(isempty(this.pRegW))
                his = [his, hisVals(this.pRegW,para.hisRW)];
            end
        end
        
        function str = objName(this)
            str = 'dnnObjFun';
        end
        
        % ------- functions for handling GPU computing and precision ----
        function this = set.useGPU(this,value)
            if isempty(value)
                return
            elseif(value~=0) && (value~=1)
                error('useGPU must be 0 or 1.')
            else
                if not(isempty(this.net)); this.net.useGPU       = value; end
                if not(isempty(this.pRegTheta)); this.pRegTheta.useGPU       = value; end
                if not(isempty(this.pRegW)); this.pRegW.useGPU       = value; end
                
                [this.Y,this.C] = gpuVar(value,this.precision,...
                                                         this.Y,this.C);
            end
        end
        function this = set.precision(this,value)
            if isempty(value)
                return
            elseif not(strcmp(value,'single') || strcmp(value,'double'))
                error('precision must be single or double.')
            else
                if not(isempty(this.net)); this.net.precision       = value; end
                if not(isempty(this.pRegTheta)); this.pRegTheta.precision       = value; end
                if not(isempty(this.pRegW)); this.pRegW.precision       = value; end
                
                [this.Y,this.C] = gpuVar(this.useGPU,value,...
                                                         this.Y,this.C);
            end
        end
        function useGPU = get.useGPU(this)
                useGPU = -ones(3,1);
                
                if not(isempty(this.net)) && not(isempty(this.net.useGPU))
                    useGPU(1) = this.net.useGPU;
                end
                if not(isempty(this.pRegTheta)) && not(isempty(this.pRegTheta.useGPU))
                    useGPU(2) = this.pRegTheta.useGPU;
                end
                if not(isempty(this.pRegW)) && not(isempty(this.pRegW.useGPU))
                    useGPU(3) = this.pRegW.useGPU;
                end
                
                useGPU = useGPU(useGPU>=0);
                if all(useGPU==1)
                    useGPU = 1;
                elseif all(useGPU==0)
                    useGPU = 0;
                else
                    error('useGPU flag must agree');
                end
        end
        function precision = get.precision(this)
            isSingle    = -ones(3,1);
            isSingle(1) = strcmp(this.net.precision,'single');
            if not(isempty(this.pRegTheta)) && not(isempty(this.pRegTheta.precision))
                isSingle(2) = strcmp(this.pRegTheta.precision,'single');
            end
            if not(isempty(this.pRegW)) &&  not(isempty(this.pRegW.precision))
                isSingle(3) = strcmp(this.pRegW.precision,'single');
            end
                isSingle = isSingle(isSingle>=0);
            if all(isSingle==1)
                precision = 'single';
            elseif all(isSingle==0)
                precision = 'double';
            else
                error('precision flag must agree');
            end

        end
        function runMinimalExample(~)
            
            nex    = 100; nf =2;
            
            blocks    = cell(2,1);
            blocks{1} = NN({singleLayer(dense([2*nf nf]))});
            blocks{2} = ResNN(doubleLayer(dense([2*nf 2*nf]),dense([2*nf 2*nf])),2,.1);
            net    = Meganet(blocks);
            nth = nTheta(net);
            theta  = randn(nth,1);
            
            % training data
            Y = randn(nf,nex);
            C = zeros(nf,nex);
            C(1,Y(2,:)>Y(1,:).^2) = 1;
            C(2,Y(2,:)<=Y(1,:).^2) = 1;
            
            % validation data
            Yv = randn(nf,nex);
            Cv = zeros(nf,nex);
            Cv(1,Yv(2,:)>Yv(1,:).^2) = 1;
            Cv(2,Yv(2,:)<=Yv(1,:).^2) = 1;
            
            
            pLoss = regressionLoss();
            W = vec(randn(2,2*nf+1));
            pRegW  = tikhonovReg(opEye(numel(W)));
            pRegTheta    = tikhonovReg(opEye(numel(theta)));
            
            fctn = dnnObjFctn(net,pRegTheta,pLoss,pRegW,Y,C);
            fval = dnnObjFctn(net,[],pLoss,[],Yv,Cv);
            % [Jc,para,dJ,H,PC] = fctn([Kb(:);W(:)]);
            % checkDerivative(fctn,[Kb(:);W(:)])
            
            opt1  = newton('out',1,'maxIter',20);
            opt2  = sd('out',1,'maxIter',20);
            opt3  = nlcg('out',1,'maxIter',20);
            [KbWopt1,His1] = solve(opt1,fctn,[theta(:); W(:)],fval);
            [KbWopt2,His2] = solve(opt2,fctn,[theta(:); W(:)],fval);
            [KbWopt3,His3] = solve(opt3,fctn,[theta(:); W(:)],fval);
            
            figure(1); clf;
            subplot(1,3,1);
            semilogy(His1.his(:,2)); hold on;
            semilogy(His2.his(:,2)); 
            semilogy(His3.his(:,2)); hold off;
            legend('newton','sd','nlcg');
            title('objective');

            subplot(1,3,2);
            semilogy(His1.his(:,4)); hold on;
            semilogy(His2.his(:,4)); 
            semilogy(His3.his(:,4)); hold off;
            legend('newton','sd','nlcg');
            title('opt.cond');

            subplot(1,3,3);
            plot(His1.his(:,10)); hold on;
            plot(His1.his(:,end),'--'); hold on;
            plot(His2.his(:,8)); 
            plot(His2.his(:,end),'--');
            plot(His3.his(:,8)); 
            plot(His3.his(:,end),'--'); hold off;
            legend('newton-train','newton-val','sd-train','sd-val','nlcg-train','nlcg-val');
            title('loss');
        end
    end
end










