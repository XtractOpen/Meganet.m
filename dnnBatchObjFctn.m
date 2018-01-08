classdef dnnBatchObjFctn < objFctn
    % classdef dnnBatchObjFctn < objFctn
    %
    % objective function for deep neural networks 
    %
    %       J(theta,C) = loss(h(W*Y(theta)), C) + Rtheta(theta) + R(W)
    %
    % Here, we evaluate objective function in batches and accumulate its
    % value and gradient
    
    properties
        net         % description of DNN to be trained
        pRegTheta   % regularizer for network parameters
        pLoss       % loss function
        pRegW       % regularizer for classifier 
        Y           % features
        C           % labels
        batchSize   % batch size
        batchIds    % indices of batches
        useGPU      % flag for GPU computing
        precision   % flag for precision
    end
    
    methods
        function this = dnnBatchObjFctn(net,pRegTheta,pLoss,pRegW,Y,C,varargin)
            
            if nargout==0 && nargin==0
                this.runMinimalExample;
                return;
            end
            batchSize = 10;
            batchIds  = randperm(size(Y,2));
            useGPU    = [];
            precision = [];
            for k=1:2:length(varargin)     % overwrites default parameter
                eval([varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            
            this.net       = net;
            this.pRegTheta = pRegTheta;
            this.pLoss     = pLoss;
            this.pRegW     = pRegW;
            if not(isempty(useGPU))
                this.useGPU = useGPU;
            end
            
            if not(isempty(precision))
                this.precision=precision;
            end
            [Y,C] = gpuVar(this.useGPU,this.precision,Y,C);
            this.Y         = Y;
            this.C         = C;
            this.batchSize = batchSize;
            this.batchIds  = batchIds;
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
            dJth = 0.0; dJW = 0.0; Hth = []; HW = []; PC = [];
            
            
            nex = size(Y,2);
            nb  = nBatches(this,nex);
            
            [theta,W] = split(this,thetaW);
            this.batchIds  = randperm(size(Y,2));
                        
            % compute loss
            F = 0.0; hisLoss = [];
            for k=nb:-1:1
                idk = this.getBatchIds(k,nex);
                if nb>1
                    Yk  = Y(:,idk);
                    Ck  = C(:,idk);
                else
                    Yk = Y;
                    Ck = C;
                end
                
                if compGrad
                    [YNk,~,tmp]                  = apply(this.net,theta,Yk); % forward propagation
                    J = getJthetaOp(this.net,theta,Yk,tmp);
                    [Fk,hisLk,dWFk,d2WFk,dYF,d2YF] = getMisfit(this.pLoss,W,YNk,Ck);
                else
                    [YNk]        = apply(this.net,theta,Yk); % forward propagation
                    [Fk,hisLk]  = getMisfit(this.pLoss,W,YNk,Ck);
                end
                F    = F    + numel(idk)*Fk;
                hisLoss  = [hisLoss;hisLk];
                if compGrad
                    dthFk = J'*dYF;
                    dJth  = dJth + numel(idk)*dthFk;
                    dJW   = dJW  + numel(idk)*dWFk;
                    if compHess
                        HW   = HW   + d2WFk*numel(idk);
                    end
                end
            end
            F    = F/nex;
            Jc   = F;
            if compGrad
                dJth = dJth/nex;
                dJW  = dJW/nex;
                if compHess
                    HW   = HW*(1/nex);
                    Hthmv = @(x) J'*(d2YF*(J*x));
                    Hth   = LinearOperator(numel(theta),numel(theta),Hthmv,Hthmv);
                end
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
            
            dJ   = [dJth; dJW];
            
            if nargout>3
                H  = blkdiag(Hth, HW);
            end
            
            if nargout>4
                PCth = getThetaPC(this,d2YF,theta,Yk,tmp);
                PC = blkdiag(PCth,getPC(this.pRegW));
            end
            
        end
        

        
        function nb = nBatches(this,nex)
            if this.batchSize==Inf
                nb = 1;
            else
                nb =  ceil(nex/this.batchSize);
            end
        end
        
        function ids = getBatchIds(this,k,nex)
            if isempty(this.batchIds) || numel(this.batchIds) ~= nex
                fprintf('reshuffle\n')
                this.batchIds = randperm(nex);
            end
            ids = this.batchIds(1+(k-1)*this.batchSize:min(k*this.batchSize,nex));
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
            str = 'dnnBatchObj';
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
            
            nex    = 400; nf =2;
            
            blocks = cell(2,1);
            blocks{1} = NN({singleLayer(dense([2*nf nf]))});
            outTimes = ones(10,1);
            blocks{2} = ResNN(doubleLayer(dense([2*nf 2*nf]),dense([2*nf 2*nf])),10,.1,'outTimes',outTimes);
            net    = Meganet(blocks);
            nth = nTheta(net);
            theta  = randn(nth,1);
            
            Y = randn(nf,nex);
            C = zeros(nf,nex);
            C(1,Y(2,:)>Y(1,:).^2) = 1;
            C(2,Y(2,:)<=Y(1,:).^2) = 1;
            
            % validation data
            Yv = randn(nf,nex);
            Cv = zeros(nf,nex);
            Cv(1,Yv(2,:)>Yv(1,:).^2) = 1;
            Cv(2,Yv(2,:)<=Yv(1,:).^2) = 1;
            
            
            
            pLoss = softmaxLoss();
            W = vec(randn(2,nDataOut(net)+1));
            pRegW  = tikhonovReg(.01*speye(numel(W)));
            pRegTheta    = tikhonovReg(.01*speye(numel(theta)));
            
            f1 = dnnObjFctn(net,pRegTheta,pLoss,pRegW,Y,C);
            f2 = dnnBatchObjFctn(net,pRegTheta,pLoss,pRegW,Y,C);
            fv1 = dnnObjFctn(net,[],pLoss,[],Yv,Cv);
            fv2 = dnnBatchObjFctn(net,[],pLoss,[],Yv,Cv);
            
            % [Jc,para,dJ,H,PC] = fctn([Kb(:);W(:)]);
            % checkDerivative(fctn,[Kb(:);W(:)])
            
            opt =sd('out',1,'maxIter',20);
            [KbW1] = solve(opt,f1,[theta(:); W(:)],fv1);
            [KbW2] = solve(opt,f2,[theta(:); W(:)],fv1);
            norm(KbW1(:)-KbW2(:))
            
        end
    end
end










