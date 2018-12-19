classdef dnnVarProBatchObjFctn < objFctn
    % classdef dnnVarProBatchObjFctn < objFctn
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
        batchSize   % batch size
        batchIds    % indices of batches
        useGPU      % flag for GPU computing
        precision   % flag for precision
    end
    
    methods
        function this = dnnVarProBatchObjFctn(net,pRegTheta,pLoss,pRegW,optClass,Y,C,varargin)
            
            if nargout==0 && nargin==0
                this.runMinimalExample;
                return;
            end
            batchSize = 10;
            batchIds  = randperm(sizeLastDim(Y));
            useGPU    = [];
            precision = [];
            for k=1:2:length(varargin)     % overwrites default parameter
                eval([varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            
            this.net       = net;
            this.pRegTheta = pRegTheta;
            this.pLoss     = pLoss;
            this.pRegW     = pRegW;
            this.optClass  = optClass;
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
        
        function [Jc,para,dJ,H,PC] = eval(this,theta,idx)
            if not(exist('idx','var')) || isempty(idx)
                Y = this.Y;
                C = this.C;
            else
                Y = this.Y(:,idx);
                C = this.C(:,idx);
            end
            colons = repmat( {':'} , 1 , ndims(Y)-1 ); % variable-length colons for indexing Y
            
            compGrad = nargout>2;
            compHess = nargout>3;
            dJ = 0.0; H = []; PC = [];
            
            szY = size(Y);
            nex = szY(end);
            szYN  = [sizeFeatOut(this.net) nex];
            
%           % forward prop
            YN = zeros(szYN,'like',this.Y);
            for k=1:nb
                if nb>1
                    idk = this.getBatchIds(k,nex);
                    Yk  = Y( colons{:} , idk);
                else
                    Yk = Y;
                end
                YNk = forwardProp(this.net,theta,Yk);
                if nb>1
                    YN( colons{:} , idk ) = YNk;
                else
                    YN=YNk;
                end
            end
            %classify
            YN = reshape(YN,[],nex);
            fctn   = classObjFctn(this.pLoss,this.pRegW,YN,C);
            W      = solve(this.optClass,fctn,zeros(size(C,1)*(size(YN,1)+1),1,'like',theta));
            
            % compute loss
            F = 0.0; hisLoss = []; dJth = 0.0;
            for k=nb:-1:1
                idk = this.getBatchIds(k,nex);
                if nb>1
                    Yk  = Y( colons{:} , idk);
                    Ck  = C(:,idk);
                else
                    Yk = Y;
                    Ck = C;
                end
                
                nBatchEx = numel(idk);
                
                if compGrad
                    [YNk,J]                  = linearizeTheta(this.net,theta,Yk); % forward propagation
                    szYNk  = size(YNk);
                    YNk = reshape(YNk,[],nBatchEx);
                    [Fk,hisLk,~,~,dYF,d2YF] = getMisfit(this.pLoss,W,YNk,Ck);
                    dYF = reshape(dYF,szYNk);
                else
                    [YNk]        = forwardProp(this.net,theta,Yk); % forward propagation
                    YNk = reshape(YNk,[],nBatchEx);
                    [Fk,hisLk]   = getMisfit(this.pLoss,W,YNk,Ck);
                end
                F    = F    + numel(idk)*Fk;
                hisLoss  = [hisLoss;hisLk];
                if compGrad
                    dthFk = J'*dYF;
                    dJth  = dJth + numel(idk)*dthFk;
                    if compHess&&k==1
                        Hthmv   =@(x) (numel(idk)/nex)*(J'*(d2YF*(J*x)));
                        H   = LinearOperator(numel(theta),numel(theta),Hthmv,Hthmv); 
                    end
                end
            end
            F  = F/nex;
            Jc = F;
            if compGrad
                dJ = dJth/nex;
            end
            
            para = struct('F',F,'hisLoss',hisLoss,'W',W);
            
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
            if nargout>4
                PC = getPC(this.pRegTheta);
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
            str = 'dnnVarProBatch';
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
            blocks{2} = ResNN(doubleSymLayer(dense([2*nf 2*nf])),10,.1);
            net    = Meganet(blocks);
            nth    = nTheta(net);
            theta  = randn(nth,1);
            
            Y = randn(nf,nex);
            C = zeros(nf,nex);
            C(1,Y(2,:)>Y(1,:).^2) = 1;
            C(2,Y(2,:)<=Y(1,:).^2) = 1;
            
            pLoss = softmaxLoss();
            W = vec(randn(2,numelFeatOut(net)+1));
            pRegW        = tikhonovReg(opEye(numel(W)),1e-3);
            pRegTheta    = tikhonovReg(opEye(numel(theta)),1e-3);
            
            newtInner =newton('out',0,'maxIter',5);
            
            fctn = dnnVarProBatchObjFctn(net,pRegTheta,pLoss,pRegW,newtInner,Y,C);
            %[Jc,para,dJ,H,PC] = fctn([Kb(:);W(:)]);
            %  checkDerivative(fctn,Kb(:))
            newtOuter =newton('out',1,'maxIter',60);
            
            [KbWopt,his] = solve(newtOuter,fctn,theta(:));
        end
    end
end










