classdef dnnBatchObjFctn < objFctn
    % classdef dnnBatchObjFctn < objFctn
    %
    % objective function for deep neural networks
    %
    %       J(theta,C) = loss(h(W*Y(theta)), C) + Rtheta(theta) + R(W)
    %
    % Here, we group the terms of the loss function in batches and accumulate its
    % value and gradient. This helps reduce the memory footprint of the
    % method but might be slower than working on the full batch implemented
    % in dnnObjFctn. Splitting the loss into batches should not affect the
    % result as long as the network does not introduce any coupling between different
    % examples (note that batch normalization does that).
    
    properties
        net         % description of DNN to be trained
        pRegTheta   % regularizer for network parameters
        pLoss       % loss function
        pRegW       % regularizer for classifier
        Y           % input features
        C           % labels
        batchSize   % batch size, default=10 (choose according to available memory,
        % choice might be different from the batch size of SGD)
        batchIds    % indices of batches, chosen randomly in each evaulation
        useGPU      % flag for GPU computing
        precision   % flag for precision
        dataAugment
    end
    
    methods
        function this = dnnBatchObjFctn(net,pRegTheta,pLoss,pRegW,Y,C,varargin)
            % constructor, required inputs are a network, regularizers,
            % loss and examples.
            if nargout==0 && nargin==0
                this.runMinimalExample;
                return;
            end
            batchSize = 10;
            batchIds  = randperm(sizeLastDim(Y));
            useGPU    = [];
            precision = [];
            dataAugment = @(x) x;
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
            this.dataAugment = dataAugment;
        end
        function [theta,W] = split(this,thetaW)
            % split thetaW = [theta;W] into parts associated with network
            % and classifier.
            nth = nTheta(this.net);
            theta  = thetaW(1:nth);
            W   = thetaW(nth+1:end);
        end
        
        
        function [Jc,para,dJ,H,PC] = eval(this,thetaW,idx)
            % evaluate the objective function
            %
            % Inputs:
            %
            %   thetaW - vector of current weights (both network and classifier)
            %   idx    - indices of examples to use for evaluating loss,
            %            default=[] -> use all examples
            %
            % Output:
            %
            %   Jc     - objective function value
            %   para   - struct containing intermediate results for
            %            printing (see hisNames and hisVals for info and parsing)
            %   dJ     - gradient
            %   H      - approximate Hessian (Gauss Newton like
            %            approximation for each block, ignore coupling between
            %            theta and W)
            %   PC     - preconditioner
            
            
            
            if not(exist('idx','var')) || isempty(idx)
                % use all examples
                Y = this.Y;
                C = this.C;
            else
                % use only examples specified in idx in the loss
                colons = repmat( {':'} , 1 , ndims(this.Y)-1 );
                Y = this.Y( colons{:} ,idx);
                C = this.C(:,idx);
            end
             Y = this.dataAugment(Y);
            
            compGrad = nargout>2;
            compHess = nargout>3;
            dJth = 0.0; dJW = 0.0; Hth = []; HW = []; PC = [];
            
            nex = sizeLastDim(Y);   % number of examples to compute loss over
            nb  = nBatches(this,nex); % determine number of batches for the computation
            
            [theta,W] = split(this,thetaW);
            this.batchIds  = randperm(nex);
            
            % compute loss
            F = 0.0; hisLoss = [];
            for k=nb:-1:1
                idk = this.getBatchIds(k,nex);
                if nb>1
                    colons = repmat( {':'} , 1 , ndims(Y)-1 );
                    Yk  = Y( colons{:} , idk);
                    Ck  = C(:,idk);
                else
                    Yk = Y;
                    Ck = C;
                end
                
                nBatchEx = sizeLastDim(Yk); % last batch may not be full
                
                if compGrad
                    [YNk,tmp]          = forwardProp(this.net,theta,Yk); % forward propagation
                    J = getJthetaOp(this.net,theta,Yk,tmp);
                    szYNk  = size(YNk);
                    YNk = reshape(YNk,[],nBatchEx); % loss expects 2D input
                    [Fk,hisLk,dWFk,d2WFk,dYF,d2YF] = getMisfit(this.pLoss,W,YNk,Ck);
                    dYF = reshape(dYF,szYNk);
                else
                    [YNk]        = forwardProp(this.net,theta,Yk); % forward propagation
                    YNk = reshape(YNk,[],nBatchEx);
                    [Fk,hisLk]  = getMisfit(this.pLoss,W,YNk,Ck);
                end
                F    = F    + numel(idk)*Fk;
                hisLoss  = [hisLoss;hisLk];
                if compGrad
                    dthFk = J'*dYF; 
                    dJth  = dJth + numel(idk)*dthFk;
                    dJW   = dJW  + numel(idk)*dWFk;
                    if compHess
                        if isempty(HW)
                            HW = d2WFk*numel(idk);
                        else
                            HW   = HW   + d2WFk*numel(idk);
                        end
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
            
            % evaluate regularizer for classification weights
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
            
            % get a preconditioner
            if nargout>4
                PC = blkdiag(opEye(numel(theta)),opEye(numel(W)));
            end
            
        end
        
        function [Cp,P] = getLabels(this,thetaW)
            % compute the predicted labels and class probabilities for
            % current weights
            %
            % Inputs:
            %
            %   thetaW - current weights
            %
            % Output:
            %
            %   Cp     - nclass x nex matrix of unit vectors that encode
            %            predicted class for each example
            %   P      - nclass x nex matrix whose columns are the class
            %            probabilities predicted by the model
            
            Cp = 0*this.C;
            P  = 0*this.C;
            nex = sizeLastDim(this.Y);
            nb = nBatches(this,nex);
            
            [theta,W] = split(this,thetaW);
            
            for k=nb:-1:1
                idk = this.getBatchIds(k,nex);
                if nb>1
                    colons = repmat( {':'} , 1 , ndims(this.Y)-1 );
                    Yk  = this.Y( colons{:} , idk);
                else
                    Yk = this.Y;
                end
                [YNk]     = forwardProp(this.net,theta,Yk); % forward propagation
                [Cp(:,idk),P(:,idk)] = getLabels(this.pLoss,W,YNk);
            end
        end
        
        function nb = nBatches(this,nex)
            % determine how many batches we need to split the examples into
            % to evaluate the loss
            if this.batchSize==Inf
                nb = 1;
            else
                nb =  ceil(nex/this.batchSize);
            end
        end
        
        function ids = getBatchIds(this,k,nex)
            % get the ids for the current term in the sum
            if isempty(this.batchIds) || numel(this.batchIds) ~= nex
                fprintf('numel(this.batchIds)=%d, nex=%d, reshuffle\n',numel(this.batchIds),nex)
                this.batchIds = randperm(nex);
            end
            ids = this.batchIds(1+(k-1)*this.batchSize:min(k*this.batchSize,nex));
        end
        
        function [str,frmt] = hisNames(this)
            % provides cell arrays for labels of history valies and
            % formatting.
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
            % take para from output eval and parse it so that intermediate
            % results can be printed by the optimizer
            his = hisVals(this.pLoss,sum(para.hisLoss,1));
            if not(isempty(this.pRegTheta))
                his = [his, hisVals(this.pRegTheta,para.hisRth)];
            end
            if not(isempty(this.pRegW))
                his = [his, hisVals(this.pRegW,para.hisRW)];
            end
        end
        
        function str = objName(this)
            % name of this objective function
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
            blocks{2} = ResNN(doubleLayer(dense([2*nf 2*nf]),dense([2*nf 2*nf])),10,.1);
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
            W = vec(randn(2,numelFeatOut(net)+1));
            pRegW  = tikhonovReg(.01*speye(numel(W)));
            pRegTheta    = tikhonovReg(.01*speye(numel(theta)));
            
            f1 = dnnObjFctn(net,pRegTheta,pLoss,pRegW,Y,C);
            f2 = dnnBatchObjFctn(net,pRegTheta,pLoss,pRegW,Y,C);
            fv1 = dnnObjFctn(net,[],pLoss,[],Yv,Cv);
            fv2 = dnnBatchObjFctn(net,[],pLoss,[],Yv,Cv);
            
            [Jc,para,dJ,H,PC] = f2.eval([theta(:);W(:)],[]);
            checkDerivative(f2,[theta(:);W(:)])
            
            opt =sd('out',1,'maxIter',20);
            [KbW1] = solve(opt,f1,[theta(:); W(:)],fv1);
            [KbW2] = solve(opt,f2,[theta(:); W(:)],fv1);
            norm(KbW1(:)-KbW2(:))
            
        end
    end
end