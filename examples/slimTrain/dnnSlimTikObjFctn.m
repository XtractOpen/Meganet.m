classdef dnnSlimTikObjFctn < handle
    
    properties
        net
        pLoss
        Y
        C
        WPrev                   % previous matrix W
        WHist
        pRegTheta
        alphaTh     = 1e-3      % regularization on theta
        alphaW
        alphaWHist
        
        WRef        = 0
        thetaRef    = 0
        updateW     = 1
        
        useGPU      = 0
        precision   = 'double'

        % slimTik properties
        M               = []
        iter            = 0
        memoryDepth     = 10
        tikParam        = tikParamFctn('allowNeg',1)
        sumLambda       = 0.0      % regularization on W (sum of Lambdak)
        Lambda                      % collect Lambdak
        Lambda0                     % initial sumLambda
    end
    
    
    methods
        
        function this = dnnSlimTikObjFctn(net,pRegTheta,pLoss,Y,C,varargin)
            
            for k = 1:2:length(varargin)     % overwrites default parameter
                this.(varargin{k}) = varargin{k + 1};
            end
            
            this.net        = net;
            this.pRegTheta  = pRegTheta;
            this.pLoss      = pLoss; 
            
            if isempty(this.Lambda0)
                this.Lambda0 = this.sumLambda;
            end
            
            
            [Y,C,thetaRef,WRef] = gpuVar(this.useGPU,this.precision,Y,C,this.thetaRef,this.WRef);
            this.Y          = Y;
            this.C          = C;
            this.thetaRef   = thetaRef;
            this.WRef       = WRef;
            
            if isempty(this.WPrev)
                this.WPrev = zeros(size(C,1),sizeFeatOut(net)+pLoss.addBias);
            end
            this.WHist = this.WPrev(:);
        end
        
        function theta = split(~,theta)
        end
        
        function [Jc,para,dJ] = eval(this,theta,idx)
            
            if not(exist('idx','var')) || isempty(idx)
                Y = this.Y;
                C = this.C;
                % error('Must use SA method')
            else
                colons = repmat({':'}, 1, ndims(this.Y)-1);
                Y = this.Y(colons{:},idx);
                C = this.C(:,idx);
            end

            compGrad  = (nargout > 2);
            
            nex       = sizeLastDim(Y);   % number of examples
            beta      = 1 / sqrt(nex);   % scaling
            % beta      = 1;
            
            % ----------------------------------------------------------- %
            % propagate through network
            if compGrad
                [Z,tmp] = forwardProp(this.net,theta,Y); % forward propagation
            else
                Z = forwardProp(this.net,theta,Y);
            end
            
            % add bias
            Zt = [Z;ones(this.pLoss.addBias,size(Z,2))];
            
            % ----------------------------------------------------------- %
            % slimTik
            if compGrad
                
                W = this.WPrev;

                % update memory matrix
                if this.iter > this.memoryDepth
                    % remove oldest stored matrix
                    this.M = this.M(:,size(Zt,2)+1:end);
                end
                this.M = [this.M,Zt];
                
                % find optimal Lambdak
                [Lambdak,W] = solve(this.tikParam,beta*Zt,beta*C,beta*this.M,W,this.sumLambda,this.Lambda0);
                
                % update
                this.WPrev     = W;
                this.WHist     = [this.WHist,W(:)];
                this.iter      = this.iter + 1;
                this.Lambda    = [this.Lambda,Lambdak];
                
                if strcmp(this.tikParam.optMethod,'none') 
                    this.alphaW = 0;
                elseif strcmp(this.tikParam.optMethod,'constant') 
                    this.alphaW    = this.Lambda0;
                    this.sumLambda = this.sumLambda + this.Lambda0;
                else
                    % this.sumLambda = this.sumLambda + Lambdak;
                    this.sumLambda = max(this.sumLambda + Lambdak,this.tikParam.lowerBound);
                    this.alphaW    = this.sumLambda / (this.iter + 1);
                end
                
                this.alphaWHist = [this.alphaWHist,this.alphaW];
            else
                W = this.WPrev;
            end
            
            % ----------------------------------------------------------- %
            % prediction
            Cpred   = W * Zt;
            resLoss = Cpred - C;
            
            % misfit
            F  = (0.5 / nex) * norm(resLoss(:))^2;
            Jc = F;
            
            % regularization on W
            resReg2 = sqrt(this.alphaW) * (vec(W) - vec(this.WRef));
            R2      = 0.5 * (resReg2(:)' * resReg2(:));
            
            Jc = Jc + R2;
            
            % gradient w.r.t. theta
            if compGrad
                J1Loss = getJthetaOp(this.net,theta,Y,tmp);
                dJth   = J1Loss' * (W(:,1:end-1)' * reshape(beta^2 * resLoss,size(C,1),[]));
            end
            
            % regularizer for theta
            if not(isempty(this.pRegTheta))
                [Rth,hisRth,dRth,~] = regularizer(this.pRegTheta,theta);
                Jc = Jc + Rth;
                if compGrad
                    dJth = dJth + dRth;
                end
            else
                resReg1 = sqrt(this.alphaTh)*(theta - this.thetaRef);
                Rth     = (1/2)*(resReg1(:)'*resReg1(:));
                Jc      = Jc + Rth;
                dRth    = sqrt(this.alphaTh) * resReg1;
                hisRth  = [Rth,this.alphaTh];
                if compGrad
                    dJth = dJth + dRth;
                end
                
            end
            if compGrad
                dJ = dJth;
            end
            
            Cp   = getLabels(this,Cpred);
            tmp  = nnz(C - Cp)/2;
            para = struct('F',F,'Rth',Rth,'R2',R2,'W',W,'resLoss',resLoss,'resAcc',(1-tmp/size(Cp,2))*100,'hisRth',hisRth,'alpha2',this.alphaW);
            
        end
        
        function [str,frmt] = hisNames(this)
            str = {'F','R(theta)','alphaTh','lambdak/iter','R(W)','alphaW','memDepth','size(M,2)','iter'};
            frmt = {'%-12.2e','%-12.2e','%-12.2e','%-12.2e','%-12.2e','%-12.2e','%-12d','%-12d','%-12d'};
        end
        
        function his = hisVals(this,para)
            
            if not(isempty(this.pRegTheta))
                hisReg = hisVals(this.pRegTheta,para.hisRth);
            else
                hisReg = [para.Rth,this.alphaTh];
            end
            
            batchSize = size(this.M,2) / (this.memoryDepth + 1);
            
            his = [para.F,hisReg, this.sumLambda / this.iter, para.R2, para.alpha2, this.memoryDepth, size(this.M,2)-batchSize,this.iter];
        end
        
        function str = objName(this)
            str = class(this);
        end
        
        function reset(this)
            this.M       = [];
            this.iter    = 1;
            this.sumLambda = 0.05;
            this.Lambda  = [];
        end
        
        % ------- functions for handling GPU computing and precision ---- %
        function this = set.useGPU(this,value)
            if isempty(value)
                return
            elseif(value~=0) && (value~=1)
                error('useGPU must be 0 or 1.')
            else
                if not(isempty(this.net)); this.net.useGPU = value; end
                
                [this.Y,this.C,this.WRef] = gpuVar(value,this.precision,...
                    this.Y,this.C,this.WRef);
            end
        end
        function this = set.precision(this,value)
            if isempty(value)
                return
            elseif not(strcmp(value,'single') || strcmp(value,'double'))
                error('precision must be single or double.')
            else
                if not(isempty(this.net)); this.net.precision = value; end
                
                [this.Y,this.C,this.WRef] = gpuVar(this.useGPU,value,...
                    this.Y,this.C,this.WRef);
            end
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