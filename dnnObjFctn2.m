classdef dnnObjFctn2 < objFctn
    % classdef dnnObjFctn2 < objFctn
    %
    % objective function for deep neural networks 
    %
    % J(theta) = loss(Y(theta), C) + Rtheta(theta);
    %
    % This function is similar to dnnObjFctn but does not handle the
    % weights of the last layer of the network differently than the others.
    
    properties
        net
        pReg
        pLoss
        Y
        C
        matrixFree  % flag for matrix-free computation, default = 1
        gnHessian   % flag for Gauss-Newton approximation of Hessian
        useGPU
        precision
    end
    
    methods
        function this = dnnObjFctn2(net,pReg,pLoss,Y,C,varargin)
            
            if nargout==0 && nargin==0
                this.runMinimalExample;
                return;
            end
            useGPU = [];
            precision = [];
            matrixFree = 1;
            gnHessian  = 1;
            for k=1:2:length(varargin)     % overwrites default parameter
                eval([varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            this.net    = net;
            this.pReg = pReg;
            this.pLoss  = pLoss;
            
            if not(isempty(useGPU))
                this.useGPU = useGPU;
            end
            if not(isempty(precision))
                this.precision=precision;
            end
            
            if not(numelFeatOut(net) == size(C,1))
                error('numelFeatOut(net)=%d != %d=size(C,1)',numelFeatOut(net),size(C,1));
            end
            
            [Y,C] = gpuVar(this.useGPU,this.precision,Y,C);
            this.Y         = Y;
            this.C         = C;
            this.matrixFree = matrixFree;
            this.gnHessian = gnHessian;
            
            
        end
        function [theta] = split(this,theta)
        end
        
        function [Jc,para,dJ,H,PC] = eval(this,theta,idx)
            if not(exist('idx','var')) || isempty(idx)
                Y = this.Y;
                C = this.C;
            else
                colons = repmat( {':'} , 1 , ndims(this.Y)-1 );
                Y = this.Y( colons{:} ,idx);
                C = this.C(:,idx);
            end
            compGrad = nargout>2;
            compHess = nargout>3;
            nex = sizeLastDim(Y);   % number of examples to compute loss over

            dJ = [];  H = []; PC = [];
            
            % evaluate loss
            %                 [YN,Yall,dA]                   = fwd(this.net,Kb,this.Y);
            if compGrad || compHess
                [YN,tmp] = forwardProp(this.net,theta,Y); % forward propagation
            else
                YN = forwardProp(this.net,theta,Y);
            end
            
            szYN  = size(YN);
            YN = reshape(YN,[],nex); % loss expects 2D input
            
            % evaluate loss function
            [F,para,dF,d2F] = eval(this.pLoss,YN,C);
            Jc = F;
            
            if compGrad && (this.matrixFree || not(compHess))
                dF  = JthetaTmv(this.net,dF,theta,Y,tmp);
                dJ = dF;
            elseif compGrad
                Jac = getJacobians(this.net,theta,Y,tmp);
                dF  = Jac'*dF(:);
                dJ = dF;
            end
            
            if compHess
                if not(this.gnHessian)
                    error('nyi');
                end
                
                if this.matrixFree
                    Hmv = @(x) JthetaTmv(this.net,reshape(d2F*Jthetamv(this.net,x,theta,Y,tmp),szYN),theta,Y,tmp); %  JTmv(this.net, reshape(d2YF* Jmv(this.net,x,[],Kb,Yall,dA),size(YN)), Kb,Yall,dA);
                    H   = LinearOperator(numel(theta),numel(theta),Hmv,Hmv);
                else
                    H = Jac' * d2F * Jac;
                end
                
            end
            para = struct('F',F);

            % evaluate regularizer for DNN weights             
            if not(isempty(this.pReg))
                [R,hisR,dR,d2R]      = regularizer(this.pReg,theta);
                Jc = Jc + R;
                if compGrad
                    dJ = dJ + dR;
                end
                if compHess
                       H  = H + d2R;
                end
                para.R = R;
                para.hisR = hisR;
            end
            
            if nargout>4
                PC = opEye(numel(theta));
            end
        end
        
        function [str,frmt] = hisNames(this)
            [str,frmt] = hisNames(this.pLoss);
            str = {'loss'};
            frmt = {'%-12.2e'};
            if not(isempty(this.pReg))
                [s,f] = hisNames(this.pReg);
                s{1} = [s{1} '(theta)'];
                str  = [str, s{:}];
                frmt = [frmt, f{:}];
            end
            
        end
        
        function his = hisVals(this,para)
            his = para.F;
            if not(isempty(this.pReg))
                his = [his, hisVals(this.pReg,para.hisR)];
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
                if not(isempty(this.pReg)); this.pReg.useGPU       = value; end
                
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
                if not(isempty(this.pReg)); this.pReg.precision       = value; end
                
                [this.Y,this.C] = gpuVar(this.useGPU,value,...
                                                         this.Y,this.C);
            end
        end
        function useGPU = get.useGPU(this)
                useGPU = -ones(2,1);
                
                if not(isempty(this.net)) && not(isempty(this.net.useGPU))
                    useGPU(1) = this.net.useGPU;
                end
                if not(isempty(this.pReg)) && not(isempty(this.pReg.useGPU))
                    useGPU(2) = this.pReg.useGPU;
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
            isSingle    = -ones(2,1);
            isSingle(1) = strcmp(this.net.precision,'single');
            if not(isempty(this.pReg)) && not(isempty(this.pReg.precision))
                isSingle(2) = strcmp(this.pReg.precision,'single');
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
            
            blocks    = cell(0,1);
            blocks{end+1} = NN({singleLayer(dense([2*nf nf]))});
%             blocks{end+1} = ResNN(doubleLayer(dense([2*nf 2*nf]),dense([2*nf 2*nf])),2,.1);
            blocks{end+1} = NN({singleLayer(dense([2,2*nf]),'activation',@identityActivation,'Bin',ones(2,1))});
            
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
            pReg    = tikhonovReg(eye(numel(theta)));
            
            fctn = dnnObjFctn2(net,pReg,pLoss,Y,C,'matrixFree',0);
            fval = dnnObjFctn2(net,[],pLoss,Yv,Cv);
            % [Jc,para,dJ,H,PC] = fctn([Kb(:);W(:)]);
            % checkDerivative(fctn,[Kb(:);W(:)])
            
            opt1  = newton('out',1,'maxIter',20);
            opt2  = sd('out',1,'maxIter',20);
            opt3  = nlcg('out',1,'maxIter',20);
            [KbWopt1,His1] = solve(opt1,fctn,theta(:),fval);
            [KbWopt2,His2] = solve(opt2,fctn,theta(:),fval);
            [KbWopt3,His3] = solve(opt3,fctn,theta(:),fval);
            
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










