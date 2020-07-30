classdef dnnVarProObjFctn < objFctn
    % classdef dnnObjFctn < objFctn
    %
    % objective function for deep neural networks 
    %
    % J(theta,W) = loss(h(W*Y(theta)), C) + Rtheta(theta) + R(W),
    %
    % Adding Tikonov regularization
    
    properties
        net
        pRegTheta
        pLoss
        pRegW
        Y
        C
        optClass
        linSol
        useGPU
        precision
        WPrev % store previous W
        
    end
    
    methods
        function this = dnnVarProObjFctn(net,pRegTheta,pLoss,pRegW,optClass,Y,C,varargin)
            
            if nargout==0 && nargin==0
                this.runMinimalExample;
                return;
            end
            useGPU = [];
            precision = [];
            linSol=[];
            for k=1:2:length(varargin)     % overwrites default parameter
                eval([varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            this.net        = net;
            this.pRegTheta  = pRegTheta;
            this.pLoss      = pLoss;
            this.pRegW      = pRegW;
            this.optClass   = optClass;
            
            if not(isempty(useGPU))
                this.useGPU = useGPU;
            end
            
            if not(isempty(precision))
                this.precision=precision;
            end
            [Y,C] = gpuVar(this.useGPU,this.precision,Y,C);
            this.Y         = Y;
            this.C         = C;
            this.linSol=linSol;
            
            
            if ~isa(pRegW,'tikhonovReg') && ~isempty(pRegW)
                warning('Changing W regularizer to tikhonovReg');
                regOpW      = opEye((prod(sizeFeatOut(net))+1)*size(C,1));
                this.pRegW  = tikhonovReg(regOpW,1e-10);
            end
            
        end
        
        function [theta,W] = split(this,thetaW)
            if isprop(this.net,'layers') % if have last layer as classification layer
                nth = 0;
                for i = 1:length(this.net.layers)-1
                    nth = nth + nTheta(this.net.layers{i});
                end
                theta = thetaW(1:nth);
                W = thetaW(nth+1:end);
            else
                nth = nTheta(this.net);
                theta  = thetaW(1:nth);
                W   = thetaW(nth+1:end);
            end
        end
        
        function [Jc,para,dJ,H,PC,JWZwrtTh,WZ] = eval(this,theta,idx)
            if not(exist('idx','var')) || isempty(idx)
                Y = this.Y;
                C = this.C;
            else
                colons = repmat( {':'} , 1 , ndims(this.Y)-1 );
                Y = this.Y(colons{:} ,idx);
                C = this.C(:,idx);
            end
            
            compGrad = (nargout > 2);
            compHess = (nargout > 3);
            
            dJ = 0;     % gradient w.r.t. theta
            H  = [];    % Hessian w.r.t. theta (Gauss-Newton approximation)
            PC = [];    % preconditioner

            % forward propagate
            if compGrad || compHess
                [Z,JNet] = linearizeTheta(this.net,theta,Y);
            else
                Z = forwardProp(this.net,theta,Y);
            end
            
            % reshape output
            nex = sizeLastDim(Z);   % number of examples to compute loss over
            szZ = size(Z);
            Z   = reshape(Z,[],nex); % loss expects 2D input
            szZv = size(Z);
            
            % solve for W
            if isempty(this.WPrev)
                this.WPrev = zeros(size(C,1)*(size(Z,1)+this.pLoss.addBias),1);
            end
            
            [Zd,Cd] = gpuVar(0,'double',Z,C); % solve classification problem in double precision on CPU
            fctn          = classObjFctn(this.pLoss,this.pRegW,Zd,Cd);     
            [W,classHis,HBar,Vm1]  = solve(this.optClass,fctn,this.WPrev);
%             if classHis.optBreak.flag~=0
%                 pRegW = this.pRegW;
%                 pLoss = this.pLoss;
%                save(sprintf('classProb-%s.mat',datestr(now,'mm-dd-yy-hh:MM:SS')), 'Zd','Cd','pRegW','pLoss')
%             end
            this.WPrev    = W;
            [W,HBar,Vm1] = gpuVar(this.useGPU,this.precision,W,HBar,Vm1); 
            para.classHis = classHis;
            para.nrmW     = norm(W(:));
            
            % get misfit
            [F,hisLoss,dWF,d2WF,dYF,d2YF,dWZF,d2WZF] = getMisfit(this.pLoss,W,Z,C);
            Jc = F;
            para.F = F;
            para.hisLoss = hisLoss;
            para.W = W;

            % - - - - - - - - - - - - - - - - - - - - - - - - - - - - %  
            % evaluate regularizer for prediction matrix W  
            % assume tikonov regularization
            if not(isempty(this.pRegW)) 
                if ~isa(this.pRegW,'tikhonovReg')
                    error('Must use Tikhonov regularization on W');
                end
                d2WF = d2WF + this.pRegW.alpha * eye(size(d2WF));
                
                para.RW = (this.pRegW.alpha / 2) * norm(W(:))^2;
                para.hisRW = [para.RW,this.pRegW.alpha];
                
                % update objective function value
                Jc = Jc + para.RW;
            end
            
            % compute Jacobian and gradient
            if compGrad
                % some helpful variables
                addBias = this.pLoss.addBias;
                WOpt    = reshape(W,size(C,1),[]);
                WZ      = WOpt * [Z; ones(addBias,nex)];

                szW     = size(WOpt);
                szWZ    = size(WZ);
                
                % - - - - - - - - - - - - - - - - - - - - - - - - - - - - %
                % form original jacobians
                JWZwrtZmv = @(x) WOpt * [reshape(x,szZv); zeros(addBias,nex)];
                JWZwrtZTmv = @(x) WOpt(:,1:end-addBias)' * reshape(x,szWZ);

                % form jacobians for W
                JWZwrtWmv = @(x) reshape(x,szW) * [Z; ones(addBias,nex)];
                JWZwrtWTmv = @(x) reshape(x,szWZ) * [Z; ones(addBias,nex)]';

                JgradWwrtZmv = @(x) d2WZF(JWZwrtZmv(x)) * [Z; ones(addBias,nex)]' + dWZF * [reshape(x,szZv); zeros(addBias,nex)]';
                JgradWwrtZTmv = @(x) JWZwrtZTmv(d2WZF(x * [Z; ones(addBias,nex)])) + x(:,1:end-addBias)' * dWZF;

                if isnumeric(d2WF)
                    JWwrtZmv = @(x) reshape(d2WF \ vec(-JgradWwrtZmv(x)),szW);
                    JWwrtZTmv = @(x) -JgradWwrtZTmv(reshape(d2WF \ x(:),szW));
                elseif isempty(this.linSol)
                    [UU,Sig,VV] = svd(HBar);
                    UU = Vm1*UU(:,1:end-1);
                    VV = Vm1(:,1:end-1)*VV;
                    sig = diag(Sig);
                    
                    JWwrtZmv = @(x) -reshape(VV*((UU'*vec(JgradWwrtZmv(x)))./sig),szW);
                    JWwrtZTmv = @(x) -JgradWwrtZTmv(reshape(UU*((VV'*x(:))./sig),szW));
                else
                    JWwrtZmv = @(x) reshape(solve(this.linSol,d2WF,-vec(JgradWwrtZmv(x)),[],[]),szW);
                    JWwrtZTmv = @(x) -JgradWwrtZTmv(reshape(solve(this.linSol,d2WF, x(:),[],[]),szW));
                end
                
                % new varpro
                JWZwrtWZmv = @(x) JWZwrtZmv(x) + JWZwrtWmv(JWwrtZmv(x));
                JWZwrtWZTmv = @(x) JWZwrtZTmv(x) + JWwrtZTmv(JWZwrtWTmv(x));
                
                % jacobian of S
                JWZwrtThmv = @(x) JWZwrtWZmv(JNet * x);
                JWZwrtThTmv = @(x) JNet' * reshape(JWZwrtWZTmv(x),szZ);
                JWZwrtTh = LinearOperator(prod(szWZ),numel(theta),JWZwrtThmv,JWZwrtThTmv);
                
%                 % add jacobian of W w.r.t. theta
%                 JWwrtThmv = @(x) JWwrtZmv(JNet * x);
%                 JWwrtThTmv = @(x) JNet' * JWwrtZTmv(reshape(x,szW));
%                 JWwrtTh = LinearOperator(numel(W),numel(theta),JWwrtThmv,JWwrtThTmv);
                
                % - - - - - - - - - - - - - - - - - - - - - - - - - - - - % 
                % compute gradient
                dJ = JWZwrtTh' * dWZF(:);
                if not(isempty(this.pRegW))
                   dJ = dJ + this.pRegW.alpha * (JNet' * reshape(JWwrtZTmv(W(:)),szZ));
                end
               
                % Gauss-Newton Hessian approximation - is this correct?
                if compHess
                    HThmv = @(x) JWZwrtTh' * (d2WZF(JWZwrtTh * x)); 
                    H = LinearOperator(numel(theta),numel(theta),HThmv,HThmv);
                end

            end

            % - - - - - - - - - - - - - - - - - - - - - - - - - - - - %  
            % evaluate regularizer for DNN weights           
            if not(isempty(this.pRegTheta))
                [Rth,hisRth,dRth,d2Rth] = regularizer(this.pRegTheta,theta);

                % update objective function value
                Jc = Jc + Rth;

                % update gradient w.r.t. theta
                if compGrad
                    dJ = dJ + dRth;
                end

                % update hessian w.r.t. theta
                if compHess
                    H  = H + d2Rth;
                end

                para.Rth = Rth;
                para.hisRth = hisRth;
            end
            
        end
        
        function [str,frmt] = hisNames(this)
            [str,frmt] = hisNames(this.pLoss);
            
            % new formatting for optimizing W stats
            strW  = {'|dW|/|dW0|','|dW|','flag','WSolveTime'};
            frmtW = {'%-12.2e','%-12.2e','%-12d','%-12.4f'};
            str = [str,strW];
            frmt = [frmt,frmtW];
            
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
            
            % new values for optimizing W stats
            his = [his,para.classHis.optBreak.hisIter(end,4),para.classHis.optBreak.hisIter(end,5)];
            
            if isfield(para.classHis,'optBreak')
                his = [his,para.classHis.optBreak.flag];
            else
                his = [his,0];
            end
            
            % add end time
            if isfield(para.classHis,'endTime')
                his = [his,para.classHis.endTime];
            else
                his = [his,0];
            end
            
            if not(isempty(this.pRegTheta))
                his = [his, hisVals(this.pRegTheta,para.hisRth)];
            end
            if not(isempty(this.pRegW))
                his = [his, hisVals(this.pRegW,para.hisRW)];
            end
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
    
    methods (Static)
        
        function[H2,H3] = compareHessiansTh(JNet,d2YF,H)
            
            % only works using the old varpro part of jacobian
            
            nTh = H.m;

            H2 = zeros(nTh,1);  % old calcluation
            H3 = zeros(nTh,1);  % new calculation
            for i = 1:nTh
                tmp = zeros(nTh,1);
                tmp(i) = 1;
                H2(:,i) = JNet' * (d2YF * (JNet * tmp));
                H3(:,i) = H * tmp;
            end
            
            
        end
        
        
        function[] = plotHessSpectrum(d2WF)
            
            
            % symmetrize
            d = eig(0.5 * (d2WF + d2WF'));
            
            fig = figure(1);
            fig.Name = 'd2WF Spectrum';
            
            semilogy(1:length(d),d,'o');
            
        end
        
    end
end










