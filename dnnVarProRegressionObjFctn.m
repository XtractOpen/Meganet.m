classdef dnnVarProRegressionObjFctn < objFctn
    % classdef dnnGNObjFctnVarPro < objFctn
    %
    % Gauss-Newton-VarPro objective function for deep neural networks.
    %
    % J(theta,W) = loss(W*net(theta,Y), C)...
    %                      + alpha1*norm(theta-thetaRef)^2  ...
    %                      + alpha2*norm(W-WRef)^2
    %
    % Currently, only regressionLoss is supported.
    %
    
    properties
        net
        pLoss
        alpha1
        alpha2
        thetaRef
        WRef
        Y
        C
        matrixFree  % flag for matrix-free computation, default = 1
        useGPU
        precision
    end
    
    methods
        function this = dnnVarProRegressionObjFctn(net,pLoss,Y,C,varargin)
            if nargout==0 && nargin==0
                this.runMinimalExample;
                return;
            end
            useGPU = [];
            precision = [];
            matrixFree = 1;
            alpha1 = 1;
            alpha2 = 1;
            thetaRef = 0;
            WRef = zeros(size(C,1),numelFeatOut(net)+pLoss.addBias);
            for k=1:2:length(varargin)     % overwrites default parameter
                eval([varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            this.net    = net;
            this.pLoss  = pLoss;
            this.alpha1 = alpha1;
            this.alpha2 = alpha2;
            if not(isempty(useGPU))
                this.useGPU = useGPU;
            end
            
            if not(isempty(precision))
                this.precision=precision;
            end
            [Y,C,thetaRef,WRef] = gpuVar(this.useGPU,this.precision,Y,C,thetaRef,WRef);
            this.Y         = Y;
            this.C         = C;
            this.thetaRef = thetaRef;
            this.WRef = WRef;
            this.matrixFree = matrixFree;
        end
        function theta = split(this,theta)
        end
        
        function [Jc,para,dJ,H,PC,res,Jac] = eval(this,theta,idx)
            % [Jc,para,res,Jac,PC] = eval(this,theta,idx)
            if not(exist('idx','var')) || isempty(idx)
                Y = this.Y;
                C = this.C;
            else
                colons = repmat( {':'} , 1 , ndims(this.Y)-1 );
                Y = this.Y( colons{:} ,idx);
                C = this.C(:,idx);
            end
            compJac = nargout>2;
            nex = sizeLastDim(Y);   % number of examples to compute loss over
            J1 = [];   PC=[];
            % propagate through network
            if compJac
                [YN,tmp] = forwardProp(this.net,theta,Y); % forward propagation
            else
                YN = forwardProp(this.net,theta,Y);
            end
            
            % solve the least-squares problem using SVD
            YNt = sqrt(1/nex)*[YN;ones(this.pLoss.addBias,size(YN,2))];
            C   = sqrt(1/nex)*C;
            [U,S,V] = svd(YNt,'econ');
            s  = diag(S);
            s2 = 1./(s.^2+this.alpha2);
            W = (((C*V).*s'+ this.alpha2*this.WRef*U).*s2')*U';
            
            % compute regression loss
            Cpred = W*YNt;
            resLoss = (Cpred - C);
%             resLoss = (Cpred - this.C); % ADD ACCURACY OPTION HERE

    
            F   = (resLoss(:)'*resLoss(:))/2;
            Jc = F;
            res = vec(resLoss);
            
            % regularizer for theta1
            resReg1 = sqrt(this.alpha1)*(theta - this.thetaRef);
            R1 = (1/2)*(resReg1(:)'*resReg1(:));
            Jc = Jc + R1;
            res = [res; resReg1];
            
            % regularizer for theta2
            resReg2 = sqrt(this.alpha2)*(vec(W) - vec(this.WRef));
            R2 = (1/2)*(resReg2(:)'*resReg2(:));
            res = [res; resReg2];
            Jc = Jc + R2;
            
            
            if  compJac
                J1Loss  = getJthetaOp(this.net,theta,Y,tmp);
                
                nC  = numel(this.C);
                szCp = size(C);
                nth     = numel(theta);
                
                % build operator for theta1
                
                Jmv = @(x) Jresmv(this,x,resLoss,J1Loss,W,YN,U,V,s,s2);
                Jtmv = @(x) Jtresmv(this,x,resLoss,J1Loss,W,YN,U,V,s,s2);
                Jac = LinearOperator(nC+nth+numel(W),nth,Jmv,Jtmv);
                
                if ~this.matrixFree
                    J = zeros(nC+nth+numel(W),nth);
                    for i = 1:nth
                        tmp = zeros(nth,1);
                        tmp(i) = 1;
                        J(:,i) = Jmv(tmp);
                    end
                    Jac = J;
                end
                
                H = @(x) Jac' * (Jac * x);
                H = LinearOperator(Jac.n, Jac.n, H, H);
                dJ = Jac' * res(:);
            else
            end
            
            
            % para = struct('F',F,'R1',R1,'R2',R2,'W',W,'resLoss',resLoss);
            
            Cp = getLabels(this,Cpred);
            tmp = nnz(C-Cp)/2;
            para = struct('F',F,'R1',R1,'R2',R2,'W',W,'resLoss',resLoss,'resAcc',(1-tmp/size(Cp,2))*100);
        end
        
        function v = Jresmv(this,x,resLoss,J1Loss,W,Z,U,V,s,s2)
            % v = Jresmv(this,x,resLoss,J1Loss,W,YN,U,V,s,s2)
            % 
            % Inputs: 
            %   this    - dnn object
            %   x       - direction to update theta
            %   resLoss - residual of loss function
            %   J1loss  - Jacobian function handle for theta (not W(theta))
            %   W       - classification matrix from VarPro
            %   YN      - output from network; [YN; ones] = U * S * V^T
            %   s       - diag(S)
            %   s2      - 1 ./ (s.^2+this.alpha2);
            %
            % Outputs:
            %   v - directional derivative of theta in direction x 
            %
            %  Jmv = @(x) (Wt*(J1Loss*x))-((((Wt*(J1Loss*x)*YN'+resLoss*(J1Loss*x)')*U).*s2')*U')*YN;

           Wt = W(:,1:end-this.pLoss.addBias);
           nex = size(this.C,2);
           
           % network jacobian applied to x
           JthZ = sqrt(1/nex) * [J1Loss * x; zeros(this.pLoss.addBias,nex)];
           WJthZ = Wt * JthZ(1:size(Wt,2),:);
           
           % temporary variable for better-conditioned calculations
           JthW_noU = (WJthZ * (V .* (s .* s2)') + resLoss * (JthZ' * U) .* s2');
           
           JthW = JthW_noU * U';
           JthWZ = (JthW_noU .* s') * V';
           
           v = [reshape(WJthZ - JthWZ,[],size(x,2)); sqrt(this.alpha1)*x; -sqrt(this.alpha2)*reshape(JthW,[],size(x,2))];
        end
        
        function v = Jtresmv(this,x,resLoss,J1Loss,W,YN,U,V,s,s2)
            % Jtmv = @(x) J1Loss'*(Wt'*reshape(x,szCp)) - J1Loss'*(Wt'*(reshape(x,szCp)/YN)*YN + (reshape(x,szCp)/YN)'*resLoss);
            Wt = W(:,1:end-this.pLoss.addBias);
                
            nex = size(this.C,2);
            nTh = size(J1Loss,2);
            x1 = reshape(x(1:numel(this.C)),size(this.C));
            x2 = x(numel(this.C)+(1:nTh));
            x3 = sqrt(this.alpha2)*reshape(x(numel(x1)+numel(x2)+1:end),size(Wt,1),[]);
            
            t1 = W'*x1;
            t2 = (x1*V.*(s'.*s2'))*U';
            
            t3  = (((x3*U).*s2')*U');
            t41 =  W'*(((x3*U).*(s'.*s2'))*V');
            t42 = t3'*resLoss;
            t4 = t41 + t42;
            t5 = W'*(x1*V.*(s.^2'.*s2')*V');
            
            rhs = (t1  - t5 - t2'*resLoss-t4);
            rhs = rhs(1:end-this.pLoss.addBias,:);
            v1 = sqrt(1/nex)*J1Loss'*rhs;
            
            v2 = sqrt(this.alpha1)*x2;
            
            v = v1+v2;
        end
        function [str,frmt] = hisNames(this)
            str = {'F',};
            frmt = {'%-12.2e'};
            
            if not(this.alpha1==0)
                str  = [str, 'alpha1','R(theta1)'];
                frmt = [frmt, '%-12.2e','%-12.2e'];
            end
            if not(this.alpha2==0)
                str  = [str, 'alpha2','R(theta2)'];
                frmt = [frmt, '%-12.2e','%-12.2e'];
            end
        end
        
        function his = hisVals(this,para)
            % his = para.F;
            his = [para.F];
            if not(this.alpha1==0);
                his = [his, this.alpha1, para.R1];
            end
            if not(this.alpha2==0);
                his = [his, this.alpha2, para.R2];
            end
        end
        
        function str = objName(this)
            str = 'dnnGNObjFctnVarPro';
        end
        
        % ------- functions for handling GPU computing and precision ----
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
        
        
        
        function runMinimalExample(~)
            nex    = 400; nf =2; nw = 16;
            
            net = singleLayer(dense([nw nf]),'Bin',eye(nw));
            Y = randn(nf,nex);
            C = zeros(nf,nex);
            C(1,Y(2,:)>Y(1,:).^2) = 1;
            C(2,Y(2,:)<=Y(1,:).^2) = 1;
            
            pLoss  = regressionLoss('addBias',1);
            alpha  = [1e-4 2];
            
            theta  = initTheta(net);
            
            fctnGN = dnnGNObjFctnVarPro(net,pLoss,Y,C,'matrixFree',0,'alpha1',alpha(1),'alpha2',alpha(2));
            [JcGN,paraGN,res,J] = eval(fctnGN,theta);
            
            %% check derivative
            dtheta = randn(size(theta));
            dJ = dtheta'*(J'*res);
            for k=1:30
                h = 2^(-k);
                tht = theta + h*dtheta;
                Jt  = eval(fctnGN,tht);
                
                E0 = abs(JcGN - Jt);
                E1 = abs(JcGN + h*dJ - Jt);
                fprintf('h=%1.2e\tE0=%1.2e\tE1=%1.2e\n',h,E0,E1);
            end
        end
    end
end










