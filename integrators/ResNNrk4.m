classdef ResNNrk4 < abstractMeganetElement
    % Residual Neural Network with 4th order Runge Kutta
    %
    %   Y_k+1 = Y_k-1 + h/6*Z1 + h/3*Z2 + h/3*Z3 + h/6*Z4
    %
    % where 
    %  
    %       Z1 = L(Y_k, theta(t_k))
    %       Z2 = L(Y_k+h/2*Z1, theta(t_k+h/2))
    %       Z3 = L(Y_k+h/2*Z2, theta(t_k+h/2))
    %       Z4 = L(Y_k+h*Z3, theta(t_k+h))
    %
    %  Here, L(Y,theta)=layer(trafo(theta),Y).
    %
    %  The time points for the states and controls can be spaced
    %  non-uniformly and are described by the nodes of the 1D grids ttheta
    %  and tY.
    
    properties
        layer        % model for the layer (i.e., the nonlinearity)
        tY           % time points for states, Y
        ttheta       % time points for controls, theta
        outTimes     % time points for measurements
        Q            % measurement matrix
        useGPU
        precision
    end
    
    methods
        function this = ResNNrk4(layer,tY,ttheta,varargin)
            if nargin==0 && nargout==0
                this.runMinimalExample;
                return;
            end
            useGPU = [];
            precision = [];
            outTimes = [];
            Q = 1.0;
            for k=1:2:length(varargin)     % overwrites default parameter
                eval([varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            if not(isempty(useGPU))
                layer.useGPU = useGPU;
            end
            if not(isempty(precision))
                layer.precision = precision;
            end
            this.layer = layer;
            if nFeatOut(layer)~=nFeatIn(layer)
                error('%s - dim. of input and output features must agree for ResNet layers',mfilename);
            end
            if not(exist('tY','var')) || isempty(tY)
                tY = ttheta;
            end
            if norm(tY(1)-ttheta(1))/norm(tY(1)) > 1e-13
                error('initial time points for states and controls must match')
            end
            if norm(tY(end)-ttheta(end))/norm(tY(end)) > 1e-13
                error('final time points for states and controls must match')
            end
            this.tY     = tY;
            this.ttheta = ttheta;
            if isempty(outTimes)
                outTimes = zeros(numel(tY)-1,1); 
                outTimes(end) = 1;
            end
            this.outTimes = outTimes;
            this.Q = Q;
        end
        
        function n = nTheta(this)
            n = numel(this.ttheta)*nTheta(this.layer);
        end
        function n = nFeatIn(this)
            n = nFeatIn(this.layer);
        end
        function n = nFeatOut(this)
            n = nFeatOut(this.layer);
        end
        
        function n = nDataOut(this)
            if numel(this.Q)==1
                n = nnz(this.outTimes)*nFeatOut(this.layer);
            else
                n = nnz(this.outTimes)*size(this.Q,1);
            end
        end

        function theta = initTheta(this)
            theta = repmat(vec(initTheta(this.layer)),numel(this.ttheta),1);
        end
        
        % ------- apply forward propagation -----------
        function [Ydata,Y,tmp] = apply(this,theta,Y0)
            nex = numel(Y0)/nFeatIn(this);
            Y   = reshape(Y0,[],nex);
            nt = numel(this.tY);
            if nargout>1;    tmp = cell(nt,9); tmp{1,1} = Y0; end
            
            theta = reshape(theta,[],numel(this.ttheta));
            Ydata = [];
            % now the rk4 steps
            for i=1:nt-1
                ti = this.tY(i); hi = this.tY(i+1)-this.tY(i);
                
                % 1)   Z1 = L(Y_k, theta(t_k))
                thetai = inter1D(theta,this.ttheta,ti);
                [A1,~,tmp{i,6}] = apply(this.layer,thetai,Y);
                Z = Y + (hi/2)*A1;
                if nargout>1, tmp{i,2} = Z; end
    
                % 2)  Z2 = L(Y_k+h/2*Z1, theta(t_k+h/2))
                thetai = inter1D(theta,this.ttheta,ti+hi/2);
                [A2,~,tmp{i,7}] = apply(this.layer,thetai,Z);
                Z = Y + (hi/2)*A2;
                if nargout>1, tmp{i,3} = Z; end
    
                % 3)  Z3 = L(Y_k+h/2*Z2, theta(t_k+h/2))
                [A3,~,tmp{i,8}] = apply(this.layer,thetai,Z);
                Z = Y + hi*A3;
                if nargout>1, tmp{i,4} = Z; end
    
                
                % 4)  Z4 = L(Y_k+h*Z3, theta(t_k+h))
                thetai = inter1D(theta,this.ttheta,ti+hi);
                [A4,~,tmp{i,9}] = apply(this.layer,thetai,Z);
                
                % Y_k+1 = Y_k-1 + h/6*Z1 + h/3*Z2 + h/3*Z3 + h/6*Z4
                Y =  Y + hi*((1/6)*A1 + (1/3)*A2 + (1/3)*A3 + (1/6)*A4);
                if this.outTimes(i)==1
                    Ydata = [Ydata; this.Q*Y];
                end
                if nargout>1, tmp{i+1,1} = Y; end
            end
        end
        
        % -------- Jacobian matvecs ---------------
        function [dYdata,dY] = JYmv(this,dY,theta,~,tmp)
            if isempty(dY)
                dY = 0.0;
            elseif numel(dY)>1
                nex = numel(dY)/nFeatIn(this);
                dY   = reshape(dY,[],nex);
            end
            theta  = reshape(theta,[],numel(this.ttheta));
            
            dYdata = [];
            
            nt = numel(this.tY);
            for i=1:nt-1
                ti = this.tY(i); hi = this.tY(i+1)-this.tY(i);
                
                % 1)   Z1 = L(Y_k, theta(t_k))
                thetai = inter1D(theta,this.ttheta,ti);
                dA1    = JYmv(this.layer,dY,thetai,tmp{i,1},tmp{i,6});
                
                
                % 2)  Z2 = L(Y_k+h/2*Z1, theta(t_k+h/2))
                thetai = inter1D(theta,this.ttheta,ti+hi/2);
                dA2    = JYmv(this.layer,dY + (hi/2)*dA1,thetai,tmp{i,2},tmp{i,7});
                
                % 3)  Z3 = L(Y_k+h/2*Z2, theta(t_k+h/2))
                dA3    = JYmv(this.layer,dY + (hi/2)*dA2,thetai,tmp{i,3},tmp{i,8});
                
                % 4)  Z4 = L(Y_k+h*Z3, theta(t_k+h))
                thetai = inter1D(theta,this.ttheta,ti+hi);
                dA4    = JYmv(this.layer,dY + hi * dA3,thetai,tmp{i,4},tmp{i,9});
                
                % Y_k+1 = Y_k-1 + h/6*Z1 + h/3*Z2 + h/3*Z3 + h/6*Z4
                dY =  dY + hi*((1/6)*dA1 + (1/3)*dA2 + (1/3)*dA3 + (1/6)*dA4);
                if this.outTimes(i)==1
                    dYdata = [dYdata; this.Q*dY];
                end
            end
        end
        
        
        function [dYdata,dY] = Jmv(this,dtheta,dY,theta,~,tmp)
            if isempty(dY)
                dY = 0.0;
            elseif numel(dY)>1
                nex = numel(dY)/nFeatIn(this);
                dY   = reshape(dY,[],nex);
            end
            theta  = reshape(theta,[],numel(this.ttheta));
            dtheta = reshape(dtheta,[],numel(this.ttheta));
            dYdata  = [];
            
            nt = numel(this.tY);
            for i=1:nt-1
                ti = this.tY(i); hi = this.tY(i+1)-this.tY(i);
                
                % 1)   Z1 = L(Y_k, theta(t_k))
                thetai  = inter1D(theta,this.ttheta,ti);
                dthetai = inter1D(dtheta,this.ttheta,ti);
                dA1    = Jmv(this.layer,dthetai,dY,thetai,tmp{i,1},tmp{i,6});
                
                % 2)  Z2 = L(Y_k+h/2*Z1, theta(t_k+h/2))
                thetai  = inter1D(theta,this.ttheta,ti+hi/2);
                dthetai = inter1D(dtheta,this.ttheta,ti+hi/2);
                dA2     = Jmv(this.layer,dthetai,dY + (hi/2)* dA1,thetai,tmp{i,2},tmp{i,7});
                
                % 3)  Z3 = L(Y_k+h/2*Z2, theta(t_k+h/2))
                dA3    = Jmv(this.layer,dthetai,dY + (hi/2)*dA2,thetai,tmp{i,3},tmp{i,8});
                
                % 4)  Z4 = L(Y_k+h*Z3, theta(t_k+h))
                thetai  = inter1D(theta,this.ttheta,ti+hi);
                dthetai = inter1D(dtheta,this.ttheta,ti+hi);
                dA4     = Jmv(this.layer,dthetai,dY + hi*dA3,thetai,tmp{i,4},tmp{i,9});
                
                % Y_k+1 = Y_k-1 + h/6*Z1 + h/3*Z2 + h/3*Z3 + h/6*Z4
                dY =  dY + hi*((1/6)*dA1 + (1/3)*dA2 + (1/3)*dA3 + (1/6)*dA4);
                if this.outTimes(i)==1
                    dYdata = [dYdata; this.Q*dY];
                end
            end
        end
        
        % -------- Jacobian' matvecs ----------------
        
        function W = JYTmv(this,Wdata,W,theta,Y,tmp)
            nex = numel(Y)/nFeatIn(this);
            if ~isempty(Wdata)
                Wdata = reshape(Wdata,[],nnz(this.outTimes),nex);
            end
            if isempty(W)
                W = 0;
            else
                W     = reshape(W,[],nex);
            end
            theta  = reshape(theta,[],numel(this.ttheta));
            
            nt = numel(this.tY);  cnt = nnz(this.outTimes);
            for i=nt-1:-1:1
                if this.outTimes(i)==1
                    W = W + squeeze(Wdata(:,cnt,:)); 
                    cnt = cnt-1;
                end
                ti = this.tY(i); hi = this.tY(i+1)-this.tY(i);
                
                % 4) Z4 = L(Y_k+h*Z3, theta(t_k+h))
                thetai = inter1D(theta,this.ttheta,ti+hi);
                dW4 = JYTmv(this.layer,(1/6)*W,[],thetai,tmp{i,4},tmp{i,9}); % (d Z4) / (d_Y3)
                
                % 3) 
                Wtt = (1/3)*W + hi*dW4;
                thetai = inter1D(theta,this.ttheta,ti+hi/2);
                dW3 = JYTmv(this.layer,Wtt,[],thetai,tmp{i,3},tmp{i,8});
                
                % 2) 
                Wtt = (1/3)*W + (hi/2)*dW3;
                dW2 = JYTmv(this.layer,Wtt,[],thetai,tmp{i,2},tmp{i,7});
                
                % 1) 
                Wtt = (1/6)*W + (hi/2)*dW2;
                thetai = inter1D(theta,this.ttheta,ti);
                dW1 = JYTmv(this.layer,Wtt,[],thetai,tmp{i,1},tmp{i,6});
                
                W =  W + hi*(dW1 + dW2 + dW3 + dW4);
            end
            
        end
        
        function [dtheta,W] = JTmv(this,Wdata,W,theta,Y,tmp,doDerivative)
            if not(exist('doDerivative','var')) || isempty(doDerivative)
               doDerivative =[1;0]; 
            end
            nex = numel(Y)/nFeatIn(this);
            if ~isempty(Wdata)
                Wdata = reshape(Wdata,[],nnz(this.outTimes),nex);
            end
            if isempty(W)
                W = 0;
            elseif numel(W)>1
                W     = reshape(W,[],nex);
            end
            
            theta  = reshape(theta,[],numel(this.ttheta));
            dtheta = 0*theta;
            
            nt = numel(this.tY); cnt = nnz(this.outTimes);
            for i=nt-1:-1:1
                if this.outTimes(i)==1
                    W = W + this.Q'*squeeze(Wdata(:,cnt,:));
                    cnt = cnt-1;
                end
                    
                ti = this.tY(i); hi = this.tY(i+1)-this.tY(i);
                
                % 4) Z4 = L(Y_k+h*Z3, theta(t_k+h))
                [thetai,wi,idi] = inter1D(theta,this.ttheta,ti+hi);
                [dth4,dW4] = JTmv(this.layer,(1/6)*W,[],thetai,tmp{i,4},tmp{i,9}); % (d Z4) / (d_Y3)
                dtheta(:,idi(1)) = dtheta(:,idi(1)) + wi(1)*hi*dth4;
                dtheta(:,idi(2)) = dtheta(:,idi(2)) + wi(2)*hi*dth4;
                
                
                % 3) 
                Wtt = (1/3)*W + hi*dW4;
                [thetai,wi,idi] = inter1D(theta,this.ttheta,ti+hi/2);
                [dth3,dW3] = JTmv(this.layer,Wtt,[],thetai,tmp{i,3},tmp{i,8});
                dtheta(:,idi(1)) = dtheta(:,idi(1)) + wi(1)*hi*dth3;
                dtheta(:,idi(2)) = dtheta(:,idi(2)) + wi(2)*hi*dth3;
                
                % 2) 
                Wtt = (1/3)*W + (hi/2)*dW3;
                [dth2,dW2] = JTmv(this.layer,Wtt,[],thetai,tmp{i,2},tmp{i,7});
                dtheta(:,idi(1)) = dtheta(:,idi(1)) + wi(1)*hi*dth2;
                dtheta(:,idi(2)) = dtheta(:,idi(2)) + wi(2)*hi*dth2;
                
                
                % 1) 
                Wtt = (1/6)*W + (hi/2)*dW2;
                [thetai,wi,idi] = inter1D(theta,this.ttheta,ti);
                [dth1,dW1] = JTmv(this.layer,Wtt,[],thetai,tmp{i,1},tmp{i,6});
                dtheta(:,idi(1)) = dtheta(:,idi(1)) + wi(1)*hi*dth1;
                dtheta(:,idi(2)) = dtheta(:,idi(2)) + wi(2)*hi*dth1;
                
                W =  W + hi*(dW1 + dW2 + dW3 + dW4);
            end
             dtheta = vec(dtheta);
            if nargout==1 && all(doDerivative==1)
                dtheta=[dtheta(:); W(:)];
            end

        end
        % ------- functions for handling GPU computing and precision ----
        function this = set.useGPU(this,value)
            if (value~=0) && (value~=1)
                error('useGPU must be 0 or 1.')
            else
                this.layer.useGPU  = value;
            end
        end
        function this = set.precision(this,value)
            if not(strcmp(value,'single') || strcmp(value,'double'))
                error('precision must be single or double.')
            else
                this.layer.precision = value;
            end
        end
        function useGPU = get.useGPU(this)
            useGPU = this.layer.useGPU;
        end
        function precision = get.precision(this)
            precision = this.layer.precision;
        end
        
        
        function runMinimalExample(~)
            nex = 10;
            nK  = [4 4];
            
            D   = dense(nK);
            S   = singleLayer(D);
            hth = rand(10,1); hth = hth/sum(hth);
            tth = [0;cumsum(hth)];
            hY  = rand(100,1); hY  = hY/sum(hY);
            tY = [0;cumsum(hY)];
            
            net = ResNNrk4(S,tth,tY);
            mb  = randn(nTheta(net),1);
            
            Y0  = randn(nK(2),nex);
            [Ydata,~,tmp]   = net.apply(mb,Y0);
            dmb = reshape(randn(size(mb)),[],numel(net.ttheta));
            dY0  = 10*randn(size(Y0));
            
            dY = net.Jmv(dmb(:),dY0,mb,Y0,tmp);
            for k=1:10
                hh = 10^(-k);
                
                Yt = net.apply(mb+hh*dmb(:),Y0+hh*dY0);
                
                E0 = norm(Yt(:)-Ydata(:));
                E1 = norm(Yt(:)-Ydata(:)-hh*dY(:));
                
                fprintf('h=%1.2e\tE0=%1.2e\tE1=%1.2e\n',hh,E0,E1);
            end
            
            W = randn(size(Ydata));
            t1  = W(:)'*dY(:);
            
             [dWdmb,dWY] = net.JTmv(W,[],mb,Y0,tmp);
             t2 = dmb(:)'*dWdmb(:) + dY0(:)'*dWY(:);
%             dWY = net.JYTmv(W,mb,Y0,tmp);
%             t2 = dY0(:)'*dWY(:);
            
            fprintf('adjoint test: t1=%1.2e\tt2=%1.2e\terr=%1.2e\n',t1,t2,abs(t1-t2));
        end
    end
    
end

