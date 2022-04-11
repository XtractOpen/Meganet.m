classdef ResNNrk4 < abstractMeganetElement
    % ****** get rid of ttheta and test a case that's not identity or
    % square
    %
    % Residual Neural Network with 4th order Runge Kutta
    %
    %   Y_k+1 = Y_k-1 + h/6*Z1 + h/3*Z2 + h/3*Z3 + h/6*Z4
    %
    % where 
    %  
    %       Z1 = L(Y_k, Theta(t_k; theta))  
    %       Z2 = L(Y_k+h/2*Z1, Theta(t_k+h/2))
    %       Z3 = L(Y_k+h/2*Z2, Theta(t_k+h/2))
    %       Z4 = L(Y_k+h*Z3, Theta(t_k+h))
    %
    %  Here, L(Y,theta)=layer(trafo(theta),Y).
    %
    %  The time points for the states and controls can be spaced
    %  non-uniformly and are described by the nodes of the 1D grids ttheta
    %  and tY.
    %
    %  Theta(t; theta) = sum_{i=1}^n p_i(t) * theta_i
    % 
    % with 
    %  - theta_i = coefficients of the weight function Theta,
    %  length(theta_i) = nTheta(layer) 
    %  - p_i = basis functions
    % 
    % Example:
    % 
    % Columns of Theta (or Columns of A): basis fntns evaluated at timept
    % t_k
    %
    % 1     2     3 4 5   6   7  8  9 ...        = Column #
    % |-----o-----|-o-|---o---|--o--|-o-|---o---|
    % tY(0)                                   tY(end)
    %
    % tY represent nodes of the 1-D grid
    % o are cell center: tY(0) + tY(1) / 2
    
    
    
    
    properties
        layer        % model for the layer (i.e., the nonlinearity)
        tY           % time points for states, Y (tY is a nodal grid in 1d)
        useGPU
        precision
        A            % A_ij = p_i(t_j), t_j = h/2 x j (odd entries in A belong to nodes, even ones are cell centers)
                     % t_j = tY((j+1)/2), for j odd
                     % t_j = [tY(j/2) + tY(j/2 + 1)] * 1\2, for j even
    end
    
    methods
        function this = ResNNrk4(layer,tY,varargin) % tY could be non-equidistant
            if nargin==0 && nargout==0
                this.runMinimalExample;
                return;
            end
            useGPU = [];
            precision = [];
            nt = numel(tY);
            A = speye(2*nt - 1);
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
            if any(sizeFeatOut(layer)~=sizeFeatIn(layer))
                error('%s - dim. of input and output features must agree for ResNet layers',mfilename);
            end
            this.A      = A;
            this.tY     = tY;
        end
        
        
        function n = nTheta(this)
            n = size(this.A,1)*nTheta(this.layer);
        end
%         function n = nTheta(this)
%             n = numel(this.ttheta)*nTheta(this.layer);
%         end
        function n = sizeFeatIn(this)
            n = sizeFeatIn(this.layer);
        end
        function n = sizeFeatOut(this)
            n = sizeFeatOut(this.layer);
        end
        
        function theta = initTheta(this) % initilizaing coefficients of polynomial so the function is 
            Theta = repmat(vec(initTheta(this.layer)),1,size(this.A,2));
            theta = Theta/this.A;
        end
        
        
        function [net2,theta2] = prolongateWeights(this,theta) 
            % add new nodes at cell-centers. initialize weights using
            % linear interpolation. 
            
            tOld = this.ttheta;
            nc   = numel(tOld)-1;  % number of cells
            tNew = linspace(tOld(1),tOld(end),2*nc+1);
            
            theta = reshape(theta,[],numel(tOld));
            
            theta2 = inter1D(theta,tOld,tNew);
            net2 = this;
            net2.ttheta = tNew;
            
        end
        
        %% ------- forwardProp forward propagation -----------
        function [Y,tmp] = forwardProp(this,theta,Y,varargin)
            doDerivative = (nargout>1);
            for k=1:2:length(varargin)     % overwrites default parameter
                eval([varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            
            nt = numel(this.tY);
            if nargout>1;    tmp = cell(nt,9); tmp{1,1} = Y; end
            
            theta = reshape(theta,nTheta(this.layer),[]); 
            Theta = theta*this.A;  % column-wise contains weights we will need for time integration
            
            %theta = reshape(theta,[],numel(this.ttheta));
            
     %% now the rk4 steps
            for i=1:nt-1  
               
                hi = this.tY(i+1)-this.tY(i);  
               
                % 1)   Z1 = L(Y_k, theta(t_k))
                thetai = Theta(:,2*i-1); 
                [A1,tmp{i,6}] = forwardProp(this.layer,thetai,Y,'doDerivative',doDerivative); % removed thetai, changed thetai in () to theta
                Z = Y + (hi/2)*A1;
                if nargout>1, tmp{i,2} = Z; end
    
                % 2)  Z2 = L(Y_k+h/2*Z1, theta(t_k+h/2))
                thetai = Theta(:,2*i);
                [A2,tmp{i,7}] = forwardProp(this.layer,thetai,Z,'doDerivative',doDerivative);
                Z = Y + (hi/2)*A2;
                if nargout>1, tmp{i,3} = Z; end
    
                % 3)  Z3 = L(Y_k+h/2*Z2, theta(t_k+h/2))
                [A3,tmp{i,8}] = forwardProp(this.layer,thetai,Z,'doDerivative',doDerivative);
                Z = Y + hi*A3;
                if nargout>1, tmp{i,4} = Z; end
    
                
                % 4)  Z4 = L(Y_k+h*Z3, theta(t_k+h))
                thetai = Theta(:,2*i + 1);
                [A4,tmp{i,9}] = forwardProp(this.layer,thetai,Z,'doDerivative',doDerivative);
                
                % Y_k+1 = Y_k-1 + h/6*Z1 + h/3*Z2 + h/3*Z3 + h/6*Z4
                Y =  Y + hi*((1/6)*A1 + (1/3)*A2 + (1/3)*A3 + (1/6)*A4);
                if nargout>1, tmp{i+1,1} = Y; end
            end
        end
        
        function this = setTimeY(this,theta,Y,AbsTol,RelTol)
            % set time steps for states
            if not(exist('AbsTol','var')) || isempty(AbsTol)
                AbsTol = 1e-1;
            end
            if not(exist('RelTol','var')) || isempty(RelTol)
                RelTol = 1e-1;
            end
            odeOptsFwd = odeset('AbsTol',AbsTol,'RelTol',RelTol,'Stats','off');
            net1   = ResNNod(this.layer,this.tY(end),this.ttheta,'odeOptsFwd',odeOptsFwd,'odeSolverFwd',@ode113,'useGPU',0);
            [YN,tmp] = forwardProp(net1,gather(theta),gather(Y));
            this.tY = tmp{1};
            fprintf('ResNNrk4: Found %d new time steps: [%s]\n',numel(this.tY),num2str(reshape(this.tY,1,[])));
        end
        %% -------- Jacobian matvecs ---------------
        function dY = JYmv(this,dY,theta,~,tmp)
            if isempty(dY)
                dY = 0.0;
            end
            theta  = reshape(theta,[],size(this.A,1));
            
            theta = reshape(theta,nTheta(this.layer),[]); 
            Theta = theta*this.A; 
            
            nt = numel(this.tY);
            for i=1:nt-1
                hi = this.tY(i+1)-this.tY(i);
                
                % 1)   Z1 = L(Y_k, theta(t_k))
                thetai = Theta(:,2*i-1);
                dA1    = JYmv(this.layer,dY,thetai,tmp{i,1},tmp{i,6});
                
                
                % 2)  Z2 = L(Y_k+h/2*Z1, theta(t_k+h/2))
                thetai = Theta(:,2*i);
                dA2    = JYmv(this.layer,dY + (hi/2)*dA1,thetai,tmp{i,2},tmp{i,7});
                
                % 3)  Z3 = L(Y_k+h/2*Z2, theta(t_k+h/2))
                dA3    = JYmv(this.layer,dY + (hi/2)*dA2,thetai,tmp{i,3},tmp{i,8});
                
                % 4)  Z4 = L(Y_k+h*Z3, theta(t_k+h))
                thetai = Theta(:,2*i+1);
                dA4    = JYmv(this.layer,dY + hi * dA3,thetai,tmp{i,4},tmp{i,9});
                
                % Y_k+1 = Y_k-1 + h/6*Z1 + h/3*Z2 + h/3*Z3 + h/6*Z4
                dY =  dY + hi*((1/6)*dA1 + (1/3)*dA2 + (1/3)*dA3 + (1/6)*dA4);
            end
        end
        
        
        function dY = Jmv(this,dtheta,dY,theta,~,tmp)
            if isempty(dY)
                dY = 0.0;
            end
            
            theta  = reshape(theta,[],size(this.A,1));
            dtheta = reshape(dtheta,[],size(this.A,1));
            
            Theta = theta*this.A;
            dTheta = dtheta*this.A; 
            
            nt = numel(this.tY);
            for i=1:nt-1
                hi = this.tY(i+1)-this.tY(i);
                
                % 1)   Z1 = L(Y_k, theta(t_k))
                thetai  = Theta(:,2*i-1);
                dthetai = dTheta(:,2*i-1); % is this correct?
                dA1     = Jmv(this.layer,dthetai,dY,thetai,tmp{i,1},tmp{i,6});
                
                % 2)  Z2 = L(Y_k+h/2*Z1, theta(t_k+h/2))
                thetai  = Theta(:,2*i);
                dthetai = dTheta(:,2*i);
                dA2     = Jmv(this.layer,dthetai,dY + (hi/2)* dA1,thetai,tmp{i,2},tmp{i,7});
                
                % 3)  Z3 = L(Y_k+h/2*Z2, theta(t_k+h/2))
                dA3    = Jmv(this.layer,dthetai,dY + (hi/2)*dA2,thetai,tmp{i,3},tmp{i,8});
                
                % 4)  Z4 = L(Y_k+h*Z3, theta(t_k+h))
                thetai  = Theta(:,2*i+1);
                dthetai = dTheta(:,2*i+1);
                dA4     = Jmv(this.layer,dthetai,dY + hi*dA3,thetai,tmp{i,4},tmp{i,9});
                
                % Y_k+1 = Y_k-1 + h/6*Z1 + h/3*Z2 + h/3*Z3 + h/6*Z4
                dY =  dY + hi*((1/6)*dA1 + (1/3)*dA2 + (1/3)*dA3 + (1/6)*dA4);
            end
        end
        
        % -------- Jacobian' matvecs ----------------
        
        function W = JYTmv(this,W,theta,Y,tmp)
            if isempty(W)
                W = 0;
            end
            theta  = reshape(theta,[],size(this.A,1));
            Theta = theta*this.A;  
            
            nt = numel(this.tY);  
            for i=nt-1:-1:1
                ti = this.tY(i); hi = this.tY(i+1)-this.tY(i);
                
                % 4) Z4 = L(Y_k+h*Z3, theta(t_k+h))
                thetai = Theta(:,2*i+1); % backward
                dW4 = JYTmv(this.layer,(1/6)*W,thetai,tmp{i,4},tmp{i,9}); % (d Z4) / (d_Y3)
                
                % 3) 
                Wtt = (1/3)*W + hi*dW4;
                thetai = Theta(:,2*i);
                dW3 = JYTmv(this.layer,Wtt,thetai,tmp{i,3},tmp{i,8});
                
                % 2) 
                Wtt = (1/3)*W + (hi/2)*dW3;
                dW2 = JYTmv(this.layer,Wtt,thetai,tmp{i,2},tmp{i,7});
                
                % 1) 
                Wtt = (1/6)*W + (hi/2)*dW2;
                thetai = Theta(:,2*i-1);
                dW1 = JYTmv(this.layer,Wtt,thetai,tmp{i,1},tmp{i,6});
                
                W =  W + hi*(dW1 + dW2 + dW3 + dW4);
            end
            
        end
        
        function [dtheta,W] = JTmv(this,W,theta,Y,tmp,doDerivative)
            if not(exist('doDerivative','var')) || isempty(doDerivative)
               doDerivative =[1;0]; 
            end
            if isempty(W)
                W = 0;
            end
            
            theta  = reshape(theta,[],size(this.A,1));
            dtheta = 0*theta;  
            Theta = theta*this.A;  
            dTheta = 0*Theta;
            nt = numel(this.tY);
            
            for i=nt-1:-1:1
                    
                hi = this.tY(i+1)-this.tY(i);
                
                % 4) Z4 = L(Y_k+h*Z3, theta(t_k+h))
                thetai = Theta(:,2*i+1); 
                [dth4,dW4] = JTmv(this.layer,(1/6)*W,thetai,tmp{i,4},tmp{i,9}); % (d Z4) / (d_Y3)
                dTheta(:,2*i+1) = hi*dth4 + dTheta(:,2*i+1) ; % gradients on the nodes            
                
                
                
                % 3) 
                Wtt = (1/3)*W + hi*dW4;
                thetai = Theta(:,2*i);
                [dth3,dW3] = JTmv(this.layer,Wtt,thetai,tmp{i,3},tmp{i,8});
                dTheta(:,2*i) = hi*dth3 + dTheta(:,2*i) ;
                % dtheta(:,2*i) = dtheta(:,idi(1)) + wi(1)*hi*dth3;
                
                
                % 2) 
                Wtt = (1/3)*W + (hi/2)*dW3;
                [dth2,dW2] = JTmv(this.layer,Wtt,thetai,tmp{i,2},tmp{i,7});
                dTheta(:,2*i) = dTheta(:,2*i) +  hi*dth2;
                
                
                % 1) 
                Wtt = (1/6)*W + (hi/2)*dW2;
                thetai = Theta(:,2*i-1);
                [dth1,dW1] = JTmv(this.layer,Wtt,thetai,tmp{i,1},tmp{i,6});
                dTheta(:,2*i-1) = dTheta(:,2*i-1) + hi*dth1;
                               
                W =  W + hi*(dW1 + dW2 + dW3 + dW4);
            end
             dtheta = vec(dTheta*this.A');
            if nargout==1 && all(doDerivative==1)
                dtheta=[dtheta(:); W(:)];
            end

        end
        %% ------- functions for handling GPU computing and precision ----
        
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
            [Ydata,tmp]   = net.forwardProp(mb,Y0);
            dmb = reshape(randn(size(mb)),[],numel(net.ttheta));
            dY0  = 10*randn(size(Y0));
            
            dY = net.Jmv(dmb(:),dY0,mb,Y0,tmp);
            for k=1:10
                hh = 10^(-k);
                
                Yt = net.forwardProp(mb+hh*dmb(:),Y0+hh*dY0);
                
                E0 = norm(Yt(:)-Ydata(:));
                E1 = norm(Yt(:)-Ydata(:)-hh*dY(:));
                
                fprintf('h=%1.2e\tE0=%1.2e\tE1=%1.2e\n',hh,E0,E1);
            end
            
            W = randn(size(Ydata));
            t1  = W(:)'*dY(:);
            
             [dWdmb,dWY] = net.JTmv(W,mb,Y0,tmp);
             t2 = dmb(:)'*dWdmb(:) + dY0(:)'*dWY(:);
%             dWY = net.JYTmv(W,mb,Y0,tmp);
%             t2 = dY0(:)'*dWY(:);
            
            fprintf('adjoint test: t1=%1.2e\tt2=%1.2e\terr=%1.2e\n',t1,t2,abs(t1-t2));
        end
    end
    
end

