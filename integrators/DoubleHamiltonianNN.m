classdef DoubleHamiltonianNN < abstractMeganetElement
    % Double Layer Hamiltonian block
    %
    % Z_k+1 = Z_k - h*layer1(Y_k, theta_1),  
    % Y_k+1 = Y_k + h*layer2(Z_k+1,theta_2) 
    %
    % The input features are divided into Y and Z here based on the sizes 
    % of the layers. The layers do not have to have the same size.
    %
    % References:
    %
    % Chang B, Meng L, Haber E, Ruthotto L, Begert D, Holtham E: 
    %      Reversible Architectures for Arbitrarily Deep Residual Neural Networks, 
    %      AAAI Conference on Artificial Intelligence 2018
    
    properties
        layer1
        layer2
        nt
        h
        outTimes
        Q
        useGPU
        precision
    end
    
    methods
        function this = DoubleHamiltonianNN(layer1,layer2,nt,h,varargin)
            if nargin==0
                this.runMinimalExample;
                return;
            end
            useGPU    = [];
            precision = [];
            outTimes  = zeros(nt,1); outTimes(end)=1;
            Q = 1.0;
            for k=1:2:length(varargin)     % overwrites default parameter
                eval([varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            if not(isempty(useGPU))
                layer1.useGPU = useGPU;
                layer2.useGPU = useGPU;
            end
            if not(isempty(precision))
                layer1.precision = precision;
                layer2.precision = precision;
            end
            if any(sizeFeatOut(layer1)~=sizeFeatIn(layer2)) || any(sizeFeatIn(layer1)~=sizeFeatOut(layer2))
                error('number of input and output features must agree');
            end
            this.layer1 = layer1;
            this.layer2   = layer2;
            this.nt       = nt;
            this.h        = h;
            this.outTimes = outTimes;
            this.Q        = Q;
        end
        
        function n = nTheta(this)
            n = this.nt*(nTheta(this.layer1)+ nTheta(this.layer2));
        end
        function n = sizeFeatIn(this)
            n = sizeFeatIn(this.layer1)+sizeFeatIn(this.layer2);
        end
        function n = sizeFeatOut(this)
            n = sizeFeatIn(this);
        end

        function n = nDataOut(this)
           if numel(this.Q)==1
               n = nnz(this.outTimes)*numelFeatOut(this);
           else
               n = nnz(this.outTimes)*size(this.Q,1);
           end
        end
        
        function theta = initTheta(this)
            theta = repmat([vec(initTheta(this.layer1));...
                            vec(initTheta(this.layer2))],this.nt,1);
        end
        
        function [net2,theta2] = prolongateWeights(this,theta)
            % piecewise linear interpolation of network weights 
            t1 = 0:this.h:(this.nt-1)*this.h;
            
            net2 = DoubleHamiltonianNN(this.layer1,this.layer2,2*this.nt,this.h/2,'useGPU',this.useGPU,'Q',this.Q,'precision',this.precision);
            net2.outTimes = (sum(this.outTimes)>0)*net2.outTimes;
          
            t2 = 0:net2.h:(net2.nt-1)*net2.h;
            
            theta2 = inter1D(theta,t1,t2);
        end
        
        function [th1,th2] = split(this,x)
           x   = reshape(x,[],this.nt);
           th1 = x(1:nTheta(this.layer1),:);
           th2 = x(nTheta(this.layer1)+1:end,:);
        end
        function [Y,Z] = splitData(this,X)
           nex = numel(X)/numelFeatIn(this);
           X   = reshape(X,[],nex);
           Y   = X(1:prod(sizeFeatIn(this.layer1)),:);
           Z   = X(prod(sizeFeatIn(this.layer1))+1:end,:);
        end
        % ------- forwardProp forward problems -----------
        function [Xdata,X,tmp] = forwardProp(this,theta,X0)
            
            [Y,Z] = splitData(this,X0);
            if nargout>1;    tmp = cell(this.nt,4);  end
            [th1,th2] = split(this,theta);
            Xdata = zeros(0,size(Y,2),'like',Y);
            for i=1:this.nt
                if nargout>1, tmp{i,1} = Y; tmp{i,2} = Z; end
                
                [dZ,~,tmp{i,3}] = forwardProp(this.layer1,th1(:,i),Y);
                Z = Z - this.h*dZ;
                if nargout>1;   tmp{i,2} = Z; end
                
                [dY,~,tmp{i,4}] = forwardProp(this.layer2,th2(:,i),Z);
                Y = Y + this.h*dY;
                
                if this.outTimes(i)==1
                    Xdata = [Xdata; this.Q*[Y;Z]];
                end
            end
            X = [Y;Z];
        end
        
        % -------- Jacobian matvecs ---------------
        function [dXdata,dX] = JYmv(this,dX,theta,Y,tmp)
            nex = numel(Y)/numelFeatIn(this);
            if isempty(dX) || (numel(dX)==0 && dX==0.0)
                dX     = zeros([sizeFeatOut(this),nex],'like',Y);
                dXdata = zeros([nDataOut(this),nex],'like',Y);
                return
            end
            
            [dY,dZ]   = splitData(this,dX);
            [th1,th2] = split(this,theta);
            dXdata = zeros(0,nex,'like',Y);
            for i=1:this.nt
                
                dZ = dZ - this.h*JYmv(this.layer1,dY,th1(:,i),tmp{i,1},tmp{i,3});
                
                dY = dY + this.h*JYmv(this.layer2,dZ,th2(:,i),tmp{i,2},tmp{i,4});
                
                if this.outTimes(i)==1
                    dXdata = [dXdata; this.Q*[dY;dZ]];
                end
            end
        end
        
        function [dXdata,dX] = Jmv(this,dtheta,dX,theta,~,tmp)
            if isempty(dX)
                dY = 0.0;
                dZ = 0.0;
            elseif numel(dX)>1
                [dY,dZ] = splitData(this,dX);
            end
            [th1,th2]   = split(this,theta);
            [dth1,dth2] = split(this,dtheta);
            dXdata = [];
            for i=1:this.nt
                 dZ = dZ - this.h*Jmv(this.layer1,dth1(:,i),dY,th1(:,i),tmp{i,1},tmp{i,3});
                 dY = dY + this.h*Jmv(this.layer2,dth2(:,i),dZ,th2(:,i),tmp{i,2},tmp{i,4});
                if this.outTimes(i)==1
                    dXdata = [dXdata; this.Q*[dY;dZ]];
                end
            end
            dX = [dY;dZ];
        end
        
        % -------- Jacobian' matvecs ----------------
        function W = JYTmv(this,Wdata,W,theta,Y,tmp)
            nex = numel(Y)/numelFeatOut(this);
            if ~isempty(Wdata)
               Wdata  = reshape(Wdata,[],nnz(this.outTimes),nex);
               WdataY = Wdata(1:sizeFeatIn(this.layer1),:,:);
               WdataZ = Wdata(sizeFeatIn(this.layer1)+1:end,:,:);
            end
            if isempty(W)
                WY = 0;
                WZ = 0;
            elseif not(isscalar(W))
                [WY,WZ] = splitData(this,W);
            end
            [th1,th2]  = split(this,theta);
            cnt = nnz(this.outTimes);
            for i=this.nt:-1:1
                if this.outTimes(i)==1
                    WY = WY + this.Q'*squeeze(WdataY(:,cnt,:));
                    WZ = WZ + this.Q'*squeeze(WdataZ(:,cnt,:));
                    cnt=cnt-1;
                end
                dWZ = JYTmv(this.layer2,WY,[],th2(:,i),tmp{i,2},tmp{i,4});
                WZ  = WZ + this.h*dWZ;
                
                dWY = JYTmv(this.layer1,WZ,[],th1(:,i),tmp{i,1},tmp{i,3});
                WY  = WY - this.h*dWY;
            end
            W = [WY;WZ];
        end
        
        function [dtheta,W] = JTmv(this,Wdata,W,theta,Y,tmp,doDerivative)
            if not(exist('doDerivative','var')) || isempty(doDerivative)
               doDerivative =[1;0]; 
            end
            
            nex = numel(Y)/numelFeatOut(this);
            if ~isempty(Wdata)
               Wdata  = reshape(Wdata,[],nnz(this.outTimes),nex);
               WdataY = Wdata(1:sizeFeatIn(this.layer1),:,:);
               WdataZ = Wdata(sizeFeatIn(this.layer1)+1:end,:,:);
            end
            if isempty(W) || numel(W)==1
                WY = 0;
                WZ = 0;
            elseif not(isscalar(W))
                [WY,WZ] = splitData(this,W);
            end
            [th1,th2]   = split(this,theta);
            [dth1,dth2] = split(this,0*theta);
            
            cnt = nnz(this.outTimes);
            for i=this.nt:-1:1
                if this.outTimes(i)==1
                    WY = WY + this.Q'*squeeze(WdataY(:,cnt,:));
                    WZ = WZ + this.Q'*squeeze(WdataZ(:,cnt,:));
                    cnt=cnt-1;
                end
                [dt2,dWZ] = JTmv(this.layer2,WY,[],th2(:,i),tmp{i,2},tmp{i,4});
                WZ  = WZ + this.h*dWZ;
                dth2(:,i) = this.h*dt2;
                
                [dt1,dWY] = JTmv(this.layer1,WZ,[],th1(:,i),tmp{i,1},tmp{i,3});
                WY  = WY - this.h*dWY;
                dth1(:,i) = -this.h*dt1;
            end
            dtheta = vec([dth1;dth2]);
            W = [WY;WZ];
            if nargout==1 && all(doDerivative==1)
                dtheta=[dtheta; W(:)];
            end

        end
        function [thFine] = prolongateConvStencils(this,theta,getRP)
            % prolongate convolution stencils, doubling image resolution
            %
            % Inputs:
            %
            %   theta - weights
            %   getRP - function for computing restriction operator, R, and
            %           prolongation operator, P. Default @avgRestrictionGalerkin
            %
            % Output
            %  
            %   thFine - prolongated stencils
            
            if not(exist('getRP','var')) || isempty(getRP)
                getRP = @avgRestrictionGalerkin;
            end
            
            [th1Fine,th2Fine] = split(this,theta);
            for k=1:this.nt
                th1Fine(:,k) = vec(prolongateConvStencils(this.layer1,th1Fine(:,k),getRP));
                th2Fine(:,k) = vec(prolongateConvStencils(this.layer2,th2Fine(:,k),getRP));
            end
            thFine = vec([th1Fine;th2Fine]);
        end
        function [thCoarse] = restrictConvStencils(this,theta,getRP)
            % restrict convolution stencils, dividing image resolution by two
            %
            % Inputs:
            %
            %   theta - weights
            %   getRP - function for computing restriction operator, R, and
            %           prolongation operator, P. Default @avgRestrictionGalerkin
            %
            % Output
            %  
            %   thCoarse - restricted stencils
            
            if not(exist('getRP','var')) || isempty(getRP)
                getRP = @avgRestrictionGalerkin;
            end
            
            
           [th1Coarse,th2Coarse] = split(this,theta);
            for k=1:this.nt
                th1Coarse(:,k) = vec(restrictConvStencils(this.layer1,th1Coarse(:,k),getRP));
                th2Coarse(:,k) = vec(restrictConvStencils(this.layer2,th2Coarse(:,k),getRP));
            end
            thCoarse = vec([th1Coarse;th2Coarse]);
        end
        % ------- functions for handling GPU computing and precision ----
        function this = set.useGPU(this,value)
            if (value~=0) && (value~=1)
                error('useGPU must be 0 or 1.')
            else
                this.layer1.useGPU  = value;
                this.layer2.useGPU  = value;
            end
        end
        function this = set.precision(this,value)
            if not(strcmp(value,'single') || strcmp(value,'double'))
                error('precision must be single or double.')
            else
                this.layer1.precision = value;
                this.layer2.precision = value;
            end
        end
        function useGPU = get.useGPU(this)
            useGPU = this.layer1.useGPU;
        end
        function precision = get.precision(this)
            precision = this.layer1.precision;
        end
        
        function runMinimalExample(~)
            
            layer = doubleSymLayer(dense([2,2]));
%             layer = singleLayer(affineTrafo(dense([2,2])));
            net   = DoubleHamiltonianNN(layer,layer,100,.1);
            Y = [1;1;1;1];
            theta =  [vec([2 1;-1 2]);vec([2 -1;1 2])];
            theta = vec(repmat(theta,1,net.nt));

            
            [Yd,YN, tmp] = forwardProp(net,theta,Y);
            Ys = reshape(cell2mat(tmp(:,1)),2,[]);
            
            
            figure(1); clf;
            plot(Y(1,:),Y(2,:),'.r','MarkerSize',20);
            hold on;
            plot(Ys(1,:),Ys(2,:),'-k');
            plot(YN(1,:),YN(2,:),'.b','MarkerSize',20);
            
            
            return
            D   = affineTrafo(dense(nK));
            S   = singleLayer(D);
            net = LeapFrogNN(S,2,.01);
            mb  = randn(nTheta(net),1);
            
            Y0  = randn(nK(2),nex);
            [Y,tmp]   = net.forwardProp(mb,Y0);
            dmb = reshape(randn(size(mb)),[],net.nt);
            dY0  = randn(size(Y0));
            
            dY = net.Jmv(dmb(:),dY0,mb,[],tmp);
            for k=1:14
                hh = 2^(-k);
                
                Yt = net.forwardProp(mb+hh*dmb(:),Y0+hh*dY0);
                
                E0 = norm(Yt(:)-Y(:));
                E1 = norm(Yt(:)-Y(:)-hh*dY(:));
                
                fprintf('h=%1.2e\tE0=%1.2e\tE1=%1.2e\n',hh,E0,E1);
            end
            
            W = randn(size(Y));
            t1  = W(:)'*dY(:);
            
            [dWdmb,dWY] = net.JTmv(W,mb,[],tmp);
            t2 = dmb(:)'*dWdmb(:) + dY0(:)'*dWY(:);
            
            fprintf('adjoint test: t1=%1.2e\tt2=%1.2e\terr=%1.2e\n',t1,t2,abs(t1-t2));
            
            
            
            
            
        end
    end
    
end

