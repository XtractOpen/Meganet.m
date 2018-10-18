classdef NN < abstractMeganetElement
    % NN Neural Network block
    %
    % Y_k+1 = layer{k}(trafo(theta{k},Y_k))
    
    
    properties
        layers  % layers of Neural Network, cell array
        outTimes
        Q
        useGPU
        precision
    end
    
    methods
        function this = NN(layers,varargin)
            % constructs a Neural Network from the provided layers.
            %
            % Input:
            %
            % layers  - cell array of individual layers, note that number
            %           of output features of k-th layer must match number
            %           of input features to (k+1)-th layer
            %
            % Output
            %
            % this    - instance of this class
            %
            % Examples:
            %
            % net = NN({singleLayer(dense([3 8])})
            
            if nargin==0
                this.runMinimalExample;
                return;
            end
            useGPU = [];
            precision = [];
            nt   = numel(layers);
            outTimes = zeros(nt,1);  outTimes(end)=1;
            Q = 1.0;
            for k=1:2:length(varargin)     % overwrites default parameter
                eval([varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            if not(isempty(useGPU))
                for k=1:nt
                    layers{k}.useGPU = useGPU;
                end
            end
            if not(isempty(precision))
                for k=1:nt
                    layers{k}.precision = precision;
                end
            end
            
            nout = sizeFeatOut(layers{1});
            for k=2:nt
                if any(sizeFeatIn(layers{k}) ~= nout)
                    error('%s - dim. of input features must match dim. of output features',...
                        mfilename);
                end
                nout = sizeFeatOut(layers{k});
            end
            this.layers   = layers;
            this.outTimes = outTimes;
            this.Q        = Q;
        end
        
        % ---------- counting thetas, input and output features -----
        function n = nTheta(this)
            n = 0;
            for k=1:numel(this.layers)
                n = n + nTheta(this.layers{k});
            end
        end
        
        function n = sizeFeatIn(this)
            n = sizeFeatIn(this.layers{1});
        end
        
        function n = sizeFeatOut(this)
            n = sizeFeatOut(this.layers{end});
        end
        
        function n = nDataOut(this)
            n=0;
            for k=1:numel(this.layers)
                n = n+this.outTimes(k)* prod(sizeFeatOut(this.layers{k}));
            end
        end
        
        function theta = initTheta(this)
            theta = [];
            for k=1:numel(this.layers)
                theta = [theta; vec(initTheta(this.layers{k}))];
            end
        end
        
        
        function vars = split(this,var)
            nb = numel(this.layers);
            vars = cell(nb,1);
            cnt = 0;
            for k=1:nb
                nk = nTheta(this.layers{k});
                vars{k} = var(cnt+(1:nk));
                cnt = cnt + nk;
            end
        end
        
        % --------- forward problem ----------
        function [Ydata,Y,tmp] = forwardProp(this,theta,Y0)
            Y = Y0;
            nt = numel(this.layers);
            
            if nargout>1;    tmp = cell(nt,2); end
            Ydata = [];
            cnt = 0;
            for i=1:nt
                ni = nTheta(this.layers{i});
                if (nargout>1), tmp{i,1} = Y; end
                [Y,~,tmp{i,2}] = this.layers{i}.forwardProp(theta(cnt+(1:ni)),Y);
                if this.outTimes(i)==1 
                    Ydata = [Ydata; this.Q*Y];
                end
                cnt = cnt + ni;
            end
        end
        function [thetaNorm] = getNormalizedWeights(this,theta,Y,nL,thetaNL)
            if not(exist('nL','var')); nL = []; end
            if not(exist('thetaNL','var')); thetaNL = []; end
            nt = numel(this.layers);
            cnt = 0;
            thetaNorm = 0*theta;
            for i=1:nt
                ni = nTheta(this.layers{i});
                thetaNorm(cnt+(1:ni)) = getNormalizedWeights(this.layers{i},theta(cnt+(1:ni)),Y,nL,thetaNL);
                Y = this.layers{i}.forwardProp(theta(cnt+(1:ni)),Y);
                cnt = cnt + ni;
            end
        end
        % -------- Jacobian matvecs --------
        function [dYdata,dY] = JYmv(this,dY,theta,~,tmp)
            nt = numel(this.layers);
            cnt = 0;
            dYdata = [];
            for i=1:nt
                ni = nTheta(this.layers{i});
                dY = JYmv(this.layers{i},dY,theta(cnt+(1:ni)),...
                    tmp{i,1},tmp{i,2});
                if this.outTimes(i)==1
                    dYdata = [dYdata; this.Q*dY];
                end
                cnt = cnt+ni;
            end
        end
        
        function [dYdata,dY] = Jmv(this,dtheta,dY,theta,~,tmp)
            nt = numel(this.layers);
            if isempty(dY); dY = 0.0; end
            
            dYdata= [];
            cnt = 0; 
            for i=1:nt
                ni = nTheta(this.layers{i});
                dY = this.layers{i}.Jmv(dtheta(cnt+(1:ni)),dY,theta(cnt+(1:ni)),...
                    tmp{i,1},tmp{i,2});
                if this.outTimes(i)==1
                    dYdata = [dYdata; this.Q*dY];
                end
                cnt = cnt+ni;
            end
        end
        
        % -------- Jacobian' matvecs --------
        function W = JYTmv(this,Wdata,W,theta,Y,tmp)
            nex = numel(Y)/numelFeatIn(this);
            if ~isempty(Wdata)
                Wdata = reshape(Wdata,[],nex);
            end
            if isempty(W)
                W = 0;
            elseif numel(W)>1
                W     = reshape(W,[],nex);
            end
            nt = numel(this.layers);
            
            cnt = 0; cnt2 = 0;
            for i=nt:-1:1
                Yi = tmp{i,1};
                ni = nTheta(this.layers{i});
                if this.outTimes(i)==1
                    nn = prod(sizeFeatOut(this.layers{i}));
                    W = W + this.Q'*Wdata(end-cnt2-nn+1:end-cnt2,:);
                    cnt2 = cnt2 + nn;
                end
                W  = JYTmv(this.layers{i}, W,[],theta(end-cnt-ni+1:end-cnt),...
                    Yi,tmp{i,2});
                cnt = cnt+ni;
            end
        end
            
        function [dtheta,W] = JTmv(this,Wdata,W,theta,Y,tmp,doDerivative)
            if not(exist('doDerivative','var')) || isempty(doDerivative); 
               doDerivative =[1;0]; 
            end
            
            nex = numel(Y)/numelFeatIn(this);
            if ~isempty(Wdata)
                Wdata = reshape(Wdata,[],nex);
            end
            if isempty(W)
                W = 0;
            elseif numel(W)>1
                W     = reshape(W,[],nex);
            end
            
            dtheta = 0*theta;
            nt = numel(this.layers);
            
            cnt = 0; cnt2 = 0;
            for i=nt:-1:1
                Yi = tmp{i,1};
                if this.outTimes(i)==1
                    nn = prod(sizeFeatOut(this.layers{i}));
                    W = W+ this.Q'*Wdata(end-cnt2-nn+1:end-cnt2,:);
                    cnt2 = cnt2 + nn;
                end
                ni        = nTheta(this.layers{i});
                [dmbi,W] = JTmv(this.layers{i},W,[],theta(end-cnt-ni+1:end-cnt),...
                    Yi,tmp{i,2});
                dtheta(end-cnt-ni+1:end-cnt)  = dmbi;
                cnt = cnt+ni;
            end
            
            if nargout==1 && all(doDerivative==1)
                dtheta=[dtheta(:); W(:)];
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
            %   thCoarse - restricted stencils
            
            if not(exist('getRP','var')) || isempty(getRP)
                getRP = @avgRestrictionGalerkin;
            end
            thFine = split(this,theta);
            for k=1:numel(this.layers)
                thFine{k} = prolongateConvStencils(this.layers{k},thFine{k},getRP);
            end
            thFine = vec(thFine);
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
            
            if not(exist('getRP','var'))
                getRP = [];
            end
            
            thCoarse = split(this,theta);
            for k=1:numel(this.layers)
                thCoarse{k} = restrictConvStencils(this.layers{k},thCoarse{k},getRP);
            end
            thCoarse = vec(thCoarse);
        end
        
        % ------- functions for handling GPU computing and precision ----
        function this = set.useGPU(this,value)
            if (value~=0) && (value~=1)
                error('useGPU must be 0 or 1.')
            else
                for k=1:length(this.layers)
                    this.layers{k}.useGPU  = value;
                end
            end
            this.Q = gpuVar(value,this.precision,this.Q);
        end
        function this = set.precision(this,value)
            if not(strcmp(value,'single') || strcmp(value,'double'))
                error('precision must be single or double.')
            else
                for k=1:length(this.layers)
                    this.layers{k}.precision  = value;
                end
            end
            this.Q = gpuVar(this.useGPU,value,this.Q);
        end
        function useGPU = get.useGPU(this)
            useGPU = this.layers{1}.useGPU;
            for k=2:length(this.layers)
                useGPU2 = this.layers{k}.useGPU;
                if useGPU~=useGPU2
                    error('both transformations need to be on GPU or CPU')
                end
            end
            
        end
        function precision = get.precision(this)
            precision = this.layers{1}.precision;
            for k=2:length(this.layers)
                precision2 = this.layers{k}.precision;
                if strcmp(precision,precision2)
                    warning('precisions of all layers must agree')
                    this.layers{k}.precision = precision;
                end
            end
        end
        
        
        function runMinimalExample(~)
            nex = 10;
            layers = cell(1,1);
            
            layers{1} = singleLayer(dense([4 2]));
             layers{2} = singleLayer(dense([8 4]));
            
            net = NN(layers);
            net.outTimes = [1;1];
            mb  = randn(nTheta(net),1);
            
            Y0  = randn(2,nex);
            [Ydata,~,tmp]   = net.forwardProp(mb,Y0);
            dmb = randn(size(mb));
            dY0 = randn(size(Y0));
            
            dY = net.Jmv(dmb(:),dY0,mb,Y0,tmp);
            for k=1:14
                hh = 2^(-k);
                
                Yt = net.forwardProp(mb+hh*dmb(:),Y0+hh*dY0);
                
                E0 = norm(Yt(:)-Ydata(:));
                E1 = norm(Yt(:)-Ydata(:)-hh*dY(:));
                
                fprintf('h=%1.2e\tE0=%1.2e\tE1=%1.2e\n',hh,E0,E1);
            end
            
            W = randn(size(Ydata));
            t1  = W(:)'*dY(:);
            
            [dWdmb,dWY] = net.JTmv(W,[],mb,Y0,tmp);
            t2 = dmb(:)'*dWdmb(:) + dY0(:)'*dWY(:);
            
            fprintf('adjoint test: t1=%1.2e\tt2=%1.2e\terr=%1.2e\n',t1,t2,abs(t1-t2));
        end
    end
    
end

