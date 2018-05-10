classdef Meganet < abstractMeganetElement
    % a Meganet object consists of several blocks (i.e., ResNN, NN) and
    % handles the forward propagation through their concatenation.
    
    properties
        blocks
        outTimes
        useGPU
        precision
    end
    
    methods
        function this = Meganet(blocks,varargin)
            if nargin==0
                this.runMinimalExample;
                return
            end
            nb = numel(blocks);
            useGPU = [];
            precision = [];
            outTimes = zeros(nb,1); outTimes(end)=1;
            for k=1:2:length(varargin)     % overwrites default parameter
                eval([varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            if not(isempty(useGPU))
                for k=1:nb
                    blocks{k}.useGPU = useGPU;
                end
            end
            if not(isempty(precision))
                for k=1:nb
                    blocks{k}.precision = precision;
                end
            end
            
            nout = nFeatOut(blocks{1});
            for k=2:nb
                if nFeatIn(blocks{k}) ~= nout
                    error('%s - dim of input features of layer %d does not match output features of layer %d',mfilename,k,k-1);
                end
                nout = nFeatOut(blocks{k});
            end
            this.blocks = blocks;
            for k=1:nb
                this.blocks{k}.outTimes = this.blocks{k}.outTimes*outTimes(k);
            end
            if nDataOut(this)==0
                this.blocks{end}.outTimes(end)=1;
            end
        end
        
        % ------- counting ----------
        function n = nTheta(this)
            nb = numel(this.blocks);
            n  = 0;
            for k=1:nb
                n = n + nTheta(this.blocks{k});
            end
        end
        function n = nFeatOut(this)
            n = nFeatOut(this.blocks{end});
        end
        
        function n = nFeatIn(this)
            n = nFeatIn(this.blocks{1});
        end
        function n = nDataOut(this)
            n = 0;
            for k=1:numel(this.blocks)
                n = n + nDataOut(this.blocks{k});
            end
        end
        
        function theta = initTheta(this)
            theta = [];
            for k=1:numel(this.blocks)
                theta = [theta; vec(initTheta(this.blocks{k}))];
            end
            theta = gpuVar(this.useGPU,this.precision,theta);
        end
        
        function [net2,theta2] = prolongateWeights(this,theta)
            % piecewise linear interpolation of network weights 
            nb   = numel(this.blocks);
            net2 = cell(nb,1);
            th   = split(this,theta);
            theta2 = [];
            for k=1:nb
                [n2,th2] = prolongateWeights(this.blocks{k},th{k});
                theta2 = [theta2; vec(th2)];
                net2{k}=n2;
            end
            net2 = Meganet(net2);
        end
        
        function vars = split(this,var)
            nb = numel(this.blocks);
            vars = cell(nb,1);
            cnt = 0;
            for k=1:nb
                nk = nTheta(this.blocks{k});
                vars{k} = var(cnt+(1:nk));
                cnt = cnt + nk;
            end
        end
        
        function idx = getBlockIndices(this)
            nb = numel(this.blocks);
            nth = nTheta(this);
            idx = zeros(nth,1);
            
            cnt = 0;
            for k=1:nb
                nk = nTheta(this.blocks{k});
                idx(cnt+(1:nk)) = k;
                cnt = cnt + nk;
            end
        end
        
        % ---------- apply forward problem ------------
        function [Ydata,Y,tmp] = apply(this,theta,Y0)
            nex = numel(Y0)/nFeatIn(this);
            Y0  = reshape(Y0,[],nex);
            nb = numel(this.blocks);
            Y  = Y0;
            tmp = cell(nb,1);
            thetas = split(this,theta);
            Ydata = [];
            for k=1:nb
                [Yd,Y,tmp{k}] = apply(this.blocks{k},thetas{k},Y);
                Ydata = [Ydata;Yd];
            end
        end
        
        function YN = applyBatch(this,theta,Y0,batchSize)
           nex = numel(Y0)/nFeatIn(this);
           YN = zeros(nDataOut(this),nex,'like',Y0);
           nb = ceil(nex/batchSize);
           id = randperm(nex);
           cnt = 1;
           for k=1:nb
               idk = id(cnt:min(nex,cnt+batchSize));
               if numel(idk)==0
                   break;
               end
                YN(:,idk) = apply(this,theta,Y0(:,idk));
                cnt = cnt + numel(idk);
            end
        end
        
        % ----------- Jacobian matvecs -------------
        function [dYdata,dY] = JYmv(this,dY,theta,~,tmp)
            nex = numel(dY)/nFeatIn(this);
            dY  = reshape(dY,[],nex);
            dYdata = [];
            nb = numel(this.blocks);
            cnt = 0;
            for k=1:nb
                nk = nTheta(this.blocks{k});
                [dYdatak,dY] = JYmv(this.blocks{k},dY,theta(cnt+(1:nk)),[],tmp{k});
                dYdata = [dYdata;dYdatak];
                cnt = cnt+nk;
            end
        end
        
        function [dYdata,dY] = Jmv(this,dtheta,dY,theta,~,tmp)
            nex = numel(dY)/nFeatIn(this);
            dY  = reshape(dY,[],nex);
            dYdata = [];
            nb = numel(this.blocks);
            cnt = 0;
            for k=1:nb
                nk = nTheta(this.blocks{k});
                [dYdatak,dY] = Jmv(this.blocks{k},dtheta(cnt+(1:nk)),dY,theta(cnt+(1:nk)),[],...
                    tmp{k});
                
                dYdata = [dYdata;dYdatak];
                cnt = cnt+nk;
            end
        end
        
        % ----------- Jacobian' matvecs -----------
        function W = JYTmv(this,Wdata,~,theta,Y,tmp)
            nex = numel(Y)/nFeatIn(this);
            Wdata  = reshape(Wdata,[],nex);
            nb  = numel(this.blocks);
            
            cnt = 0; cntW = 0; W = [];
            for k=nb:-1:1
                nk = nTheta(this.blocks{k});
                if cntW < size(Wdata,1)
                    no = nFeatOut(this.blocks{k});
                    Wdk =  Wdata(end-cntW-no+1:end-cntW,:);
                    cntW = cntW + no;
                else
                    Wdk = [];
                end
                W = JYTmv(this.blocks{k},Wdk,W,theta(end-cnt-nk+1:end-cnt),tmp{k}{1,1},...
                    tmp{k});
                cnt = cnt+nk;
            end
        end
        
       
        function [dtheta,W] = JTmv(this,Wdata,W,theta,Y,tmp,doDerivative)
            if not(exist('doDerivative','var')) || isempty(doDerivative); 
               doDerivative =[1;0]; 
            end
            
            nex = numel(Y)/nFeatIn(this);
            if isempty(W)
                W=0;
            else
                W  = reshape(W,[],nex);
            end
            if not(isempty(Wdata))
                Wdata = reshape(Wdata,[],nex);
            end
            
            nb  = numel(this.blocks);
            dtheta = 0*theta;
            
            cnt = 0;cntW = 0; cntWd = 0;
            for k=nb:-1:1
                nk = nTheta(this.blocks{k});
                no = nFeatOut(this.blocks{k});
                %                 W  = Wdata(end-cntW-no+1:end-cntW,:);
                if any(this.blocks{k}.outTimes)
                    ndk = nDataOut(this.blocks{k});
                    Wd  = Wdata(end-cntWd-ndk+1:end-cntWd,:);
                else
                    Wd = [];
                end
                cntW = cntW + no;
                [dmbk,W] = JTmv(this.blocks{k},Wd,W,theta(end-cnt-nk+1:end-cnt),tmp{k}{1,1},tmp{k});
                dtheta(end-cnt-nk+1:end-cnt) = dmbk;
                cnt = cnt+nk;
            end
            if nargout==1 && all(doDerivative==1)
                dtheta=[dtheta(:); W(:)];
            end
        end
        
        % ------- functions for handling GPU computing and precision ----
        function this = set.useGPU(this,value)
            if (value~=0) && (value~=1)
                error('useGPU must be 0 or 1.')
            else
                for k=1:length(this.blocks)
                    this.blocks{k}.useGPU  = value;
                end
            end
        end
        function this = set.precision(this,value)
            if not(strcmp(value,'single') || strcmp(value,'double'))
                error('precision must be single or double.')
            else
                for k=1:length(this.blocks)
                    this.blocks{k}.precision  = value;
                end
            end
        end
        function this = set.outTimes(this,value)
            nb = numel(this.blocks);
            for k=1:nb
                this.blocks{k}.outTimes = this.blocks{k}.outTimes*value(k);
                this.outTimes = value;
            end
        end
        function useGPU = get.useGPU(this)
            useGPU = this.blocks{1}.useGPU;
            for k=2:length(this.blocks)
                useGPU2 = this.blocks{k}.useGPU;
                if useGPU~=useGPU2
                    error('all transformations need to be on GPU or CPU')
                end
            end
            
        end
        function precision = get.precision(this)
            precision = this.blocks{1}.precision;
            for k=2:length(this.blocks)
                precision2 = this.blocks{k}.precision;
                if not(isempty(precision)) && not(isempty(precision2)) && not(strcmp(precision,precision2))
                    error('precisions of all blocks must agree')
                end
            end
        end
        
        function runMinimalExample(~)
            nex    = 10;
            blocks = cell(0,1);
            blocks{end+1} = NN({singleLayer(dense([4 2]))},'outTimes',0);
            blocks{end+1} = ResNN(singleLayer(dense([4 4])),3,1);
            net    = Meganet(blocks);
            net.useGPU=0;
            net.precision = 'double';
            np  = nTheta(net);
            mb     = randn(np,1);
            
            Y0 = randn(2,nex);
            
            [Ydata,~,tmp] = net.apply(mb,Y0);
            
            dmb = randn(np,1);
            dY0 = randn(size(Y0));
            dY  = net.Jmv(dmb,dY0,mb,[],tmp);
            
            for k=1:14
                hh = 2^(-k);
                
                Yt = net.apply(mb+hh*dmb(:),Y0+hh*dY0);
                
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

