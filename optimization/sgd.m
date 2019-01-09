classdef sgd < optimizer
    % classdef sgd < optimizer
    %
    % stochastic gradient descent optimizer for minimizing nonlinear objectives
    
    properties
        maxEpochs
        miniBatch
        atol
        rtol
        maxStep
        out
        learningRate
        momentum
        nesterov
		ADAM
        P
    end
    
    methods
        
        function this = sgd(varargin)
            this.maxEpochs = 10;
            this.miniBatch = 16;
            this.atol    = 1e-3;
            this.rtol    = 1e-3;
            this.maxStep = 1.0;
            this.out     = 0;
            this.learningRate = 0.1;
            this.momentum  = .9;
            this.nesterov  = true;
			this.ADAM      = false;
            this.P = @(x) x;
            for k=1:2:length(varargin)     % overwrites default parameter
                eval(['this.' varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            if this.ADAM && this.nesterov
                warning('sgd(): ADAM and nestrov together - choosing ADAM');
                this.nesterov  = false;
            end
        end
        
        function [str,frmt] = hisNames(this)
            str  = {'epoch', 'Jc','|x-xOld|','learningRate'};
            frmt = {'%-12d','%-12.2e','%-12.2e','%-12.2e'};
        end
        
        function [xc,His,xOptAcc,xOptLoss] = solve(this,fctn,xc,fval,varargin)
            optValAcc    = 0;
            optValLoss   = Inf;
            for k=1:2:length(varargin)     % overwrites default parameter
                eval([varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            
            if not(exist('fval','var')); fval = []; end;
            xOptAcc = [];  % iterate with optimal validation accuracy
            xOptLoss = []; % iterate with optimal validation loss
            [str,frmt] = hisNames(this);
            
            % parse objective functions
            [fctn,objFctn,objNames,objFrmt,objHis]     = parseObjFctn(this,fctn);
            str = [str,objNames{:}]; frmt = [frmt,objFrmt{:}];
            [fval,obj2Fctn,obj2Names,obj2Frmt,obj2His] = parseObjFctn(this,fval);
            str = [str,obj2Names{:}]; frmt = [frmt,obj2Frmt{:}];
            doVal     = not(isempty(obj2Fctn));
            
            % evaluate training and validation
            
            epoch = 1;
            xOld = xc;
            dJ = 0*xc;
            mJ = 0;
            vJ = 0;
            if this.ADAM
                mJ = 0*xc;
                vJ = 0*xc;
                this.learningRate = 0.001;
            end
            beta2 = 0.999;
            beta1 = this.momentum;
            
            if isnumeric(this.learningRate)
                learningRate    = @(epoch) this.learningRate;
            elseif isa(this.learningRate,'function_handle')
                learningRate  = this.learningRate;
            else
                error('%s - learningRate must be numeric or function',mfilename);
            end
            
            if this.out>0
                fprintf('== sgd (n=%d,maxEpochs=%d, lr = %1.1e, momentum = %1.1e, ADAM = %d, Nesterov = %d, miniBatch=%d) ===\n',...
                    numel(xc), this.maxEpochs, learningRate(1) ,this.momentum, this.ADAM,this.nesterov,this.miniBatch);
                fprintf([repmat('%-12s',1,numel(str)) '\n'],str{:});
            end
            xc = this.P(xc);
            his = zeros(1,numel(str));
            
            while epoch <= this.maxEpochs
                nex = sizeLastDim(objFctn.Y);
                ids = randperm(nex);
                lr = learningRate(epoch);
                for k=1:floor(nex/this.miniBatch)
                    idk = ids((k-1)*this.miniBatch+1: min(k*this.miniBatch,nex));
                    if this.nesterov && ~this.ADAM
                        [Jk,~,dJk] = fctn(xc-this.momentum*dJ,idk);
                    else
                        [Jk,~,dJk] = fctn(xc,idk); 
                    end
                    
                    if this.ADAM
                       mJ = beta1*mJ + (1-beta1)*(dJk);
                       vJ = beta2*vJ + (1-beta2)*(dJk.^2);
%                        lr = learningRate(e);
                       dJ = lr*((mJ./(1-beta1^(epoch)))./sqrt((vJ./(1-beta2^(epoch)))+1e-8)); 
                    else
                       dJ = lr*dJk + this.momentum*dJ;
                    end
                    xc = this.P(xc - dJ);
                end
                % we sample 2^12 images from the training set for displaying the objective.     
                [Jc,para] = fctn(xc,ids(1:min(nex,2^15))); 
                if doVal
                    [Fval,pVal] = fval(xc,[]); % evaluate loss for validation data
                    valAcc = obj2His(pVal);
                    if (nargout>2) && (valAcc(2)>optValAcc)
                        xOptAcc = gather(xc);
                        optValAcc = valAcc(2);
                    end
                    if (nargout>3) && (valAcc(1)<optValLoss)
                        xOptLoss = gather(xc);
                        optValLoss = valAcc(1);
                    end
                    
                    valHis = gather(obj2His(pVal));
                else
                    valHis =[];
                end
                
                his(epoch,1:4)  = [epoch,gather(Jc),gather(norm(xOld(:)-xc(:))),lr];
                if this.out>0
                    fprintf([frmt{1:4}], his(epoch,1:4));
                end
                xOld       = xc;
                
                if size(his,2)>=5
                    his(epoch,5:end) = [gather(objHis(para)), valHis];
                    if this.out>0
                        fprintf([frmt{5:end}],his(epoch,5:end));
                    end
                end
                if this.out>0
                    fprintf('\n');
                end
                epoch = epoch + 1;
            end
            His = struct('str',{str},'frmt',{frmt},'his',his(1:min(epoch,this.maxEpochs),:));
        end
        
        function [fctn,objFctn,objNames,objFrmt,objHis] = parseObjFctn(this,fctn)
            if exist('fctn','var') && not(isempty(fctn)) && isa(fctn,'objFctn')
                objFctn  = fctn;
                [objNames,objFrmt] = objFctn.hisNames();
                objHis   = @(para) objFctn.hisVals(para);
                fctn = @(x,ids) eval(fctn,x,ids);
            else
                objFctn  = [];
                objNames = {};
                objFrmt  = {};
                objHis   = @(x) [];
            end
        end

    end
end