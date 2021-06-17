classdef sgd < optimizer
    % classdef sgd < optimizer
    %
    % stochastic gradient descent optimizer for minimizing nonlinear objectives
    
    properties
        maxEpochs = 10
        miniBatch = 16
        atol      = 1e-3
        rtol      = 1e-3
        lossTol   = -Inf
        maxStep   = 1.0
        out       = 0
        learningRate = 0.1
        momentum  = 0.9
        nesterov  = true
		ADAM      = false
        P         = @(x) x
        maxWorkUnits = Inf
    end
    
    methods
        
        function this = sgd(varargin)
            
            for k = 1:2:length(varargin)
                this.(varargin{k}) = varargin{k+1};
            end

            if this.ADAM && this.nesterov
                warning('sgd(): ADAM and nestrov together - choosing ADAM');
                this.nesterov  = false;
            end
        end
        
        function [str,frmt] = hisNames(this)
            str  = {'epoch', 'Jc','|x-xOld|','lr','TotalWork','Time'};
            frmt = {'%-12d','%-12.2e','%-12.2e','%-12.2e','%-12d','%-12.2f'};
        end
        
        function [xc,His,infoOptAcc,infoOptLoss] = solve(this,fctn,xc,fval,varargin)
            optValAcc    = 0;
            optValLoss   = Inf;
            for k=1:2:length(varargin)     % overwrites default parameter
                eval([varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            
            if not(exist('fval','var')); fval = []; end;
            infoOptAcc = [];  % iterate with optimal validation accuracy
            infoOptLoss = []; % iterate with optimal validation loss
            [str,frmt] = hisNames(this);
            
            if contains(class(fctn),'SlimTik')
                W0 = fctn.WPrev;
            end
            
            % parse objective functions
            [fctn,objFctn,objNames,objFrmt,objHis]     = parseObjFctn(this,fctn);
            str = [str,objNames{:}]; frmt = [frmt,objFrmt{:}];
            [fval,obj2Fctn,obj2Names,obj2Frmt,obj2His] = parseObjFctn(this,fval);
            str = [str,obj2Names{:}]; frmt = [frmt,obj2Frmt{:}];
            doVal     = not(isempty(obj2Fctn));
            
            % evaluate training and validation
            
            epoch = 1;
            workUnits = 2;
            xOld = xc;
            dJ = 0*xc;
            mJ = 0;
            vJ = 0;
            if this.ADAM
                mJ = 0*xc;
                vJ = 0*xc;
%                 this.learningRate = 0.001;
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
                fprintf('== sgd (n=%d,maxEpochs=%d, maxWorkUnits=%d, lr = %1.1e, momentum = %1.1e, ADAM = %d, Nesterov = %d, miniBatch=%d, lossTol=%1.1e) ===\n',...
                    numel(xc), this.maxEpochs, this.maxWorkUnits, learningRate(1) ,this.momentum, this.ADAM,this.nesterov,this.miniBatch,this.lossTol);
                fprintf([repmat('%-12s',1,numel(str)) '\n'],str{:});
            end
            xc = this.P(xc);
            his = zeros(1,numel(str));


            
                nex = sizeLastDim(objFctn.Y);
                % ids = randperm(nex);
             while (epoch <= this.maxEpochs && workUnits <= this.maxWorkUnits)
                startTime = tic;
                ids = randperm(nex);
                
                WFull = [];
                if exist('W0','var')
                    WFull = cat(2,WFull,W0(:));
                else
                    WFull = cat(2,WFull,xc(:));
                end

                lr = learningRate(epoch);
                for k=1:floor(nex/this.miniBatch)
                    idk = ids((k-1)*this.miniBatch+1: min(k*this.miniBatch,nex));

%                     if k==1
%                         % update time stepping
%                         Yk = objFctn.Y(:,:,:,idk);
%                         setTimeY(objFctn.net,xc,Yk);
%                     end
                        
                        
                    
                    if this.nesterov && ~this.ADAM
                        [Jk,para,dJk] = fctn(xc-this.momentum*dJ,idk);
                    else
                        [Jk,para,dJk] = fctn(xc,idk); 
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
                    if contains(class(objFctn),'VarPro') || contains(class(objFctn),'Tik')
                        [Fval,pVal] = fval([xc; para.W(:)],[]);
                    else
                        [Fval,pVal] = fval(xc,[]);
                    end
                    valAcc = obj2His(pVal);
                    if (nargout>2) && (valAcc(2)>optValAcc)
                        xOptAcc    = gather(xc);
                        paraOptAcc = para;
                        optValAcc  = valAcc(2);
                        infoOptAcc = struct('xOptAcc',xOptAcc,'paraOpt',paraOptAcc,'optValAcc',optValAcc);
                    end
                    if (nargout>3) && (valAcc(1)<optValLoss)
                        xOptLoss    = gather(xc);
                        paraOptLoss = para;
                        optValLoss  = valAcc(1);
                        infoOptLoss  = struct('xOptLoss',xOptLoss,'paraOptLoss',paraOptLoss,'optValLoss',optValLoss);
                    end
                    
                    valHis = gather(obj2His(pVal));
                else
                    valHis =[];
                end
                endTime = toc(startTime);
                his(epoch,1:6)  = [epoch,gather(Jc),gather(norm(xOld(:)-xc(:))),lr,workUnits,endTime];
                if this.out>0
                    fprintf([frmt{1:6}], his(epoch,1:6));
                end
                xOld       = xc;
                
                if size(his,2)>=7
                    his(epoch,7:end) = [gather(objHis(para)), valHis];
                    if this.out>0
                        fprintf([frmt{7:end}],his(epoch,7:end));
                    end
                end
                if this.out>0
                    fprintf('\n');
                end
                if (this.lossTol>-Inf) && (his(epoch,7) < this.lossTol)
                    fprintf('--- %s reached loss tolerance: terminate ---\n',mfilename);
                    break;
                end
                epoch = epoch + 1;
                workUnits = workUnits + 2;
            end
            % His = struct('str',{str},'frmt',{frmt},'his',his(1:min(epoch,this.maxEpochs),:));
            His = struct('str',{str},'frmt',{frmt},'his',his(1:min(epoch,size(his,1)),:));
        end
        
        function [fctn,objFctn,objNames,objFrmt,objHis] = parseObjFctn(this,fctn)
            if exist('fctn','var') && not(isempty(fctn)) && (isa(fctn,'objFctn') || isa(fctn,'handle'))
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