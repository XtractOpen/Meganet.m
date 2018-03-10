classdef opBlkdiag < LinearOperator
    % identity operator
    
    properties
        blocks
        useGPU
        precision
    end
    
    methods
        function this = opBlkdiag(varargin)
            m = 0; n = 0;
            for k=1:numel(varargin)
                m = m + varargin{k}.m;
                n = n + varargin{k}.n;
            end
            this.blocks = varargin;
            this.m = m; 
            this.n = n;
            this.Amv = @(x) blkdiagmv(this,x);
            this.ATmv = @(x)blkdiagTmv(this,x);
        end
        
        function Ax = blkdiagmv(this,x)
            cntm = 0;
            cntn = 0;
            Ax  = zeros(this.m,size(x,2),'like',x);
            for k=1:numel(this.blocks)
                nk = this.blocks{k}.n;
                mk = this.blocks{k}.m;
               xk = x(cntn+(1:nk),:);
               Ax(cntm+(1:mk),:) = this.blocks{k}*xk;
               
               cntm = cntm+mk;
               cntn = cntn+nk;
            end
        end
        
        function Ax = blkdiagTmv(this,x)
            cntm = 0;
            cntn = 0;
            Ax  = zeros(this.n,size(x,2),'like',x);
            for k=1:numel(this.blocks)
                nk = this.blocks{k}.n;
                mk = this.blocks{k}.m;
                xk = x(cntm+(1:mk),:);
                Ax(cntn+(1:nk),:) = this.blocks{k}'*xk;
                
                cntm = cntm+mk;
                cntn = cntn+nk;
            end
        end
        
        function PCop = getPCop(this,x)
            pcs = cell(numel(this.blocks),1);
            for k=1:numel(pcs)
                pcs{k} = getPCop(this.blocks{k});
            end
            PCop = opBlkdiag(pcs{:});       
        end
        
        function y = PCmv(this,x,alpha,gamma)
            % x = argmin_x alpha/2*|D*x|^2+gamma/2*|x-y|^2
            % minimum norm solution when rank-deficient
            if not(exist('alpha','var')) || isempty(alpha)
                alpha = 1;
            end
            if not(exist('gamma','var')) || isempty(gamma)
                gamma = 0;
            end
            cntn = 0;
            y  = zeros(this.n,size(x,2),'like',x);
            for k=1:numel(this.blocks)
                nk = this.blocks{k}.n;
                xk = x(cntn+(1:nk),:);
                y(cntn+(1:nk),:) = PCmv(this.blocks{k},xk,alpha,gamma);
               
               cntn = cntn+nk;
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
        function useGPU = get.useGPU(this)
            useGPU = this.blocks{1}.useGPU;
            for k=2:length(this.blocks)
                useGPU2 = this.blocks{k}.useGPU;
                if not(isempty(useGPU2)) && not(isempty(useGPU)) && (useGPU~=useGPU2)
                    error('all blocks need to be on GPU or CPU')
                end
            end
        end
        
   
        function precision = get.precision(this)
            precision = this.blocks{1}.precision;
            for k=2:length(this.blocks)
                precision2 = this.blocks{k}.precision;
                if not(isempty(precision2)) && not(isempty(precision)) && not(strcmp(precision,precision2))
                    error('precisions of all blocks must agree')
                end
                if isempty(precision) && not(isempty(precision2))
                    precision = precision2;
                end
            end
        end
        
        function this = convertGPUorPrecision(this,useGPU,precision)
            if strcmp(this.precision,'double') && (isa(this.lam,'single') || isa(this.lamInv,'single'))
                [this.lam,this.lamInv] = getEigs(this);
            end
            [this.lam, this.lamInv] = gpuVar(useGPU,precision,this.lam,this.lamInv);
        end
        
        
%         function this = gpuVar(this,useGPU,precision)
%             %% TODO: rethink this
%         end

    end
end

