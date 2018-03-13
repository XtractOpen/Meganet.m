classdef blockReg
    % classdef blockReg
    %
    % parent class that collects regularizers for each block. 
    %
    
    properties
        blocks
        useGPU
        precision
    end
    
    methods
        
        function this = blockReg(blocks,varargin)
            if nargin==0
                this.runMinimalExample()
                return
            end
           useGPU    = blocks{1}.useGPU;
           precision = blocks{1}.precision;
           for k=1:2:length(varargin)     % overwrites default parameter
                eval([varargin{k},'=varargin{',int2str(k+1),'};']);
           end
           this.blocks = blocks;
           
           if not(isempty(useGPU))
                this.useGPU    = useGPU;
           end
           if not(isempty(precision))
                this.precision = precision;
           end
           
        end
        
        function nth = nTheta(this)
            nb = numel(this.blocks);
            nth = 0;
            for k=1:nb
                nth = nth + nTheta(this.blocks{k});
            end
        end
        
        function [Sc,para,dS,d2S] = regularizer(this,x)
            nb  = numel(this.blocks);
            Sc  = 0;
            dS  = zeros(nTheta(this),1,'like',x);
            d2S = cell(nb,1);
            para = [];
            cnt = 0;
            for k=1:nb
                nk  = nTheta(this.blocks{k});
                idx = cnt + (1:nk);
                [Sk,~,dSk,d2Sk] = regularizer(this.blocks{k},x(idx));
                para = [para Sk];
                Sc = Sc + Sk;
                dS(idx) = dSk;
                d2S{k} = d2Sk;
                cnt = cnt + nk;                
            end
            d2S = blkdiag(d2S{:});
            para = [Sc 1.0 para];
        end
        
        function [str,frmt] = hisNames(this)
            str  = {'R','alpha'};
            frmt = {'%-12.2e','%-12.2e'};
        end
        function str = hisVals(this,para)
            str = para(1:2);
        end       
        
        function PC = getPC(this)
            nb  = numel(this.blocks);
            PC = cell(nb,1);
            for k=1:nb
                PC{k} = getPCop(this.blocks{k}.B);
            end
            PC = blkdiag(PC{:});
        end
        
        function runMinimalExample(~)
            nt = 10;
            h = 0.1;
            nTh = 100;
            B1 = opTimeDer(nTh,nt,h);
            reg1 = l1Reg(B1,[1.0;0.0]);
            reg2 = tikhonovReg(B1,rand(1),0);
            reg = blockReg({reg1,reg2});
            
            th = randn(nTheta(reg),1);
            [Sc,para,dS,d2S] = regularizer(reg,th);
            fctn = @(th) regularizer(reg,th);
            checkDerivative(fctn,th,'out',2)
        end
        % ------- functions for handling GPU computing and precision ----
        function this = set.useGPU(this,value)
            if isempty(value)
                return
            elseif(value~=0) && (value~=1)
                error('useGPU must be 0 or 1.')
            else
                for k=1:numel(this.blocks)
                    this.blocks{k}.useGPU = value;
                end
            end
        end
        function this = set.precision(this,value)
            if isempty(value)
                return
            elseif not(strcmp(value,'single') || strcmp(value,'double'))
                error('precision must be single or double.')
            else
                for k=1:numel(this.blocks)
                    this.blocks{k}.precision = value;
                end
            end
        end
        function useGPU = get.useGPU(this)
            useGPU = this.blocks{1}.useGPU;
            
        end
        function precision = get.precision(this)
            precision = this.blocks{1}.precision;
        end
    end
    
end

