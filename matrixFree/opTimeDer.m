classdef opTimeDer < RegularizationOperator
    % time derivative computed using conv2 (for fast GPU performance)
    
    properties
        ntheta  % number of thetas overall
        nt      % number of time steps
        h       % spacing between layers
        beta    % regularization parameter
        lam     % eigenvalues of time derivative L'*L
        lamInv  % eigenvalues of pinv(L'*L)
    end
    
    methods
        function this = opTimeDer(ntheta,nt,h,useGPU,precision)
            if not(exist('h','var')) || isempty(h)
                h = 1;
            end
            if not(exist('useGPU','var')) || isempty(useGPU)
                useGPU = 0;
            end
            if not(exist('precision','var')) || isempty(precision)
                precision = 'double';
            end
            this.useGPU = useGPU;
            this.precision = precision;
            this.ntheta = ntheta;
            this.nt     = nt;
            this.h      = h;
            this.beta  = 1;
            [lam,lamInv] = getEigs(this);
            
            
            [this.lam, this.lamInv] = gpuVar(useGPU,precision,lam,lamInv);
            nth1 = ntheta/nt;
            
            this.m = nth1*(nt-1);
            this.n = ntheta;
            this.Amv = @(x) der(this,x);
            this.ATmv = @(x)derTranspose(this,x);
        end
        function this = convertGPUorPrecision(this,useGPU,precision)
            if strcmp(this.precision,'double') && (isa(this.lam,'single') || isa(this.lamInv,'single'))
                [this.lam,this.lamInv] = getEigs(this);
            end
            [this.lam, this.lamInv] = gpuVar(useGPU,precision,this.lam,this.lamInv);
        end
        
        
        function [lam,lamInv] = getEigs(this)
            lam         = eigLap1D(this.nt,this.h);
            lamInv      = 1./(lam+.5*min(lam(lam>0)));
%             lamInv(isnan(lamInv) | isinf(lamInv)) = min(lamInv(~isnan(lamInv)));
            lamInv(isnan(lamInv) | isinf(lamInv)) = min(lamInv(~isnan(lamInv)));
        end
        
        function Z = der(this,theta)
            theta = reshape(theta,[],this.nt);
            Z     = this.beta*vec(conv2(theta,[1 -1], 'valid'))/this.h;
        end
        
        function Y = derTranspose(this,Z)
            Z   = reshape(Z,[],this.nt-1);
            Y   = this.beta*vec(conv2(Z,[-1 1]))/this.h;
        end
        
        function PCop = getPCop(this)
            PCop = LinearOperator(this.n,this.n, @(x) PCmv(this,x), @(x) PCmv(this,x));
        end
        
        function y = PCmv(A,x,alpha,gamma)
            % x = argmin_x alpha/2*|A*x|^2+gamma/2*|x-y|^2
            % minimum norm solution when rank-deficient
            if not(exist('alpha','var')) || isempty(alpha)
                alpha = 1;
            end
            if not(exist('gamma','var')) || isempty(gamma)
                gamma = 0;
            end
            x    = reshape(x,[],A.nt);
            xhat = dctn(x,'dimFlag',[0 1]);
            s=1./ reshape((alpha*A.beta^2)*A.lam + gamma,1,[]);
            s(isnan(s))=0;
            s(isinf(s))=0;
            xhat = xhat.*s ;
            y    = vec(idctn(xhat,'dimFlag',[0,1]));
        end
        

        function this = set.beta(this,val)
           this.beta = val;
            this.Amv = @(x) der(this,x);
            this.ATmv = @(x)derTranspose(this,x);
        end
        
    end
end

