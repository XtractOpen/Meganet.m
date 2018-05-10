classdef opGrad < RegularizationOperator
    % gradient operator computed using conv2 (for fast GPU performance)
    
    properties
        nImg      % n=[nx1,nx2,nx3] number of pixels/voxels
        nChannels % number of channels
        h         % h=[hx1,hx2,hx3] pixel/voxel size
        beta      % regularization parameter
        lam       % eigenvalues of G'*G
        lamInv    % eigenvalues of pinv(G'*G)
    end
    
    methods
        function this = opGrad(nImg,nChannels,h,useGPU,precision)
            if not(exist('h','var')) || isempty(h)
                h = 1;
            end
            if not(exist('useGPU','var')) || isempty(useGPU)
                useGPU = 0;
            end
            if not(exist('precision','var')) || isempty(precision)
                precision = 'double';
            end
            this.nImg = nImg;
            this.h    = h;
            
            m = prod(this.nImg-[1,0])+prod(this.nImg-[0,1]);

            
            this.useGPU = useGPU;
            this.precision = precision;
            this.beta = 1;
            
            [lam,lamInv] = getEigs(this);
            [this.lam, this.lamInv] = gpuVar(useGPU,precision,lam,lamInv);
            
            this.nChannels = nChannels;
            
            this.m = nChannels*m;
            this.n = nChannels*prod(nImg);
            this.Amv = @(x) der(this,x);
            this.ATmv = @(x)derTranspose(this,x);
        end
        
        function Z = der(this,x)
            x     = reshape(x,[this.nImg,this.nChannels]);
            dx1   = reshape(convn(x,[1 -1]', 'valid')/this.h(1), [],this.nChannels);
            dx2   = reshape(convn(x,[1 -1], 'valid')/this.h(2),[],this.nChannels);
            Z     = [dx1;dx2];
            Z     = this.beta*vec(Z);
        end
        
        function [lam,lamInv] = getEigs(this)
            dim = numel(this.nImg);
            switch dim
                case 2
                    lam1 = eigLap1D(this.nImg(1),this.h(1));
                    lam2 = eigLap1D(this.nImg(2),this.h(2));
                    lam = lam1+lam2';
                otherwise
                    error('nyi');
            end
%             lam         = lam;% + .5*min(lam(lam>0));
            lamInv      = 1./lam;
            lamInv(isnan(lamInv) | isinf(lamInv)) = 0;
        end
        function this = convertGPUorPrecision(this,useGPU,precision)
            if strcmp(this.precision,'double') && (isa(this.lam,'single') || isa(this.lamInv,'single'))
                [this.lam,this.lamInv] = getEigs(this);
            end
            [this.lam, this.lamInv] = gpuVar(useGPU,precision,this.lam,this.lamInv);
        end
        
        
        function Y = derTranspose(this,Z)
            m1 = prod(this.nImg-[1,0]);
            Z  = reshape(Z,[],this.nChannels);
            Z1 = Z(1:m1,:); 
            Z1 = reshape(Z1,[this.nImg this.nChannels] -[1,0,0]);
            Z2 = Z(m1+1:end,:);
            Z2 = reshape(Z2,[this.nImg this.nChannels] -[0,1,0]);
            
            Y   =    vec(convn(Z1,[-1 1]'))/this.h(1)...
                   + vec(convn(Z2,[-1 1]))/this.h(2);
            Y   = this.beta*Y;
        end
        
        function PCop = getPCop(this)
            PCop = LinearOperator(this.n,this.n, @(x) PCmv(this,x), @(x) PCmv(this,x));
        end
        
        function y = PCmv(A,y,alpha,gamma)
            % x = argmin_x alpha/2*|A*x|^2+gamma/2*|x-y|^2
            if not(exist('alpha','var')) || isempty(alpha)
                alpha = 1;
            end
            if not(exist('gamma','var')) || isempty(gamma)
                gamma = 0;
            end
            % computes (beta^2*alpha*L'*L+gamma*I) \ x (used in preconditioning)
            y    = reshape(y,[A.nImg, A.nChannels]);
            
            xhat = dctn(y,'dimFlag',[1 1,0]);
            s = 1./(alpha*A.beta^2*A.lam+gamma);
            s(isnan(s))=0;
            s(isinf(s))=0;
            
            xhat = xhat.*s;
            y    = vec(idctn(xhat,'dimFlag',[1,1,0]));
        end
        
        function this = set.beta(this,val)
           this.beta = val;
            this.Amv = @(x) der(this,x);
            this.ATmv = @(x)derTranspose(this,x);
        end
        
    end
end

