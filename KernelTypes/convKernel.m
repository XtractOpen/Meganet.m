classdef convKernel  < handle
    % classdef convKernel < handle
    %
    % Superclass for convolution kernels
    %
    % Transforms feature using affine linear mapping
    %
    %     Y(theta,Y0)  = K(Q*theta) * Y0
    %
    %  where
    %
    %      K - convolution matrix (computed using FFTs for periodic bc)
    %
    % convKernels must provide at least the following methods
    %
    %  apply - evaluate transformation
    %  Jmv   - Jacobian*vector
    %  JTmv  - Jacobian'*vector
    
    properties
        nImg  % image size
        sK    % kernel size: [nxfilter,nyfilter,nInputChannels,nOutputChannels]
        Q     % parametrization of kernel, default = identity
        stride
        useGPU
        precision
    end
    
    methods
        function this = convKernel(nImg, sK , varargin)
            if nargin==0
                return;
            end
            nImg = nImg(1:2);
            stride = 1;
            useGPU = 0;
            precision = 'double';
            Q = [];
            for k=1:2:length(varargin)     % overwrites default parameter
                eval([ varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            if isempty(Q)
                Q = opEye(prod(sK));
            end
            this.nImg = nImg;
            this.sK   = sK;
            this.stride = stride;
            this.useGPU = useGPU;
            this.precision = precision;
            this.Q = gpuVar(useGPU,precision,Q);
        end
        
        function n = sizeFeatIn(this)
            n = nImgIn(this);
        end
        function n = sizeFeatOut(this)
            n = nImgOut(this);
        end
        function n = numelFeatIn(this)
            n = prod(nImgIn(this));
        end
        function n = numelFeatOut(this)
            n = prod(nImgOut(this));
        end
        
        function n = nImgIn(this)
            n = [this.nImg(1:2) this.sK(3)];
        end
        
        function n = nImgOut(this)
            n = [this.nImg(1:2)./this.stride this.sK(4)];
        end
        function this = gpuVar(this,useGPU,precision)
        end
        
        
        function theta = initTheta(this)
            n = size(this.Q,2);
            sd = sqrt(2/n);
            theta = sd*randn(this.nTheta(),1);
            id1 = find(theta>2*sd);
            theta(id1(:)) = randn(numel(id1),1);

            id2 = find(theta< -2*sd);
            theta(id2(:)) = randn(numel(id2),1);

            theta = max(min(2*sd, theta),-2*sd);
            
            theta = gpuVar(this.useGPU,this.precision,theta);
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
            thFine = theta;
            if isa(this.Q,'opEye') %%%%%% TODO
                if all(this.sK(1:2)==3)
                    [WH,A,Qp] = getFineScaleConvAlgCC([0;-1;0;0;0;0;0;1;0],'getRP',getRP);
                    thFine = A\(Qp\reshape(theta,9,[]));
                    thFine = thFine(:);
                elseif any(this.sK(1:2)>1) && any(this.sK(1:2)~=3)
                    error('nyi')
                end
            end
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
            
                
            if isa(this.Q,'opEye') %%%%%% TODO : remove?
                if all(this.sK(1:2)==3)
                    [WH,A,Qp] = getFineScaleConvAlgCC([0;-1;0;0;0;0;0;1;0],'getRP',getRP);
                    thCoarse = Qp*(A*reshape(theta,9,[]));
                    thCoarse = thCoarse(:);
                elseif any(this.sK(1:2)>1) && any(this.sK(1:2)~=3)
                    error('nyi')
                else
                    thCoarse = theta;
                end
            else
                thCoarse = theta;
            end
        end
        
        % ------ handling GPU and precision -------
        function set.useGPU(this,value)
            switch value
                case {1,0}
                    this.useGPU  = value;
                    [this.Q] = gpuVar(value,this.precision,this.Q);
                otherwise
                    error('useGPU must be 0 or 1.')
            end

        end
        function set.precision(this,value)
            if not(strcmp(value,'single') || strcmp(value,'double'))
                error('precision must be single or double.')
            else
                this.precision = value;
                [this.Q] = gpuVar(this.useGPU,value,this.Q);
            end
        end
        
        function A = getOp(this,K)
            n   = sizeFeatIn(this);
            m   = sizeFeatOut(this);
            Af  = @(Y) this.Amv(K,Y);
            ATf = @(Y) this.ATmv(K,Y);
            A   = LinearOperator(m,n,Af,ATf);
        end
        
        function n = nTheta(this)
            n = size(this.Q,2);
        end
        
        function dY = Jthetamv(this,dtheta,~,Y,~)
            nex    =  numel(Y)/numelFeatIn(this);
            Y      = reshape(Y,[],nex);
            dY = getOp(this,dtheta)*Y;
        end
        
        function Z = implicitTimeStep(this,theta,Y,h)
            %K = zeros(n1,n2);
            %K(1:sz1,1:sz2) = reshape(theta,sz1,sz2);
            %Kh = fft2(K);
            %V  = diag(vec((h*conj(Kh).*Kh + 1)));
            %W   = V\vec(fft2(Y)));
            %Z = real(ifft2(reshape(W,n1,n2));
            disp('NIY');
        end
        
        
        
    end
end

