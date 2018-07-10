classdef convKernel
    % classdef convKernel < handle
    %
    % Superclass for convolution kernels
    %
    % Transforms feature using affine linear mapping
    %
    %     Y(theta,Y0)  = K(theta) * Y0
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
        Q
        stride
        useGPU
        precision
    end
    
    methods
        function this = convKernel(nImg, sK,varargin)
            if nargin==0
                return;
            end
            nImg = nImg(1:2);
            stride = 1;
            useGPU = 0;
            Q =[];;
            precision = 'double';
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
            this.Q = Q;
        end
        
        function n = nFeatIn(this)
            n = prod(nImgIn(this));
        end
        function n = nFeatOut(this)
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
            if all(this.sK(1:2)==1)
                theta = randn(squeeze(this.sK));
                [U,S,V] = svd(squeeze(theta),'econ');
                s = min(1,diag(S));
                theta(1,1,:,:) = U*diag(s)*V';
            elseif this.sK(3)==this.sK(4)
                n = prod(this.sK([1,2]));
                e = zeros(prod(this.sK(1:2)),1);
                e(fix((prod(this.sK(1:2))+1)/2)) = 1;
                sd = sqrt(2/n);
                % put weights on diagonal only
                theta = sd*randn(prod(this.sK(1:3)),prod(this.sK(4)));
                mask  = kron(eye(this.sK(3)),ones(prod(this.sK(1:2)),1)) *0+0* kron(ones(this.sK(3)),e);
                theta = theta.*mask;
                id1 = find(theta>2*sd);
                theta(id1(:)) = randn(numel(id1),1);
                
                id2 = find(theta< -2*sd);
                theta(id2(:)) = randn(numel(id2),1);
                
                theta = max(min(2*sd, theta),-2*sd);
                
            else        
                n = prod(this.sK([1,2,4]));
                sd = sqrt(2/n);
                theta = sd*randn(this.nTheta(),1);
                id1 = find(theta>2*sd);
                theta(id1(:)) = randn(numel(id1),1);
                
                id2 = find(theta< -2*sd);
                theta(id2(:)) = randn(numel(id2),1);
                
                theta = max(min(2*sd, theta),-2*sd);
            end
            theta = gpuVar(this.useGPU,this.precision,theta);
            %            theta = randn(this.sK);
            %            if this.sK(3)==this.sK(4) && this.sK(4)>1
            %             theta = theta/sum(theta(:));
            %            end
        end
        
        function [thFine] = prolongateConvStencils(this,theta)
            % prolongate convolution stencils, doubling image resolution
            thFine = theta;
            if all(this.sK(1:2)==3)
                [WH,A,Q] = getFineScaleConvAlgCC([0;-1;0;0;0;0;0;1;0]);
                thFine = Q*(A\reshape(theta,9,[]));
                thFine = thFine(:);
            elseif any(this.sK(1:2)>1) && any(this.sK(1:2)~=3)
                error('nyi')
            end
        end
        function [thCoarse] = restrictConvStencils(this,theta)
            % restrict convolution stencils, dividing image resolution by two
            thCoarse = theta;
            if all(this.sK(1:2)==3)
                [WH,A,Q] = getFineScaleConvAlgCC([0;-1;0;0;0;0;0;1;0]);
                thCoarse = Q*(A*reshape(theta,9,[]));
                thCoarse = thCoarse(:);
            elseif any(this.sK(1:2)>1) && any(this.sK(1:2)~=3)
                error('nyi')
            end
        end
        
        % ------ handling GPU and precision -------
        function this = set.useGPU(this,value)
            if (value~=0) && (value~=1)
                error('useGPU must be 0 or 1.')
            else
                this.useGPU = value;
                this = gpuVar(this,this.useGPU,this.precision);
            end
        end
        function this = set.precision(this,value)
            if not(strcmp(value,'single') || strcmp(value,'double'))
                error('precision must be single or double.')
            else
                this.precision = value;
                this = gpuVar(this,this.useGPU,this.precision);
            end
        end
        
        function A = getOp(this,K)
            n   = nFeatIn(this);
            m   = nFeatOut(this);
            Af  = @(Y) this.Amv(K,Y);
            ATf = @(Y) this.ATmv(K,Y);
            A   = LinearOperator(m,n,Af,ATf);
        end
        
        function n = nTheta(this)
            n = size(this.Q,2);
        end
        
        function dY = Jthetamv(this,dtheta,~,Y,~)
            nex    =  numel(Y)/nFeatIn(this);
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

