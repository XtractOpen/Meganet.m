classdef convFFT < convKernel 
    % classdef convFFT < convKernel
    % 2D coupled convolutions. Computed using FFTs
    %
    % Transforms feature using affine linear mapping
    %
    %     Y(theta,Y0) =  K(Q*theta) * Y0 
    %
    %  where 
    % 
    %      K - convolution matrix (computed using FFTs for periodic bc)
    
    properties
        S % the eigenvalues
    end
    
    methods
        function this = convFFT(varargin)
            this@convKernel(varargin{:});
            if nargout==0 && nargin==0
                this.runMinimalExample;
                return;
            end
            this.S = gpuVar(this.useGPU, this.precision, getEigs(this));
            
        end
        function S = getEigs(this)
            S = zeros(prod(this.nImg),prod(this.sK(1:2)));
            for k=1:prod(this.sK(1:2))
                Kk = zeros(this.sK(1:2));
                Kk(k) = 1;
                Ak = getConvMatPeriodic(Kk,[this.nImg 1]);
                
                S(:,k) = vec(fft2( reshape(full(Ak(:,1)),this.nImg(1:2)) ));
            end
        end
        function this = gpuVar(this,useGPU,precision)
            if strcmp(this.precision,'double') && (isa(gather(this.S),'single'))
                this.S = getEigs(this);
            end
            this.S = gpuVar(useGPU,precision,this.S);
        end

        function runMinimalExample(~)
            nImg   = [16 16];
            sK     = [3 3,2,4];   % [nRows, nCols, nChanIn, nChanOut]
            % sK     = [1 1,4,4]; 
            
            kernel = feval(mfilename,nImg,sK);
            
            theta1 = rand(sK); 
            theta1(:,:,1:2,1:2) = -1; theta1(:,2,:,:) = 1; % vert edge detection
            theta1(:,:,:,4) = -1; theta1(2,:,:,3:4) = 1; % horix edge detection
            theta  = [theta1(:);];

            % create random input, but with a huge rectangle in the middle
            nex = 7;
            I  = rand([nImgIn(kernel) nex]); I(4:12,4:12,:,:) = 2;
            % Ik = reshape(Amv(kernel,theta,I),kernel.nImgOut);
            Ik = Amv(kernel,theta,I);
            ITk = ATmv(kernel,theta,Ik);
            
            % display how the learned filters perform
            for i=1:4
                figure(i); clf;
                subplot(1,2,1);
                imagesc(I(:,:,i));
                title('input');

                subplot(1,2,2);
                imagesc(Ik(:,:,i));
                title('output');
            end
            
            
            
        end
        
        function A = getMat(this,theta)
            A = [];
            theta = reshape(theta,this.sK);
            for i=1:this.sK(3)
                Acol = [];
                for j=1:this.sK(4)
                    Acol = [Acol; getConvMatPeriodic(flipdim(flipdim(theta(:,:,i,j),1),2),[vec(this.nImg(1:2)); 1]')];
                end
                A = [A Acol];
            end
        end
                        
            
        
        function Y = Amv(this,theta,Y)
            nex   = numel(Y)/prod(nImgIn(this));
            % nex = sizeLastDim(Y); % fails if nex=1
            % nex = size(Y, numel(nImgOut(this))+1);
            
            % compute convolution
            AY = zeros([nImgOut(this) nex],'like',Y); %start with transpose
            % theta reshaped to [nRows*nCols, nChanIn, nChanOut]
            theta    = reshape(this.Q*vec(theta), [prod(this.sK(1:2)),this.sK(3:4)]);
            Yh = ifft2(reshape(Y,[nImgIn(this) nex]));
            
            % for each one of the output channels,
            for k=1:this.sK(4) 
                Sk = reshape(this.S*theta(:,:,k),nImgIn(this)); % use eigs to perform conv
                T  = Sk .* Yh;
                AY(:,:,k,:)  = sum(T,3);
            end
            Y = real(fft2(AY));
        end
        
        function ATY = ATmv(this,theta,Z)
            nex =  numel(Z)/prod(nImgOut(this));
            ATY = zeros([nImgIn(this) nex],'like',Z); % start with transpose
            theta    = reshape(this.Q*vec(theta), [prod(this.sK(1:2)),this.sK(3:4)]);
            
            Yh = fft2(reshape(Z,[this.nImgOut nex]));
            for k=1:this.sK(3)
                tk = squeeze(theta(:,k,:));
                if size(this.S,2) == 1
                    tk = reshape(tk,1,[]);
                end
                Sk = reshape(this.S*tk,nImgOut(this));
                T  = Sk.*Yh;
                ATY(:,:,k,:) = sum(T,3);
            end
            ATY = real(ifft2(ATY));
        end
        
        function dY = Jthetamv(this,dtheta,~,Y,~)
            dY   = getOp(this,dtheta)*Y;
        end
        
        function dtheta = JthetaTmv(this,Z,~,Y)
            
            dth1    = zeros([this.sK(1)*this.sK(2),this.sK(3:4)],'like',Y);
            Y     = permute(Y,[1 2 4 3]);
            Yh    = reshape(fft2(Y),prod(this.nImg(1:2)),[]);
            Zh    = permute(ifft2(Z),[1 2 4 3]);
            Zh     = reshape(Zh,[], this.sK(4));
            
            for k=1:prod(this.sK(1:2)) % loop over kernel components
                temp = bsxfun(@times,conj(this.S(:,k)),Yh);
                temp = reshape(temp,[],this.sK(3));
                dth1(k,:,:) = conj(temp')*Zh;
            end
            dtheta = real(this.Q'*dth1(:));
        end
    end
end


