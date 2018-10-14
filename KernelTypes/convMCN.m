classdef convMCN < convKernel
    % classdef convMCN < convKernel
    %
    % 2D convolution using MatConvNet
    %
    % Transforms feature using affine linear mapping
    %
    %      Y(theta,Y0) K(theta) * Y0 
    %
    %  where 
    % 
    %      K - convolution matrix
    
    
    properties
        pad
    end
    
    methods
        function this = convMCN(varargin)
            this@convKernel(varargin{:});
            if nargout==0 && nargin==0
                this.runMinimalExample;
                return;
            end
            this.pad    = floor((this.sK(1)-1)/2);
        end
        

        function runMinimalExample(~)
            nImg   = [16 18];
            sK     = [3 3,1,2];
            kernel = feval(mfilename,nImg,sK,'stride',2);
            kernel.pad = 0;
            theta = rand(sK); 
            theta(:,1,:) = -1; theta(:,3,:) = 1;
            
            I  = rand(nImg); I(4:12,4:12) = 2;
            Ik = Amv(kernel,theta,I);
            Ik2 = ATmv(kernel,theta,Ik);
            Ik = reshape(Ik,kernel.nImgOut());
            figure(1); clf;
            subplot(1,2,1);
            imagesc(I);
            title('input');
            
            subplot(1,2,2);
            imagesc(Ik(:,:,1));
            title('output');
        end
        
        function [Y,tmp] = Amv(this,theta,Y)
            tmp   = []; % no need to store any intermediates
            % nex   = numel(Y)/prod(nImgIn(this));
            nex = sizeLastDim(Y);
            
            % compute convolution
            Y   = reshape(Y,[nImgIn(this) nex]);  % unnecessary in the tensor scenario
            K   = reshape(this.Q*theta(:),this.sK);
            Y   = vl_nnconv(Y,K,[],'pad',this.pad,'stride',this.stride);
            Y   = reshape(Y,[],nex);
        end

        
        function dY = Jthetamv(this,dtheta,~,Y,~)
            nex    =  numel(Y)/prod(nFeatIn(this));
            Y      = reshape(Y,[],nex);
            dY = getOp(this,this.Q*dtheta(:))*Y;
        end
        
        
        function dtheta = JthetaTmv(this,Z,~,Y,~)
            %  derivative of Z*(A(theta)*Y) w.r.t. theta
            nex    =  numel(Y)/prod(nImgIn(this));
            Y      = reshape(Y,[nImgIn(this) nex]);
            Z      = reshape(Z,[nImgOut(this) nex]);
            % get derivative w.r.t. convolution kernels
            [~,dtheta] = vl_nnconv(Y,zeros(this.sK,'like',Y), [],Z,'pad',this.pad,'stride',this.stride);
            dtheta = this.Q'*dtheta(:);
        end

       function dY = ATmv(this,theta,Z)
            
            nex    =  numel(Z)/prod(nImgOut(this));
            Z      = reshape(Z,[nImgOut(this) nex]);
            theta = reshape(this.Q*theta(:),this.sK);

            crop = this.pad;
            if this.stride==2 && this.sK(1)==3
                crop=this.pad.*[1,0,1,0];
            elseif this.stride==2 && this.sK(1)==2
                crop=0*crop;
            end
            dY = vl_nnconvt(Z,theta,[],'crop',crop,'upsample',this.stride);
            if this.stride==2 && this.sK(1)==1
                dY = padarray(dY,[1 1],0,'post');
            end
            dY = reshape(dY,[],nex);
       end
    end
end

