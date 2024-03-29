classdef convMCNt < convKernel
    % classdef convMCNt < convKernel
    %
    % transpose 2D convolution using MatConvNet
    %
    % Transforms feature using affine linear mapping
    %
    %      Y(theta,Y0) K(Q*theta) * Y0 
    %
    %  where 
    % 
    %      K - convolution matrix
    
    
    properties
        pad
    end
    
    methods
        function this = convMCNt(varargin)
            this@convKernel(varargin{:});
            if nargout==0 && nargin==0
                this.runMinimalExample;
                return;
            end
            pad =[];
            for k=3:2:length(varargin)     % repeat so that MCN can get the pad
                eval([ varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            if not(isempty(pad))
                this.pad = pad;
            else
                this.pad = floor((this.sK(1)-1)/2);
            end
        end
        
        function runMinimalExample(~)
            nImg   = [16 18];
            sK     = [3 3,1,2];
            kernel = convMCN(nImg,sK,'stride',2,'pad',1);
            kernelt = feval(mfilename,nImg,sK,'stride',2,'pad',1);
            
            theta = rand(sK); 
            theta(:,1,:) = -1; theta(:,3,:) = 1;
            
            I  = rand([nImg sK(3)]); I(4:12,4:12,:) = 2;
            Ik = Amv(kernel,theta,I);
            Ik2 = ATmv(kernel,theta,Ik);
            Ikt = Amv(kernelt,theta,Ik);
            norm(Ik2(:)-Ikt(:),2)
            figure(1); clf;
            subplot(1,2,1);
            imagesc(I);
            title('input');
            
            subplot(1,2,2);
            imagesc(Ik(:,:,1));
            title('output');
        end
        function n = nImgIn(this)
            n =  [((this.nImg(1:2) - (this.sK(1:2)-1) + 2*this.pad))./this.stride this.sK(4)];
            n = ceil(n);
        end
        
        function n = nImgOut(this)
            n = [this.nImg(1:2) this.sK(3)];
        end
        
        function [Y,tmp] = ATmv(this,theta,Y)
            % compute convolution
            K = reshape(this.Q*theta(:),this.sK);
            Y = vl_nnconv(Y,K,[],'pad',this.pad,'stride',this.stride);
        end

        
        function dY = Jthetamv(this,dtheta,~,Y,~)
            dY = getOp(this,dtheta)*Y;
        end
        
        
        function dtheta = JthetaTmv(this,Y,~,Z,~)
            %  derivative of Z*(A(theta)*Y) w.r.t. theta
            [~,dtheta] = vl_nnconv(Y,zeros(this.sK,'like',Y), [],Z,'pad',this.pad,'stride',this.stride);
            dtheta = this.Q'*dtheta(:);
        end

       function dY = Amv(this,theta,Z)
            
            theta = reshape(this.Q*theta(:),this.sK);

             crop = this.pad * [1 1 1 1];
             if this.stride==2 && (this.sK(1)==3 || this.sK(1)==5)
                 crop = max(crop-[0 1 0 1],0);
             end
            dY = vl_nnconvt(Z,theta,[],'crop',crop,'upsample',this.stride);
            szY = sizeFeatOut(this);
            if (size(dY,1)==szY(1)-1) && (size(dY,2)==szY(2)-1)
                dY = padarray(dY,[1 1],0,'post');
            end
       end
    end
end

