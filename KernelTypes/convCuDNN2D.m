classdef convCuDNN2D < convKernel
    % classdef convCuDNN < convKernel
    %
    % 2D convolution using CuDNN
    %
    % Transforms feature using affine linear mapping
    %
    %      Y(theta,Y0) K(theta) * Y0 
    %
    %  where 
    % 
    %      K - convolution matrix
    %
    % If initialized with a session, the descriptors and workspace memory
    % will "live" in the cudnnSession class, and reused in every application.
    % If the session remains empty, the descriptors and workspace memory
    % will be allocated and released in every convolution (this adds a bit 
    % of overhead, depends on the convolution size)
    
    properties
        session % a cudnn session. 
    end
    
    methods
        function this = convCuDNN2D(session,varargin)
            this@convKernel(varargin{:});
            if nargout==0 && nargin==0
                this.runMinimalExample;
                return;
            end
            if isa(session,'convCuDNN2DSession')||isempty(session)
                this.session = session;
            else
                warning('convCuDNN2D: session is not a convCuDNN2DSession and not an empty array.');
                this.session = [];
            end
            this.useGPU = 1;
            this.precision = 'single';
        end
        

        function runMinimalExample(~)
            nImg   = [16 18];
            sK     = [3 3,1,2];
            s = convCuDNN2DSession();
%             s = [];
            kernel = feval(mfilename,s,nImg,sK,'stride',1);
            kernel.stride = 1;
            
            theta = gpuArray(single(rand(sK))); 
            theta(:,1,:) = -1; theta(:,3,:) = 1;
            
            I   = gpuArray(single(rand(nImg))); I(4:12,4:12) = 2;
            Ik  = Amv(kernel,theta,I);
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
            nex   = numel(Y)/prod(nImgIn(this));
            % compute convolution
            Y   = reshape(Y,[nImgIn(this) nex]);
            Z = Y;
            K   = reshape(theta(:),this.sK);
            if  isempty(this.session)
                Y = convCuDNN2D_conv(Y,[nImgIn(this),nex],K,this.sK,this.stride);
            else
                Y = convCuDNN2D_conv(Y,[nImgIn(this),nex],K,this.sK,this.stride,this.session.sessionArray);
            end
%             Z = convCuDNN2D_conv(Z,[nImgIn(this),nex],K,this.sK,this.stride);
            Y    = reshape(Y,[],nex);            
            %%%%%%%% JUST TO TEST AGAINST MATCONVNET %%%%%%%%
%             Z = vl_nnconv(Z,K,[],'pad',(this.sK(1)-1)/2,'stride',this.stride);
%             Z = reshape(Z,[],nex);
%             
%             if norm(Z(:) - Y(:),1)/numel(Z) > 1e-3
%                 imagesc(squeeze(Ygpu(:,:,1,1))); figure; imagesc(squeeze(Znew(:,:,1,1)))
%                 error('problem!!')
%             end
%           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
        end

        function dY = Jthetamv(this,dtheta,~,Y,~)
            nex    =  numel(Y)/numelFeatIn(this);
            Y      = reshape(Y,[],nex);
            dY = getOp(this,dtheta)*Y;
        end
        
        function dtheta = JthetaTmv(this,Z,~,Y,~)
            %  derivative of Z*(A(theta)*Y) w.r.t. theta
            nex    =  numel(Y)/prod(nImgIn(this));
            Y      = reshape(Y,[nImgIn(this) nex]);
            Z      = reshape(Z,[nImgOut(this) nex]);
            % get derivative w.r.t. convolution kernels
            if  isempty(this.session)
                dtheta  = convCuDNN2D_dYdK_T(Y,[nImgIn(this) nex],Z,this.sK,this.stride);
            else
                dtheta  = convCuDNN2D_dYdK_T(Y,[nImgIn(this) nex],Z,this.sK,this.stride,this.session.sessionArray);
            end
            % dtheta = this.Q'*dtheta;
%             [~,dtheta] = vl_nnconv(Y,zeros(this.sK,'like',Y), [],Z,'pad',(this.sK(1)-1)/2,'stride',this.stride);
        end
        function n = nTheta(this)
            % n = size(this.Q,2); %%%%% TODO: remove Q
            n = prod(this.sK);
        end
       function dY = ATmv(this,theta,Z)
            nex     = numel(Z)/prod(nImgOut(this));
            Z       = reshape(Z,[nImgOut(this) nex]);
            K       = reshape(theta,this.sK);
            if  isempty(this.session)
                 dY  = convCuDNN2D_dYdX_T(K,[nImgIn(this) nex],Z,this.sK,this.stride);
            else
                 dY  = convCuDNN2D_dYdX_T(K,[nImgIn(this) nex],Z,this.sK,this.stride,this.session.sessionArray);
            end
            dY = reshape(dY,[],nex);
%             crop = (this.sK(1)-1)/2;
%             if this.stride==2
%                 crop=crop*[1,0,1,0];
%             end
%             Z = vl_nnconvt(Z,K,[],'crop',crop,'upsample',this.stride);
%             dY = reshape(dY,[],nex);
       end
    end
end

