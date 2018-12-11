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
        function this = convCuDNN2D(varargin)
            this@convKernel(varargin{:});
            session = [];
            for k=1:2:length(varargin)     % overwrites default parameter
               eval([varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            
            if nargout==0 && nargin==0
                this.runMinimalExample;
                return;
            end
            if isa(session,'convCuDNN2DSession') || isempty(session)
                this.session = session;
            else
                warning('convCuDNN2D: session is not a convCuDNN2DSession and not an empty array.');
                this.session = [];
            end
            this.precision = 'single';
            this.Q = gpuVar(1,'single',this.Q);
        end
        

        function runMinimalExample(~)
            nImg   = [16 18];
            sK     = [3 3,1,2];
            s = convCuDNN2DSession();

            kernel = feval(mfilename,nImg,sK,'stride',1,'session',s);
            kernel.stride = 1;
            
            theta = gpuArray(single(rand(sK))); 
            theta(:,1,:) = -1; theta(:,3,:) = 1;
            
            I   = gpuArray(single(rand(nImg))); I(4:12,4:12) = 2;
            Ik  = Amv(kernel,theta,I);
            Ik2 = ATmv(kernel,theta,Ik);
            
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
            % compute convolution
            K   = reshape(this.Q*vec(theta),this.sK);
            nex = numel(Y)/numelFeatIn(this);
            if  isempty(this.session)
                Y = convCuDNN2D_conv(Y,[nImgIn(this),nex],K,this.sK,this.stride);
            else
                Y = convCuDNN2D_conv(Y,[nImgIn(this),nex],K,this.sK,this.stride,this.session.sessionArray);
            end
%             Z = convCuDNN2D_conv(Z,[nImgIn(this),nex],K,this.sK,this.stride);
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
            dY = getOp(this,dtheta)*Y;
        end
        
        function dtheta = JthetaTmv(this,Z,~,Y,~)
            %  derivative of Z*(A(theta)*Y) w.r.t. theta
            % get derivative w.r.t. convolution kernels
            nex = numel(Y)/numelFeatIn(this);
            if  isempty(this.session)
                dtheta  = convCuDNN2D_dYdK_T(Y,[nImgIn(this),nex],Z,this.sK,this.stride);
            else
                dtheta  = convCuDNN2D_dYdK_T(Y,[nImgIn(this),nex],Z,this.sK,this.stride,this.session.sessionArray);
            end
            dtheta = this.Q'*vec(dtheta);
%             [~,dtheta] = vl_nnconv(Y,zeros(this.sK,'like',Y), [],Z,'pad',(this.sK(1)-1)/2,'stride',this.stride);
        end
        function n = nTheta(this)
            n = size(this.Q,2); 
        end
       function dY = ATmv(this,theta,Z)
            K       = reshape(this.Q*vec(theta),this.sK);
            nex = numel(Z)/numelFeatOut(this);
            if  isempty(this.session)
                 dY  = convCuDNN2D_dYdX_T(K,[nImgIn(this) nex],Z,this.sK,this.stride);
            else
                 dY  = convCuDNN2D_dYdX_T(K,[nImgIn(this) nex],Z,this.sK,this.stride,this.session.sessionArray);
            end
%             crop = (this.sK(1)-1)/2;
%             if this.stride==2
%                 crop=crop*[1,0,1,0];
%             end
%             Z = vl_nnconvt(Z,K,[],'crop',crop,'upsample',this.stride);
%             dY = reshape(dY,[],nex);
       end

    end
end

