function Ygpu = convCuDNN2D_conv(Xgpu,X_size,Kgpu,K_size,stride,session)
if nargin == 4
    stride = 1;
end
X_size = int32(X_size);
K_size = int32(K_size);
if nargin < 6 || isempty(session)
    Ygpu = convCuDNN2D_mex(Xgpu,X_size,Kgpu,K_size,int32(0),int32(stride));
else
    Ygpu = convCuDNN2D_mex(Xgpu,X_size,Kgpu,K_size,int32(0),int32(stride),session);
end
end