
%%%%
% convCouple_dYdK_T computes the Jacobian transpose times vector product.
% Y(X,K) = K*X. K - convolution kernel, arranged in [w,h,cin,cout] format . X - image arranged by [w,h,c,n] format.
%%%%
function dYdX_T_times_dY = convCuDNN2D_dYdX_T(Kgpu,X_size,dYgpu,K_size,stride,session)
X_size = int32(X_size);
K_size = int32(K_size);
if nargin < 6 || isempty(session)
    dYdX_T_times_dY = convCuDNN2D_mex(Kgpu,X_size,dYgpu,K_size,int32(2),int32(stride));
else
    dYdX_T_times_dY = convCuDNN2D_mex(Kgpu,X_size,dYgpu,K_size,int32(2),int32(stride),session);
end
end