
%%%%
% convCouple_dYdK_T computes the Jacobian transpose times vector product.
% Y(X,K) = K*X. K - convolution kernel, arranged in [w,h,cin,cout] format . X - image arranged by [w,h,c,n] format.
%%%%
function dYdK_T_times_dY  = convCuDNN2D_dYdK_T(Xgpu,X_size,dYgpu,K_size,stride,session)
X_size = int32(X_size);
K_size = int32(K_size);

if nargin < 6 || isempty(session)
    dYdK_T_times_dY = convCuDNN2D_mex(Xgpu,X_size,dYgpu,K_size,int32(1),int32(stride));
else
    dYdK_T_times_dY = convCuDNN2D_mex(Xgpu,X_size,dYgpu,K_size,int32(1),int32(stride),session);
end
end