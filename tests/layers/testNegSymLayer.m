% testing linearNegLayer - the old fasion way

Kernel = convBlkDiagFFT([32,32],[3,3,19]);
%layer = linearNegLayer(dense([24 14]));
layer = linearNegLayer(Kernel);
%layer = linearNegLayer(convMCN([1,1]));

theta = initTheta(layer);
%Y     = randn(14,7);
Y      = randn(32,32,19,100);
h     = 0.1; 

%% test the forwardProp function and its derivatives
[Z,Yt] = forwardProp(layer,theta,Y);

dtheta = randn(size(theta))*1e-3;
[Z1] = forwardProp(layer,theta+dtheta,Y);
dZ = Jthetamv(layer,dtheta,theta,Y,Yt);

disp('Derivative check')
fprintf('%3.2e   %3.2e\n\n\n',norm(vec(Z1)-vec(Z)),norm(vec(Z1)-vec(Z)-vec(dZ)))

%% 

[Z] = applyInv(layer,theta,Y,h);

dtheta = randn(size(theta))*1e-3;
[Z1] = applyInv(layer,theta+dtheta,Y,h);
dZ = iJthetamv(layer,dtheta,theta,Y,h);

disp('inverse Derivative check')
fprintf('%3.2e   %3.2e\n\n\n',norm(vec(Z1)-vec(Z)),norm(vec(Z1)-vec(Z)-vec(dZ)))

dW = randn(size(dZ));
dt = iJthetaTmv(layer,dW,theta,Y,[],h);

disp('inverse transpose check')
fprintf('%3.2e   %3.2e\n\n\n',dW(:)'*dZ(:),dtheta(:)'*dt(:))
