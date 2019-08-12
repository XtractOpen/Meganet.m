clear all; clc;

K     = dense([3 3]);
layer = singleLayer(K,'Bin',rand(3,1));
Y     = randn(3,10);
th = initTheta(layer);
dth = randn(size(th));
dY  = randn(size(Y));

% evaluate function
[Z,tmp]   = forwardProp(layer,th,Y);
dZ = randn(size(Z));
%% matrix free codes
t1mf = Jthetamv(layer,dth,th,Y,tmp);
t2mf = JYmv(layer,dY,th,Y,tmp);

s1mf = JthetaTmv(layer,dZ,th,Y,tmp);
s2mf = JYTmv(layer,dZ,th,Y,tmp);


%% matrix based code
[Jth,JY] = getJacobians(layer,th,Y,tmp);
t1mb = Jth*dth(:);
t2mb = JY*dY(:);
s1mb = Jth'*dZ(:);
s2mb = JY'*dZ(:);


errTh = norm(t1mb - t1mf(:))/norm(t1mf(:))
errY = norm(t2mb - t2mf(:))/norm(t2mf(:))
errTh = norm(s1mb - s1mf(:))/norm(s1mf(:))
errY = norm(s2mb - s2mf(:))/norm(s2mf(:))


%%
figure(1); clf;
subplot(1,2,1);
spy(Jth);
title('Jth');
subplot(1,2,2);
spy(JY);
title('JY');

