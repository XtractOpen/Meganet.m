
clear all; clc;

K     = dense([3 3]);
layer = singleLayer(K,'Bin',rand(3,1));
% layer.activation = @identityActivation;
Y     = randn(3,10);
th = initTheta(layer);
dth = randn(size(th));
D     = randn(size(Y));
% evaluate function
[Z,tmp]   = forwardProp(layer,th,Y);
res = Z - D;
f   = 0.5*res(:)'*res(:);
% gradient
df  = JthetaTmv(layer,res,th,Y,tmp);
dfdth = df'*dth;
% Hessian matvecs
d2fdth1 = dth'* JthetaTmv(layer,Jthetamv(layer,dth,th,Y,tmp),th,Y,tmp);
d2fdth2 = dth'*JthJthetaTmv(layer,dth,res,th,Y,tmp);
d2fdth = d2fdth1 + d2fdth2;

% [HKb1,HKb2] = getHessian(layer,res,eye(numel(res)),th,Y,tmp);


err = zeros(20,4);
for k=1:size(err,1)
    h = 2^(-k);
    tht = th + h*dth;
    
    % evaluate function
    Zt   = forwardProp(layer,tht,Y);
    rest = Zt - D;
    ft   = 0.5*rest(:)'*rest(:);
    
    err(k,1) = h;
    err(k,2) = abs(ft-f);
    err(k,3) = abs(ft-f-h*dfdth);
    err(k,4) = abs(ft-f-h*dfdth-.5*h^2*d2fdth);
    
    fprintf('%1.2e\t%1.2e\t%1.2e\t%1.2e\n',err(k,:))

end

figure(1); clf;
loglog(err(:,1),err(:,2),'DisplayName','err0');
hold on;
loglog(err(:,1),err(:,3),'DisplayName','err1');
loglog(err(:,1),err(:,4),'DisplayName','err2');
legend()

%% compare matrix-based and matrix-free code
x = randn(size(th));
H1xmf = JthetaTmv(layer,Jthetamv(layer,x,th,Y,tmp),th,Y,tmp);
H2xmf = JthJthetaTmv(layer,x,res,th,Y,tmp);

[H1,H2] = getHessian(layer,res,1.0,th,Y,tmp);
H1xmb = H1*x;
H2xmb = H2*x;
[norm(H1xmf - H1xmb)/norm(H1xmf) norm(H2xmf - H2xmb)/norm(H2xmf)]