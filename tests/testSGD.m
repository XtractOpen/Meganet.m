close all; clear all; clc;
nex    = 30; nf =20;
A      = abs(sprandn(nex,20,.1));
Av      = abs(sprandn(nex,20,.1));
x      = randn(nf,1);

Y      = A*x;
Yv    = Av*x;
pReg  = tikhonovReg(.001*speye(numel(x)));

f1 = lsObjFctn(A,Y,pReg);
fv = lsObjFctn(Av,Yv);

x0 = randn(size(x));
opt =newton('out',1,'maxIter',20,'atol',1e-10,'rtol',1e-10);
opt.linSol.maxIter = 20;
opt.linSol.tol = 1e-8;
[xn,his] = solve(opt,f1,x0,fv);

err =norm(xn(:)-x(:))

%%
miniBatch = 1;
learningRate = 0.001;
maxEpochs = 5000;


x0 = randn(size(x));
opt1 =sgd('out',1,'maxEpochs',maxEpochs,'miniBatch',miniBatch,'learningRate',learningRate,'momentum',0,'nesterov',false,'ADAM',false);
[xn1,his1] = solve(opt1,f1,x0,fv);
err1 = norm(xn1(:)-x(:));

opt2 =sgd('out',1,'maxEpochs',maxEpochs,'miniBatch',miniBatch,'learningRate',learningRate,'momentum',0.9,'nesterov',false,'ADAM',false);
[xn2,his2] = solve(opt2,f1,x0,fv);
err2 = norm(xn2(:)-x(:));

opt3 =sgd('out',1,'maxEpochs',maxEpochs,'miniBatch',miniBatch,'learningRate',learningRate,'momentum',.9,'nesterov',true,'ADAM',false);
[xn3,his3] = solve(opt3,f1,x0,fv);
err3 = norm(xn3(:)-x(:));


opt4 =sgd('out',1,'maxEpochs',maxEpochs,'miniBatch',miniBatch,'learningRate',learningRate,'momentum',.9,'nesterov',false,'ADAM',true);
[xn4,his4] = solve(opt4,f1,x0,fv);
err4 = norm(xn4(:)-x(:));

%% solve LS problem
xt = [A;pReg.B]\[Y;zeros(size(pReg.B,1),1)];

f1t = eval(f1,xt);
fvt = eval(fv,xt);

%%
figure(1); clf;
subplot(1,2,1)
semilogy(his1.his(:,2),'-','lineWidth',2);
hold on;
semilogy(his2.his(:,2),'lineWidth',2);
semilogy(his3.his(:,2),'--','lineWidth',2);
semilogy(his4.his(:,2),'lineWidth',2);
semilogy(his.his(:,2),'lineWidth',2);
semilogy(f1t*ones(size(his4.his,1),1),'-','lineWidth',2);
% legend('sgd','sgd+momentum','sgd+momentum+nest','ADAM')
title('training loss, 1/(2*N)*|A*x-y|^2')
xlabel('epochs')
set(gca,'FontSize',14)
axis tight
ax = axis;

subplot(1,2,2)
semilogy(his1.his(:,end),'-','lineWidth',2);
hold on;
semilogy(his2.his(:,end),'lineWidth',2);
semilogy(his3.his(:,end),'--','lineWidth',2);
semilogy(his4.his(:,end),'lineWidth',2);
semilogy(fvt*ones(size(his4.his,1),1),'--','lineWidth',2);

legend('sgd','sgd+momentum','sgd+momentum+nest','ADAM','ground truth','Location','SouthWest')
title('validation loss, 1/(2*N)*|Av*x-yv|^2')
xlabel('epochs')
axis(ax)
% axis tight
set(gca,'FontSize',14)
