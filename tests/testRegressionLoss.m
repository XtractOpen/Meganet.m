
pLoss = regressionLoss();

a = 3;
b = 2;
% Sour
Y = randn(500,2);
W = randn(10,501);
C = randn(10,2);
W  = W(:);

%% check calls of regression
[F,~,dWF,d2WF,dYF,d2YF] = getMisfit(pLoss,W,Y,C);
if not(numel(F)==1)
    error('first output argument of softMax must be a scalar')
end
if any(size(dWF)~=size(W))
    error('size of gradient dWF and W must match')
end
if any(numel(dYF)~=numel(Y))
    error('size of gradient dYF and Y must match')
end

%% check derivatives and Hessian w.r.t. W
W0 = randn(size(W));
dW = randn(size(W));

[F,~,dF,d2F] = getMisfit(pLoss,W0,Y,C);
dFdW = dF'*dW;
d2FdW = dW'*(d2F*dW);
err    = zeros(30,4);
for k=1:size(err,1)
    h = 2^(-k);
    Ft = getMisfit(pLoss,W0+h*dW,Y,C);
    
    err(k,:) = [h, norm(F-Ft), norm(F+h*dFdW-Ft), norm(F+h*dFdW+h^2/2*d2FdW-Ft)];
    
    fprintf('h=%1.2e\tE0=%1.2e\tE1=%1.2e\tE1=%1.2e\n',err(k,:))
end

figure; clf;
loglog(err(:,1),err(:,2),'-b','linewidth',3);
hold on;
loglog(err(:,1), err(:,3),'-r','linewidth',3);
hold on;
loglog(err(:,1), err(:,4),'-k','linewidth',3);
legend('E0','E1','E2');


%% check derivatives and Hessian w.r.t. Y
Y0 = randn(size(Y));
dY = randn(size(Y));

[F,~,~,~,dF,d2F] = getMisfit(pLoss,W,Y0,C);
dFdY = dF'*dY(:);
d2FdY = dY(:)'*(d2F*dY(:));
err    = zeros(30,4);
for k=1:size(err,1)
    h = 2^(-k);
    Ft = getMisfit(pLoss,W,Y0+h*dY,C);
    
    err(k,:) = [h, norm(F-Ft), norm(F+h*dFdY-Ft), norm(F+h*dFdY+h^2/2*d2FdY-Ft)];
    
    fprintf('h=%1.2e\tE0=%1.2e\tE1=%1.2e\tE1=%1.2e\n',err(k,:))
end

figure; clf;
loglog(err(:,1),err(:,2),'-b','linewidth',3);
hold on;
loglog(err(:,1), err(:,3),'-r','linewidth',3);
hold on;
loglog(err(:,1), err(:,4),'-k','linewidth',3);
legend('E0','E1','E2');