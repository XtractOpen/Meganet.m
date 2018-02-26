function [isOK,Err,Order] = checkDerivative(fctn,x0,varargin)
% function [isOK,Err,Order] = checkDerivative(fctn,x0,varargin)
% 
% check derivative of fctn using its Taylor expansion.

if nargin==0;
    feval(mfilename,@quadTestFun,.4);
    return;
end

out      = 0;
tol      = 1.9;
nSuccess = 3;
v        = randn(size(x0));

for k=1:2:length(varargin)     % overwrites default parameter
    eval([varargin{k},'=varargin{',int2str(k+1),'};']);
end

if isa(fctn,'objFctn')
    fctn = @(x) eval(fctn,x);
end


if out
    fprintf('%9s\t%9s\t%9s\t%9s\t%9s\t%5s\n','h','E0','E1','O1','O2','OK?');
end
[F,dF] = fctn(x0);
if isstruct(dF) || numel(dF)~=numel(x0);
    [F,~,dF] = fctn(x0);
end

if size(dF,1)==size(v,1) || size(dF,1)==numel(v)
    dvF = dF'*v;
elseif size(dF,2)==size(v,1) || size(dF,2)==numel(v)
    warning('Assume, F returns Jacobian instead of gradient');
    dvF = dF*v;
else 
    error('cannot deal this derivative')
end

if norm(vec(dvF))/(norm(x0)+norm(x0)==0) < 1e-10
    warning('gradient is small');
end
nF     = norm(vec(F));
Err   = zeros(10,2);
Order   = zeros(10,2);
Success = zeros(10,1);
for j=1:30
    Ft = fctn(x0+2.0^(-j)*v);      % function value
    Err(j,1) = gather(norm(vec(F-Ft))/nF);    % Error TaylorPoly 0
    Err(j,2) = gather(norm(vec(F + 2.0^(-j)*dvF - Ft))/nF); % Error TaylorPoly 1
    if j>1
        Order(j,:) = log2(Err(j-1,:)./Err(j,:));
    end
    if (Order(j,2)>tol) || (Err(j,1)/Err(j,2) > 100); Success(j)=1; end
    if out
        fprintf('%1.3e\t%1.3e\t%1.3e\t%1.3e\t%1.3e\t%5d\n',...
            2.0^(-j), Err(j,1:2), Order(j,1:2),Success(j));
    end
end
isOK = sum(Success) > nSuccess;
end