function[Y0,C] = setupPeaks(np,nc)
%[Y0,C] = setupPeaks(np)
if not(exist('np','var')) || isempty(np)
    np = 8000;
end

if not(exist('nc','var')) || isempty(nc)
    nc = 5;
end
ns = 256;
[xx,yy,cc] = peaks(ns);
t1 = linspace(min(xx(:)),max(xx(:)),ns);
t2 = linspace(min(yy(:)),max(yy(:)),ns);

% Binarize it
mxcc = max(cc(:)); mncc = min(cc(:)); 
hc = (mxcc - mncc)/(nc);
ccb = zeros(size(cc));
for i=1:nc
    ii = find( (mncc + (i-1)*hc)< cc & cc <= (mncc+i*hc));
    ccb(ii) = i-1;
end

% draw same number of points per class
Y0 = [];
npc = ceil(np/nc);
for k=0:nc-1
   xk = [xx(ccb==k) yy(ccb==k)];
   inds = randi(size(xk,1),npc,1);
   
   Y0 = [Y0; xk(inds,:)];
end

Y0 = Y0';
C = kron(eye(nc),ones(npc,1))';


