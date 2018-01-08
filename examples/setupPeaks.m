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

figure(1)
imagesc(t1,t2,reshape(ccb,ns,ns))
% keyboard
% keyboard
rng('default');

% draw same number of points per class
Y0 = [];
npc = ceil(np/nc);
for k=0:nc-1
   xk = [xx(ccb==k) yy(ccb==k)];
   inds = randi(size(xk,1),npc,1);
   
   Y0 = [Y0; xk(inds,:)];
end

C = kron(eye(nc),ones(npc,1));

% viewFeatures2D(Y0,C);
% 
% 
% inds = randi(numel(ccb),np,1);
% x1 = xx(inds);
% x2 = yy(inds);
% c  = ccb(inds);
% xb = [ x1(c==0) x2(c==0)];  % Blue points
% xr = [ x1(c==1) x2(c==1)];  % Red points
% xg = [ x1(c==2) x2(c==2)];  % Green points
% xm = [ x1(c==3) x2(c==3)];  % Black points
% xy = [ x1(c==4) x2(c==4)];  % Yellow points
% 
% [c,sp] = sort(c);
% 
% % make sure to have a balanced set
% lenb = size(xb,1);
% lenr = size(xr,1);
% leng = size(xg,1);
% lenm = size(xm,1);
% leny = size(xy,1);
% % make the problem balanced
% len = min([lenb;lenr;leng;lenm;leny]);
% xb = xb(1:len,:);
% xr = xr(1:len,:);
% xm = xm(1:len,:);
% xg = xg(1:len,:);
% xy = xy(1:len,:);
% 
% % figure(1);
% % hold on
% % plot(xb(:,1),xb(:,2),'ob');
% % plot(xr(:,1),xr(:,2),'or');
% % plot(xg(:,1),xg(:,2),'og');
% % plot(xm(:,1),xm(:,2),'ok');
% % plot(xy(:,1),xy(:,2),'oy');
% % hold off
% 
% Y0 = [xb;xr;xg;xm;xy];
% 
% C  = [[ones(len,1) zeros(len,4)]; ...
%       [zeros(len,1) ones(len,1) zeros(len,3)]; ...
%       [zeros(len,2) ones(len,1) zeros(len,2)]; ...
%       [zeros(len,3) ones(len,1) zeros(len,1)]; ...
%       [zeros(len,4) ones(len,1)]];
%   
% disp('Number of points per class');
% fprintf('%3d   %3d   %3d   %3d   %3d\n',lenb,lenr,leng,lenm,leny)  
