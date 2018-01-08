function[Yt,Ct,Yv,Cv] = setupSpiral(n)

if nargin == 0
    n = 256;
end
if nargin==0 && nargout==0
    runMinimalExample
    return
end

theta = [0:4*pi/(2*n):4*pi];
r     = linspace(0,1,length(theta));
theta = theta(:); r = r(:);

xr = r.*cos(theta); yr = r.*sin(theta);
% plot(xr,yr,'.r','markerSize',30);
hold on
r = linspace(0.2,1.2,length(theta)); 
r = r(:);
xb = r.*cos(theta); yb = r.*sin(theta);
% plot(xb,yb,'.b','markerSize',30);
% hold off

n = length(xr);
Y = [[xr;xb],[yr;yb]];
C = [[ones(n,1); zeros(n,1)],[zeros(n,1); ones(n,1)]];

Yt = Y(1:2:end,:)';
Ct = C(1:2:end,:)';

Yv = Y(2:2:end,:)';
Cv = C(2:2:end,:)';

function runMinimalExample
[Yt,Ct,Yv,Cv] = setupSpiral(200)
figure(1); clf;
subplot(1,2,1);
viewFeatures2D(Yt,Ct);
title('training data')
axis equal tight
subplot(1,2,2);
viewFeatures2D(Yv,Cv);
title('validation data');
axis equal tight
