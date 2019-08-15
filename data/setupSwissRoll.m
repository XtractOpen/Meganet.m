function[Yt,Ct,Yv,Cv] = setupSwissRoll(n)

if nargin == 0
    n = 256;
end

theta = [0:4*pi/(2*n):4*pi];
r     = linspace(0,1,length(theta));
theta = theta(:); r = r(:);

xr = r.*cos(theta); yr = r.*sin(theta);
r = linspace(0.2,1.2,length(theta)); 
r = r(:);
xb = r.*cos(theta); yb = r.*sin(theta);

n = length(xr);
Y = [[xr;xb],[yr;yb]];
C = [[ones(n,1); zeros(n,1)],[zeros(n,1); ones(n,1)]];

Yt = Y(1:2:end,:)';
Ct = C(1:2:end,:)';

Yv = Y(2:2:end,:)';
Cv = C(2:2:end,:)';