function [Y0,C,Ytest,Ctest] = setupBox(varargin)
%[Y0,C,Ytest,Ctest] = setupPeaks(np)

ntrain = 100;
nval   = 100;

for k=1:2:length(varargin)    % overwrites default parameter
  eval([varargin{k},'=varargin{',int2str(k+1),'};']);
end;

% get training data

rb     = rand(ntrain,1);
thetab = rand(ntrain,1)*2*pi; 
xb1 = rb .* cos(thetab);
xb2 = .4*rb .* sin(thetab);
rr     = 1+rand(ntrain,1);
thetar = rand(ntrain,1)*2*pi; 
xr1 = rr .* cos(thetar);
xr2 = .4*rr .* sin(thetar);
Y0 = [[xb1; xr1],[xb2; xr2]];
C  = [ [ones(ntrain,1); zeros(ntrain,1)], [zeros(ntrain,1); ones(ntrain,1)]];

% get validation data
rb     = rand(nval,1);
thetab = rand(nval,1)*2*pi; 
xb1 = rb .* cos(thetab);
xb2 = .4*rb .* sin(thetab);
rr     = 1+rand(nval,1);
thetar = rand(nval,1)*2*pi; 
xr1 = rr .* cos(thetar);
xr2 = .4*rr .* sin(thetar);
Ytest = [[xb1; xr1],[xb2; xr2]];
Ctest = [ [ones(nval,1); zeros(nval,1)], [zeros(nval,1); ones(nval,1)]];

Y0 = Y0';
C  = C';
Ytest = Ytest';
Ctest = Ctest';
