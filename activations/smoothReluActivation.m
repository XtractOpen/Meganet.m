function [A,dA] = smoothReluActivation(Y,varargin)
% [A,dA] = smoothReluActivation(Y,varargin)
%
% smoothed relu activation function A = smoothReluActivation(Y). The idea
% is to use a quadratic model close to the origin to ensure
% differentiability. The implementation here follows FAIR
%
% Input:
%  
%   Y - array of features
%
% Optional Input:
%
%   doDerivative - flag for computing derivative, set via varargin
%                  Ex: smoothReluActivation(Y,'doDerivative',0,'eta',.1);
%
% Output:
%
%  A  - activation
%  dA - derivatives

if nargin==0
    runMinimalExample;
    return
end
eta          = 0.1;
doDerivative = nargout==2;
for k=1:2:length(varargin)    % overwrites default parameter
  eval([varargin{k},'=varargin{',int2str(k+1),'};']);
end

a = 1/(4*eta);
b = 0.5;
c = 0.25*eta;

dA = [];

A              = max(Y,0);
A(abs(Y)<=eta) = a.*Y(abs(Y)<=eta).^2 + b.*Y(abs(Y)<=eta) + c;

if doDerivative
    dA              = sign(A);
    dA(abs(Y)<=eta) = 2*a*Y(abs(Y)<=eta) + b;
end



function runMinimalExample
Y  = linspace(-.5,.5,101);
[A,dA] = feval(mfilename,Y);

fig = figure(100);clf;
fig.Name = mfilename;
plot(Y,A);
hold on;
plot(Y,dA);
xlabel('y')
legend('relu(y)','relu''(y)')
axis equal