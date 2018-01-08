function [Y,dA] = reluActivation(Y,varargin)
% [A,dA] = reluActivation(Y,varargin)
%
% relu activation function A = relu(Y)
%
% Input:
%  
%   Y - array of features
%
% Optional Input:
%
%   doDerivative - flag for computing derivative, set via varargin
%                  Ex: reluActivation(Y,'doDerivative',0);
%
% Output:
%
%  A  - activation
%  dA - derivatives

if nargin==0
    runMinimalExample;
    return
end

doDerivative = nargout==2;
for k=1:2:length(varargin)    % overwrites default parameter
  eval([varargin{k},'=varargin{',int2str(k+1),'};']);
end


dA = [];

Y = max(Y,0);

if doDerivative
    dA = sign(Y);
end



function runMinimalExample
Y  = linspace(-3,3,101);
[A,dA] = feval(mfilename,Y);

fig = figure(100);clf;
fig.Name = mfilename;
plot(Y,A);
hold on;
plot(Y,dA);
xlabel('y')
legend('relu(y)','relu''(y)')