function [A,dA,d2A] = tanhActivation(Y,varargin)
% [A,dA] = tanhActivation(Y,varargin)
%
% hyperbolic tan activation function A = tanh(Y)
%
% Input:
%  
%   Y - array of features
%
% Optional Input:
%
%   doDerivative - flag for computing derivative, set via varargin
%                  Ex: tanhActivation(Y,'doDerivative',0);
%
% Output:
%
%  A  - activation
%  dA - derivatives

if nargin==0
    runMinimalExample;
    return
end

doDerivative = nargout>=2;
for k=1:2:length(varargin)    % overwrites default parameter
  eval([varargin{k},'=varargin{',int2str(k+1),'};']);
end;


dA = []; d2A = [];

A = tanh(Y);

if doDerivative 
     dA = 1-A.^2;
end

if doDerivative && nargout ==3
     d2A = -2*A.*dA;
end


function runMinimalExample
Y  = linspace(-3,3,101);
[A,dA,d2A] = feval(mfilename,Y);

fig = figure(100);clf;
fig.Name = mfilename;
plot(Y,A);
hold on;
plot(Y,dA);
plot(Y,d2A);
xlabel('y')
legend('tanh(y)','1-tanh(y)^2','2 tanh(x) (tanh(x)^2 - 1)')