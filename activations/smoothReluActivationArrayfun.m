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

fun = @(x) act(x,0.1,2.5,0.5,0.025);
A = arrayfun(fun,Y);
if doDerivative
    fun = @(x) dact(x,0.1,2.5,0.5,0.025);
    dA = arrayfun(fun,Y);
end

function ac = act(y,eta,a,b,c)
% to be called component-wise
if y<-eta
    ac = 0*y;
elseif y>=eta
    ac = y;
else
    ac = a*(y.*y)+b*y+c;
end

function ac = dact(y,eta,a,b,c)
% to be called component-wise
if y<-eta
    ac = 0*y;
elseif y>=eta
    ac=0*y+1;
else
    ac = 2*a*y+b;
end


function runMinimalExample
Y  = gpuArray(linspace(-.5,.5,101));
[A,dA] = feval(mfilename,Y);

fig = figure(100);clf;
fig.Name = mfilename;
plot(Y,A);
hold on;
plot(Y,dA);
xlabel('y')
legend('relu(y)','relu''(y)')
axis equal