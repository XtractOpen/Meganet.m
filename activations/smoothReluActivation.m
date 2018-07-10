function [A,dA] = smoothReluActivation(Y,varargin)
% [A,dA] = smoothReluActivation(Y,varargin)
%
% smoothed relu activation function A = smoothReluActivation(Y). The idea
% is to use a quadratic model on [-eta,eta] to ensure differentiability. 
%
% Input:
%  
%   Y - array of features
%
% Optional Input:
%
%   eta          - controls interval of quadratic, default=0.1
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

if isa(Y,'gpuArray')
    % use MATLAB's arrayfun to accelerate GPU computation and save on
    % memory
    
    fun = eval(sprintf('@(x) act(x,%e,%e,%e,%e)',eta,a,b,c)); % eval is clunky but arrayfun 
                                                              % had problems
                                                              % with additional inputs
    A = arrayfun(fun,Y);
    if doDerivative
        fun =eval(sprintf('@(x) dact(x,%e,%e,%e,%e)',eta,a,b,c));  
        dA = arrayfun(fun,Y);
    end
    
else
    % Y is not on the GPU, use regular MATLAB syntax which is much faster
    % here
    A              = max(Y,0);
    A(abs(Y)<=eta) = a.*Y(abs(Y)<=eta).^2 + b.*Y(abs(Y)<=eta) + c;
    
    if doDerivative
        dA              = sign(A);
        dA(abs(Y)<=eta) = 2*a*Y(abs(Y)<=eta) + b;
    end
    
end

function ac = act(y,eta,a,b,c)
% ac = act(y,eta,a,b,c)
%
% Called by MATLAB's arrayfun, computes the activation component-wise
if y<-eta
    ac = 0*y;
elseif y>=eta
    ac = y;
else
    ac = a*(y.*y)+b*y+c;
end

function ac = dact(y,eta,a,b,c)
% ac = act(y,eta,a,b,c)
%
% Called by MATLAB's arrayfun, computes the derivative of the activation component-wise
if y<-eta
    ac = 0*y;
elseif y>=eta
    ac=0*y+1;
else
    ac = 2*a*y+b;
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