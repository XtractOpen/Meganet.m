function [thi,wi,idi]=  nnInter1D(theta,ttheta,ti)
% [thi,wi,idi]=  nnInter1D(theta,ttheta,ti)
% 
% Next neighbor 1D interpolation of vector-valued data. 
%
% Given theta(:,i), which is the data measured at time point ttheta(i),
% compute values at all time points ti(j) using next neighbor interpolation.
%
% Example: 
%   fctn   = @(x) [sin(x);cos(x);x.^2]
%   ttheta = linspace(0,1,11).^2; % points do not need to be equidistantly spaced
%   theta  = fctn(ttheta);
%   ti     = linspace(0,1,101);
%   thi    = nnInter1D(theta,ttheta,ti);
%   figure; plot(ttheta,theta,'o',ti,thi,'-')
%
% Inputs:
%
%  theta  - data matrix
%  ttheta - discretizatation points of data
%  ti     - interpolation points
%
% Outputs:
%
%  thi    - interpolated points at ti.
%  wi     - interpolation weights at each ti, used for derivative computation
%  idi    - indices of left and right data point for each ti, used for dervatives

if nargin==0
   runMinimalExample;
   return
end

theta = reshape(theta,[],numel(ttheta));

thi = zeros(size(theta,1),numel(ti),'like',theta);
wi  = zeros(1,numel(ti));
idi = zeros(1,numel(ti));

for k=1:numel(ti)
    % get theta for current time point
    idth   = find(ttheta<=ti(k),1,'last');
    if  isempty(idth)
        thi(:,k) = theta(:,1);
         wi(:,k) = 1;
        idi(:,k) = 1;
    elseif idth == numel(ttheta)
        thi(:,k) = theta(:,end);
        wi(:,k)  = 1;
        idi(:,k) = numel(ttheta);
    else
        idi(:,k)    = idth;
        wi(:,k)     = 1;
        thi(:,k) = theta(:,idth);
    end
end

function runMinimalExample
theta  = [ 2 4 2 6; 4 2 1 2];
ttheta = [ 1 2 4 5];
ti     = linspace(0,6,101);
thi    = feval(mfilename,theta,ttheta,ti);

figure(99); clf;
plot(ttheta,theta(1,:)','or',ttheta,theta(2,:),'sb');
hold on;
plot(ti,thi(1,:)','-r',ti,thi(2,:)','-b');

