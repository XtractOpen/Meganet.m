function ph = viewFeatures3D(Y,C,varargin)

if nargin==0
    runMinimalExample;
    return;
end
markerSize = 10;

for k=1:2:length(varargin)     % overwrites default parameter
    eval([varargin{k},'=varargin{',int2str(k+1),'};']);
end

nclass = size(C,1);
if size(Y,1)>3
    warning('Y has too many features. Using first three columns');
    Y = Y(1:3,:);
end

if nclass==1
    p1 = plot3(Y(1,C==1),Y(2,C==1),Y(3,C==1),'.','MarkerSize',markerSize);
    hold on;
    p2 = plot3(Y(1,C==0),Y(2,C==0),Y(3,C==0),'.','MarkerSize',markerSize);
    ph = [p1;p2];    
else
    ph = [];
    for k=1:nclass
        pk = plot3(Y(1,C(k,:)==1),Y(2,C(k,:)==1),Y(3,C(k,:)==1),'.','MarkerSize',markerSize);
        hold on;
        ph = [ph;pk];
    end
end
hold off

function runMinimalExample

[Y,C] = setupPeaks;
Y = [Y; 0*Y(1,:)];
figure(1); clf;
feval(mfilename,Y,C);
title('labeled points');
