function[Y,C] = setupPeaksRegression(np)

if nargin==0
    runMinimalExample;
    return;
end

Y = 6*(rand(2,np)-0.5); % random points in [-3,3]^2
C = peaks(Y(1,:),Y(2,:));

end


function runMinimalExample
    [Y,C] = feval(mfilename,10000);
    figure(1); clf;
    scatter(Y(1,:),Y(2,:),20,C,'filled');
    
end
