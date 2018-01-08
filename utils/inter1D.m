function [thi,wi,idi]=  inter1D(theta,ttheta,ti)


% get theta for current time point
idth   = find(ttheta<=ti,1,'last');
if idth == numel(ttheta)
    thi = theta(:,end);
    wi  = [0 1];
    idi = [numel(ttheta)-1 numel(ttheta)];
else
    idi    = [idth idth+1];
    thl    = ttheta(idi(1));  thr = ttheta(idi(2));
    hth    = thr-thl;
    wi     = [thr-ti ti-thl]/hth;
    thi = wi(1) *theta(:,idth)  + wi(2) * theta(:,idth+1);
end
