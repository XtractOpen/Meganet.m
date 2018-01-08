function [J,para,dJ,d2J,PC] = quadTestFun(x)
J   = 0.5*(x'*x);
dJ  = x;
d2J = speye(numel(x));
para = struct([]);
PC  = @(x) x;
end
